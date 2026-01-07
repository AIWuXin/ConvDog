from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger
import onnx_graphsurgeon as gs

class GemmFusionPass(BasePass):
    """
    O1 级优化：将 MatMul + Add 融合为单个 Gemm 节点。
    公式：Y = alpha * A * B + beta * C
    """
    def process(self, graph: "gs.Graph") -> bool:
        changed = False
        # 在开始前进行清理，确保节点的 outputs (consumers) 索引是最新的
        graph.cleanup().toposort()

        # 1. 遍历图中所有节点，寻找 MatMul
        # 使用列表推导式创建快照，防止遍历过程中修改图导致迭代器失效
        for node in [n for n in graph.nodes if n.op == "MatMul"]:

            # --- [关键修复：秩检查] ---
            # 获取 MatMul 的两个输入 A 和 B
            A = node.inputs[0]
            B = node.inputs[1]

            # 核心逻辑：Gemm 只支持 2D 矩阵乘法。
            # Transformer 中的 MatMul 往往是 3D [Batch, Seq, Dim]，这种不能融合为 Gemm
            # 如果 shape 为 None (推断失败) 或 长度不等于 2，则跳过
            if A.shape is None or len(A.shape) != 2:
                continue
            if B.shape is None or len(B.shape) != 2:
                continue
            # ------------------------

            # 2. 检查 MatMul 的输出是否只连接到了一个节点
            if len(node.outputs) != 1:
                continue

            # 在 gs 中，Variable.outputs 存放的就是该张量的消费者节点 (consumers)
            consumers = node.outputs[0].outputs
            if len(consumers) != 1:
                continue

            consumer = consumers[0]

            # 3. 检查下游节点是否为 Add
            if consumer.op == "Add":
                # --- 进阶优化：检查 B 是否来自一个 Transpose 节点 ---
                trans_b = 0
                # 如果 B 的产生者是一个 Transpose 节点，且属性是 [1, 0]
                if len(B.inputs) == 1 and B.inputs[0].op == "Transpose":
                    trans_node = B.inputs[0]
                    if trans_node.attrs.get("perm") == [1, 0]:
                        # 直接吃掉 Transpose，将数据源上移，并在 Gemm 属性中标记 transB=1
                        B = trans_node.inputs[0]
                        trans_b = 1
                        logger.debug(f"[O1] 成功吃掉权重的 Transpose: {trans_node.name}")

                # 4. 寻找 Add node 中非 MatMul 输出的那个输入作为 Bias (C)
                C = None
                for inp in consumer.inputs:
                    if inp.name != node.outputs[0].name:
                        # 增加判断：Bias 应该是合法的常量或 Initializer
                        if isinstance(inp, gs.Constant) or (hasattr(inp, 'values')):
                            C = inp
                            break

                if C is None:
                    logger.debug(f"[O1] 跳过融合：找不到合法的 Bias 输入 {node.name}")
                    continue

                # 5. 创建新的 Gemm 节点
                gemm_node = gs.Node(
                    op="Gemm",
                    name=f"Fused_Gemm_{node.name}",
                    attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": trans_b},
                    inputs=[A, B, C],
                    outputs=consumer.outputs
                )

                # 6. 将新节点加入图，并排干旧节点的引用
                graph.nodes.append(gemm_node)

                # 这种“清空输出”的操作能让接下来的 cleanup() 自动剔除孤立的旧节点和中间张量
                node.outputs = []
                node.inputs = []
                consumer.outputs = []
                consumer.inputs = []

                changed = True
                logger.debug(f"[O1] 成功融合: {node.name} + {consumer.name} -> Gemm")

        # 7. 手术结束后再次清理图
        if changed:
            graph.cleanup().toposort()

        return changed
