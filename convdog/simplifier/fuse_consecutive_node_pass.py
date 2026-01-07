import onnx_graphsurgeon as gs
from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger

class FuseConsecutiveNodePass(BasePass):
    """
    O1 Pass: 融合连续的恒等算子。
    举例：如果一个 ReLU 的输入来自于另一个 ReLU 的输出，则跳过第二个 ReLU。
    """

    IDEMPOTENT_OPS = {
        "Relu": [],
        "LeakyRelu": ["alpha"],
        "Clip": ["min", "max"],
        "Abs": [],
        "Sign": [],
        "Floor": [],
        "Ceil": [],
        "Identity": [],
        "Dropout": []
    }

    def process_identity(self, graph: gs.Graph) -> gs.Graph:
        # 只处理f(f(x)) = f(x)的变换

        # 记录融合数量
        fusion_count = 0

        # 遍历图中所有节点
        for node in graph.nodes:
            if node.op in self.IDEMPOTENT_OPS:
                # 获取该节点的输入 Tensor
                input_tensor = node.inputs[0]
                cur_op = node.op

                # 检查产生该输入 Tensor 的上游节点是谁
                # input_tensor.inputs 返回的是产生该 tensor 的节点列表
                if len(input_tensor.inputs) > 0:
                    prev_node = input_tensor.inputs[0]

                    # 如果上游节点也是 ReLU
                    if prev_node.op == cur_op:
                        # 【核心手术】：将当前 ReLU 的所有输出，直接挂到上游 ReLU 的输入上
                        # 也就是：Relu1 -> TensorX -> Relu2 -> TensorY
                        # 变成：Relu1 -> TensorY

                        # 检查关键属性是否一致
                        attrs = self.IDEMPOTENT_OPS[node.op]
                        for attr in attrs:
                            if node.attrs.get(attr) != prev_node.attrs.get(attr):
                                break

                        # 把当前节点的所有后继消费者的输入，指向前一个节点的输出
                        # 在 graphsurgeon 中，最简单的方法是合并输出 Tensor
                        prev_node.outputs = node.outputs

                        # 清空当前节点的输出，方便后面 cleanup 自动回收
                        node.outputs = []
                        fusion_count += 1

        if fusion_count > 0:
            # cleanup 会自动清除因为失去输出连接而变成死节点的“第二个节点”
            graph.cleanup()
            logger.success(f"[O1]: 成功切除了 {fusion_count} 个冗余的恒等节点！")
        else:
            logger.debug("[O1]: 未发现连续的恒等节点，保持原样。")

        return graph

    def process_gemm(self, graph: gs.Graph) -> gs.Graph:
        fusion_count = 0
        for node in graph.nodes:
            if node.op == "Gemm":
                # 检查是否只连接一个算子
                if len(node.outputs) != 1:
                    continue

                # 查看是否为单消费者
                consumers = node.outputs[0].outputs
                if len(consumers) != 1:
                    continue

                consumer = consumers[0]
                if consumer.op != "Gemm":
                    continue

                can_fuse = isinstance(node.inputs[1], gs.Constant) and \
                           isinstance(consumer.inputs[1], gs.Constant)
                if len(consumer.inputs) == 3:
                    can_fuse = can_fuse and isinstance(consumer.inputs[2], gs.Constant)
                if len(node.inputs) == 3:
                    can_fuse = can_fuse and isinstance(node.inputs[2], gs.Constant)
                if can_fuse:
                    w1 = node.inputs[1].values
                    w2 = consumer.inputs[1].values
                    b1 = node.inputs[2].values if len(node.inputs) > 2 else 0
                    b2 = consumer.inputs[2].values if len(consumer.inputs) > 2 else 0
                    if node.attrs.get("transB", 0): w1 = w1.T
                    if consumer.attrs.get("transB", 0): w2 = w2.T
                    alpha1, beta1 = node.attrs.get("alpha", 1.0), node.attrs.get("beta", 1.0)
                    alpha2, beta2 = consumer.attrs.get("alpha", 1.0), consumer.attrs.get("beta", 1.0)

                    alpha = alpha1 * alpha2
                    beta = 1.0
                    a = node.inputs[0]
                    b = w1 @ w2
                    c = alpha2 * beta1 * (b1 @ w2) + beta2 * b2
                    trans_a = node.attrs.get("transB", 0)
                    trans_b = 0

                    # 创建融合后节点
                    b = gs.Constant(
                        "w1",
                        b
                    )
                    c = gs.Constant(
                        "b1",
                        c
                    )
                    gemm_node = gs.Node(
                        op="Gemm",
                        name=f"Fused_Gemm_{node.name}",
                        attrs={"alpha": alpha, "beta": beta, "transA": trans_a, "transB": trans_b},
                        inputs=[a, b, c],
                        outputs=consumer.outputs
                    )

                    graph.nodes.append(gemm_node)
                    node.outputs = []
                    node.inputs = []
                    consumer.outputs = []
                    consumer.inputs = []
                    graph.cleanup().toposort()
                    fusion_count += 1
                    logger.debug("[O1] 成功融合: {node.name} + {consumer.name} -> Gemm")

        return graph

    def process_non_identity(self, graph: gs.Graph) -> gs.Graph:
        graph = self.process_gemm(graph)
        return graph

    def process(self, graph: gs.Graph) -> gs.Graph:
        logger.debug("[O1]: 正在嗅探并融合连续重复的节点...")
        logger.debug("[O1]: 处理恒等变换节点......")
        graph = self.process_identity(graph)
        logger.debug("[O1]: 处理非恒等变换节点......")
        graph = self.process_non_identity(graph)

        return graph
