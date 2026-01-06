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

    def process(self, graph: gs.Graph) -> gs.Graph:
        logger.debug("[O1]: 正在嗅探并融合连续的 ReLU...")

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
            # cleanup 会自动清除因为失去输出连接而变成死节点的“第二个 ReLU”
            graph.cleanup()
            logger.success(f"[O1]: 成功切除了 {fusion_count} 个冗余的恒等节点！")
        else:
            logger.debug("[O1]: 未发现连续的恒等节点，保持原样。")

        return graph
