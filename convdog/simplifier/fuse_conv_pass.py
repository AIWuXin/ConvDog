from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger
import onnx_graphsurgeon as gs
import onnx.onnx_pb


class FuseConvPass(BasePass):
    Custom_Op_Map = {
        "ConvRelu": "FusedConv"
    }


    def process_conv_relu(self, graph: gs.Graph) -> bool:
        changed = False
        fusion_count = 0

        for node in graph.nodes:
            if node.op == "Conv":
                # 1. 检查 Conv 是否只有一个输出张量
                if len(node.outputs) != 1:
                    continue

                # 2. 检查输出张量是否只有一个消费者（单消费者检查）
                consumers = node.outputs[0].outputs
                if len(consumers) != 1:
                    continue

                consumer = consumers[0]
                # 3. 检查消费者是否为 ReLU
                if consumer.op != "Relu":
                    continue

                # --- 命中优化条件：执行算子融合 (Conv + ReLU -> Fused Conv) ---

                # 4. 仿照 ONNX Runtime，通过属性注入实现融合
                # 设置 activation 为 Relu，并将 auto_pad 显式设为 NOTSET（表示 Padding 已固化）
                node.op = "ConvRelu"
                node.domain = "convdog.ai"  # 标记注入自定义算子
                node.attrs["activation"] = "Relu"
                node.attrs["auto_pad"] = "NOTSET"

                # 5. 接管输出：
                # Conv 原本的输出：T_conv
                # ReLU 原本的输出：T_relu
                # 我们让后续所有原本消费 T_relu 的节点，全部改去消费 T_conv
                conv_out_var = node.outputs[0]
                relu_out_var = consumer.outputs[0]

                # 同步元数据信息，确保图的一致性
                conv_out_var.shape = relu_out_var.shape
                conv_out_var.dtype = relu_out_var.dtype

                # 找到 T_relu 的后续所有消费者
                for next_node in list(relu_out_var.outputs):
                    for i, node_input in enumerate(next_node.inputs):
                        if node_input == relu_out_var:
                            next_node.inputs[i] = conv_out_var

                # 6. 断开ReLU节点的连接，以便 cleanup 清理节点
                consumer.inputs = []
                consumer.outputs = []

                # 7. 刷新图结构
                graph.cleanup().toposort()
                fusion_count += 1
                changed = True
                logger.debug(f"[O2] 成功融合: {node.name} + {consumer.name} -> {node.name}")

        return changed

    def replace_custom_ops(self, graph: gs.Graph) -> None:
        """
        替换自定义算子为微软算子。
        """
        for node in graph.nodes:
            if node.op in self.Custom_Op_Map:
                node.op = self.Custom_Op_Map[node.op]
                node.domain = "com.microsoft"  # 变成微软算子域
                node.version = 1

    def process(self, graph: "gs.Graph") -> bool:
        """
        融合卷积层。
        """

        logger.debug("[O2]: 正在融合Conv+Relu......")
        graph.cleanup().toposort()
        changed0 = self.process_conv_relu(graph)
        changed = changed0

        return changed
