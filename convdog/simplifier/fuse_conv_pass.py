import numpy as np
import onnx_graphsurgeon as gs

from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger


class FuseConvPass(BasePass):
    Custom_Op_Map = {
        "ConvRelu": "com.microsoft:FusedConv"
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

    def process_conv1_convk(
            self, conv1: gs.Node,
            consumer: gs.Node
    ) -> None:
        """
        处理 Conv(1x1, S=1) -> Conv(KxK, S=S2) 的融合。
        融合后的结果是一个 KxK 卷积，继承第二层的 Stride 和 Padding。
        """
        # --- 1. 提取权重和偏置 ---
        # w1: (CO1, CI1, 1, 1)
        # w2: (CO2, CO1, K, K)
        w1 = conv1.inputs[1].values
        w2 = consumer.inputs[1].values

        # 提取偏置，若无则初始化为 0
        b1 = conv1.inputs[2].values if len(conv1.inputs) > 2 else np.zeros(w1.shape[0], dtype=w1.dtype)
        b2 = consumer.inputs[2].values if len(consumer.inputs) > 2 else np.zeros(w2.shape[0], dtype=w2.dtype)

        # --- 2. 核心数学融合逻辑 ---

        # A. 权重融合 (Weight Fusion)
        # 公式: W_new[o, i, h, w] = sum_k( W2[o, k, h, w] * W1[k, i, 0, 0] )
        # 将 w1 重塑为 (CO1, CI1) 以便进行矩阵乘法/einsum
        w1_linear = w1.reshape(w1.shape[0], w1.shape[1])
        # 使用 einsum 实现跨通道合并：o=CO2, k=CO1, i=CI1, h=K, w=K
        w_new = np.einsum('okhw, ki -> oihw', w2, w1_linear)

        # B. 偏置融合 (Bias Fusion)
        # 公式: b_new = Conv2(b1) + b2.
        # 由于 b1 在空间上是常数，等价于: b_new = (W2在空间维度求和) @ b1 + b2
        w2_spatial_sum = np.sum(w2, axis=(2, 3)) # 形状: (CO2, CO1)
        b_new = (w2_spatial_sum @ b1) + b2

        # --- 3. 属性继承 ---
        # 1x1 卷积不改变感受野和空间尺寸，因此融合后的算子完全继承第二层的空间属性
        new_pads = consumer.attrs.get("pads", [0, 0, 0, 0])
        new_strides = consumer.attrs.get("strides", [1, 1])
        new_dilations = consumer.attrs.get("dilations", [1, 1])

        # --- 4. 更新图结构 ---
        # 更新权重
        conv1.inputs[1].values = w_new

        # 更新或注入偏置
        if len(conv1.inputs) > 2:
            conv1.inputs[2].values = b_new
        else:
            bias_const = gs.Constant(name=f"{conv1.name}_fused_bias", values=b_new)
            conv1.inputs.append(bias_const)

        # 更新 Conv1 的属性使之接管 Conv2 的职能
        conv1.attrs["pads"] = new_pads
        conv1.attrs["strides"] = new_strides
        conv1.attrs["dilations"] = new_dilations
        # kernel_shape 会由下游框架根据权重 shape 自动推导，或手动指定
        if "kernel_shape" in consumer.attrs:
            conv1.attrs["kernel_shape"] = consumer.attrs["kernel_shape"]

        # --- 5. 拓扑接管 (Wiring) ---

        # 获取 Conv2 的输出变量
        conv2_out_var = consumer.outputs[0]
        conv1_out_var = conv1.outputs[0]

        # 同步 shape/dtype 信息
        conv1_out_var.shape = conv2_out_var.shape
        conv1_out_var.dtype = conv2_out_var.dtype

        # 将所有原本消费 Conv2 的节点，重定向至消费 Conv1
        for next_node in list(conv2_out_var.outputs):
            for i, node_input in enumerate(next_node.inputs):
                if node_input == conv2_out_var:
                    next_node.inputs[i] = conv1_out_var

        # 断开 Conv2 节点的连接，以便 graph.cleanup() 彻底移除它
        consumer.inputs = []
        consumer.outputs = []

        logger.debug(f"[O2] 成功融合Conv(1x1->KxK): {conv1.name} + {consumer.name} -> {conv1.name}")

    def process_conv1_conv1(
            self, conv1: gs.Node,
            consumer: gs.Node
    ) -> None:
        # --- 1. 提取权重与形状 ---
        w1 = conv1.inputs[1].values  # (CO1, CI1, 1, 1)
        w2 = consumer.inputs[1].values # (CO2, CO1, 1, 1)

        # 使用 reshape 替代 squeeze，安全转成 (Out, In) 的 2D 矩阵
        co2, co1_w2 = w2.shape[0], w2.shape[1]
        co1_w1, ci1 = w1.shape[0], w1.shape[1]

        w1_mat = w1.reshape(co1_w1, ci1)
        w2_mat = w2.reshape(co2, co1_w2)

        # --- 2. 权重融合 (Matrix Multiplication) ---
        # (CO2, CO1) @ (CO1, CI1) -> (CO2, CI1)
        w_new_mat = w2_mat @ w1_mat
        # 重新打回 4D Tensor 形状: (CO2, CI1, 1, 1)
        w_new = w_new_mat.reshape(co2, ci1, 1, 1)

        # --- 3. 偏置融合 ---
        b1 = conv1.inputs[2].values if len(conv1.inputs) > 2 else np.zeros(co1_w1, dtype=w1.dtype)
        b2 = consumer.inputs[2].values if len(consumer.inputs) > 2 else np.zeros(co2, dtype=w2.dtype)

        # b_new = W2 * b1 + b2
        b_new = (w2_mat @ b1) + b2

        # --- 4. 属性继承 ---
        # 1x1 -> 1x1 融合，Padding 恒定为 0 (ONNX 默认为 [0,0,0,0])
        new_pads = [0, 0, 0, 0]
        # 步长继承自第二层
        new_strides = consumer.attrs.get("strides", [1, 1])
        new_dilations = consumer.attrs.get("dilations", [1, 1])

        # --- 5. 图结构更新 ---
        conv1.inputs[1].values = w_new
        if len(conv1.inputs) > 2:
            conv1.inputs[2].values = b_new
        else:
            # 注入新常量作为 Bias
            bias_const = gs.Constant(name=f"{conv1.name}_fused_bias", values=b_new)
            conv1.inputs.append(bias_const)

        # 更新 Conv1 的关键属性
        conv1.attrs["pads"] = new_pads
        conv1.attrs["strides"] = new_strides
        conv1.attrs["dilations"] = new_dilations
        conv1.attrs["kernel_shape"] = [1, 1]

        # --- 6. 下游节点接管 ---
        conv2_out = consumer.outputs[0]
        conv1_out = conv1.outputs[0]

        # 同步元数据
        conv1_out.shape = conv2_out.shape
        conv1_out.dtype = conv2_out.dtype

        # 将 Conv2 的消费者全部指向 Conv1
        for next_node in list(conv2_out.outputs):
            for i, inp in enumerate(next_node.inputs):
                if inp == conv2_out:
                    next_node.inputs[i] = conv1_out

        # 清除 Conv2（consumer）的连接，等待 cleanup
        consumer.inputs, consumer.outputs = [], []

        logger.debug(f"[O2] 成功融合Conv(1x1->1x1): {conv1.name} + {consumer.name} -> {conv1.name}")

    def process_conv_conv(self, graph: gs.Graph) -> bool:
        changed = False
        for node in graph.nodes:
            if node.op != "Conv": continue
            if len(node.outputs) != 1 or len(node.outputs[0].outputs) != 1: continue

            conv1 = node
            consumer = conv1.outputs[0].outputs[0]
            if consumer.op != "Conv": continue

            # --- 约束检查 ---
            k1 = conv1.attrs.get("kernel_shape", [1, 1])
            k2 = consumer.attrs.get("kernel_shape", [1, 1])
            s1 = conv1.attrs.get("strides", [1, 1])
            s2 = consumer.attrs.get("strides", [1, 1])
            # O2 只处理常规卷积
            cur_size0 = k1[0]
            for k in k1:
                if k != cur_size0:
                    continue

            cur_size1 = k2[0]
            for k in k2:
                if k != cur_size1:
                    continue

            cur_stride0 = s1[0]
            for s in s1:
                if s != cur_stride0:
                    continue

            cur_stride1 = s2[0]
            for s in s2:
                if s != cur_stride1:
                    continue

            if not (isinstance(conv1.inputs[1], gs.Constant) and isinstance(consumer.inputs[1], gs.Constant)):
                continue

            # --- 判断情况 ---
            if cur_size0 == 1 and cur_size1 == 1 and cur_stride0 == 1:
                self.process_conv1_conv1(
                    conv1,
                    consumer
                )
            if cur_size0 == 1 and cur_size1 > 1 and cur_stride0 == 1:
                self.process_conv1_convk(
                    conv1,
                    consumer
                )

            graph.cleanup().toposort()
            changed = True

        return changed

    def replace_custom_ops(self, graph: gs.Graph) -> None:
        """
        替换自定义算子为微软算子。
        """
        for node in graph.nodes:
            if node.op in self.Custom_Op_Map:
                node_op = self.Custom_Op_Map[node.op]
                domain, op = node_op.split(":")
                if domain == "No":
                    domain = None
                node.op = op
                node.domain = domain  # 变成微软算子域
                node.version = 1

    def process(self, graph: "gs.Graph") -> bool:
        """
        融合卷积层。
        """

        graph.cleanup().toposort()
        logger.debug("[O2]: 正在融合Conv+Relu......")
        changed0 = self.process_conv_relu(graph)
        logger.debug("[O2]: 正在融合Conv+Conv......")
        changed1 = self.process_conv_conv(graph)
        changed = changed0 or changed1

        return changed
