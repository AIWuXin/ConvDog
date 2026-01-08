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

    def process_conv_conv(self, graph: gs.Graph) -> bool:
        changed = False
        fusion_count = 0

        # 遍历图中所有节点
        for node in graph.nodes:
            # 1. 检查当前节点是否为卷积
            if node.op != "Conv":
                continue

            # 2. 拓扑检查：Conv1 必须只有一个输出，且该输出只有一个消费者
            if len(node.outputs) != 1 or len(node.outputs[0].outputs) != 1:
                continue

            conv1_out_var = node.outputs[0]
            consumer = conv1_out_var.outputs[0]

            # 3. 检查消费者是否为第二个卷积 (Conv2)
            if consumer.op != "Conv":
                continue

            # --- O2 级别优化约束 ---
            # 4. 步长约束：Conv1 的 stride 必须为 1 (O2 只做 S1 融合)
            strides1 = node.attrs.get("strides", [1, 1])
            if any(s != 1 for s in strides1):
                continue

            # 5. 静态权重检查：融合需要操作权重，权重必须是 Initializer (gs.Constant)
            if not (isinstance(node.inputs[1], gs.Constant) and
                    isinstance(consumer.inputs[1], gs.Constant)):
                continue

            # --- 命中优化条件：执行卷积融合 (Conv1 + Conv2 -> Fused Conv) ---

            # 6. 提取算子参数
            w1 = node.inputs[1].values  # (CO1, CI1, K1, K1)
            w2 = consumer.inputs[1].values # (CO2, CO1, K2, K2)

            # 提取偏置，若无则初始化为 0
            b1 = node.inputs[2].values if len(node.inputs) > 2 else np.zeros(w1.shape[0], dtype=w1.dtype)
            b2 = consumer.inputs[2].values if len(consumer.inputs) > 2 else np.zeros(w2.shape[0], dtype=w2.dtype)

            # 提取 Conv2 的属性用于融合计算
            pads2 = consumer.attrs.get("pads", [0, 0, 0, 0])
            strides2 = consumer.attrs.get("strides", [1, 1])
            dilations2 = consumer.attrs.get("dilations", [1, 1])

            # 目前主要处理 Group=1 的经典卷积合并
            if node.attrs.get("group", 1) != 1 or consumer.attrs.get("group", 1) != 1:
                continue

            # --- 7. 数学核心：Numpy 实现权重与偏置融合 ---

            # A. 融合权重 W_new = Conv2(W1)
            # 处理 Padding: 对 W1 的空间维度进行补齐
            p = pads2[0] # 简化处理，取一侧 padding
            if p > 0:
                w1_padded = np.pad(w1, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
            else:
                w1_padded = w1

            CO2, CO1, K2, _ = w2.shape
            _, CI1, _, _ = w1_padded.shape
            K_new_h = w1_padded.shape[2] - K2 + 1
            K_new_w = w1_padded.shape[3] - K2 + 1

            w_new = np.zeros((CO2, CI1, K_new_h, K_new_w), dtype=w1.dtype)

            # 爱因斯坦求和约定：模拟多通道卷积
            # 我们遍历空间位置，计算 W2 与 W1 滑窗的内积
            for h in range(K_new_h):
                for w in range(K_new_w):
                    roi = w1_padded[:, :, h:h+K2, w:w+K2]
                    # 'okxy, kixy -> oi': 对 CO1(k), K2_H(x), K2_W(y) 进行约减
                    w_new[:, :, h, w] = np.einsum('okxy,kixy->oi', w2, roi)

            # B. 融合偏置 b_new = W2 * b1 + b2 (安全版：避开卷积核尺寸限制)
            # 原理：b1 在空间上具有平移不变性，Conv2 作用在 b1 上等价于 W2 的空间权重和乘以 b1
            w2_spatial_sum = np.sum(w2, axis=(2, 3)) # (CO2, CO1)
            b_new = (w2_spatial_sum @ b1) + b2

            # --- 8. 图结构接管与更新 ---

            # 更新 Conv1 的权重
            node.inputs[1].values = w_new

            # 更新或新增 Conv1 的偏置
            if len(node.inputs) > 2:
                node.inputs[2].values = b_new
            else:
                # 若原本没有偏置项，构造一个新的 Constant 插入
                bias_const = gs.Constant(name=node.name + "_fused_bias", values=b_new)
                node.inputs.append(bias_const)

            # 继承 Conv2 的核心属性
            node.attrs["strides"] = strides2
            node.attrs["dilations"] = dilations2
            node.attrs["pads"] = [0, 0, 0, 0] # 空间偏移已在融合权重的计算中消耗

            # 获取输出变量
            conv2_out_var = consumer.outputs[0]

            # 同步元数据
            conv1_out_var.shape = conv2_out_var.shape
            conv1_out_var.dtype = conv2_out_var.dtype

            # 找到 T_conv2 的后续所有消费者，全部改去消费 T_conv1
            for next_node in list(conv2_out_var.outputs):
                for i, node_input in enumerate(next_node.inputs):
                    if node_input == conv2_out_var:
                        next_node.inputs[i] = conv1_out_var

            # 9. 断开 Conv2 节点的连接，准备清理
            consumer.inputs = []
            consumer.outputs = []

            # 10. 刷新图结构
            graph.cleanup().toposort()
            fusion_count += 1
            changed = True
            logger.debug(f"[O2] 融合成功: {node.name} + {consumer.name} -> {node.name}")

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

        logger.debug("[O2]: 正在融合Conv+Relu......")
        graph.cleanup().toposort()
        changed0 = self.process_conv_relu(graph)
        changed1 = self.process_conv_conv(graph)
        changed = changed0 or changed1

        return changed
