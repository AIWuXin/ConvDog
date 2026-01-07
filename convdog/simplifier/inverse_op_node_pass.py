import onnx_graphsurgeon as gs

from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger


class InverseOpCancellationPass(BasePass):
    """
    O1 Pass: 清除互逆的算子。
    举例：如果检测到先squeeze(1)再unsqueeze(1)的操作，则直接清除
    """
    Inverse_Ops = {
        "Squeeze": "Unsqueeze",
        "Unsqueeze": "Squeeze",
        "Concat": "Split",
        "Split": "Concat",
    }

    def process_squeeze_unsqueeze(self, graph: gs.Graph) -> bool:
        changed = False
        fusion_count = 0

        for node in graph.nodes:
            if node.op in ["Squeeze", "Unsqueeze"]:
                op_type = node.op
                if len(node.inputs) < 2:
                    continue
                node_axis = node.inputs[1].values.item()
                # 检查是否只连接一个算子
                if len(node.outputs) != 1:
                    continue

                # 查看是否为单消费者
                consumers = node.outputs[0].outputs
                if len(consumers) != 1:
                    continue

                consumer = consumers[0]
                if consumer.op != self.Inverse_Ops[op_type]:
                    continue

                consumer_axis = consumer.inputs[1].values.item()
                can_dce = consumer_axis == node_axis

                if can_dce:
                    prev_var = node.inputs[0]
                    next_var = consumer.outputs[0]
                    for next_node in next_var.outputs:
                    # 找到这个消费者节点中，哪个输入位置是 T3，然后把它换成 T1
                        for i, node_input in enumerate(next_node.inputs):
                            if node_input == next_var:
                                next_node.inputs[i] = prev_var
                    node.inputs = []
                    node.outputs = []
                    consumer.inputs = []
                    consumer.outputs = []

                    graph.cleanup().toposort()
                    fusion_count += 1
                    changed = True
                    logger.debug(f"[O1] 成功消除: {node.name} + {consumer.name} -> None")

        return changed

    def process_split_concat(self, graph: gs.Graph) -> bool:
        changed = False
        fusion_count = 0

        # 使用 list(graph.nodes) 避免在迭代时修改图结构导致的迭代器失效
        for node in list(graph.nodes):
            if node.op == "Split":
                # 1. 获取 Split 的 Axis (通常在 attrs 里)
                split_axis = node.attrs.get("axis", 0)
                split_outputs = node.outputs

                if len(split_outputs) == 0:
                    continue

                # 2. 检查 Split 的所有输出是否都只去往同一个消费者，且该消费者是 Concat
                # 先看第一个输出的第一个消费者
                first_out_consumers = split_outputs[0].outputs
                if len(first_out_consumers) != 1:
                    continue

                consumer = first_out_consumers[0]
                if consumer.op != "Concat":
                    continue

                # 3. 核心校验：
                # a) Concat 的 axis 必须与 Split 一致
                # b) Concat 的输入列表必须完全等于 Split 的输出列表（顺序和数量都要一致）
                # c) Split 的每个输出张量都不能有第二个消费者（旁路）
                concat_axis = consumer.attrs.get("axis", 0)
                if split_axis != concat_axis:
                    continue

                # 检查是否满足“原路拆解原路合并”：Concat的输入就是Split的输出
                if consumer.inputs != split_outputs:
                    continue

                # 检查是否有旁路消费者（这步非常重要，否则删了 Split 会导致其他支路断路）
                has_side_consumer = any(len(out.outputs) > 1 for out in split_outputs)
                if has_side_consumer:
                    continue

                # --- 命中优化条件：执行 A -> Split -> Concat -> B  =>  A -> B ---
                prev_var = node.inputs[0]   # Split 的输入
                next_var = consumer.outputs[0] # Concat 的输出

                # 将 Concat 之后的所有消费者，重新接到 Split 之前的张量上
                for next_node in list(next_var.outputs):
                    for i, node_input in enumerate(next_node.inputs):
                        if node_input == next_var:
                            next_node.inputs[i] = prev_var

                # 彻底断开旧节点的连接，以便 cleanup
                node.inputs = []
                node.outputs = []
                consumer.inputs = []
                consumer.outputs = []

                # 如果有必要，手动把原本 next_var 的元数据同步给 prev_var
                # 但通常 Split->Concat 不改变 shape/dtype，直接连即可

                graph.cleanup().toposort()
                fusion_count += 1
                changed = True
                logger.debug(f"[O1] 成功消除冗余链路: {node.name} (Split) + {consumer.name} (Concat) -> None")

        return changed

    def process(self, graph: "gs.Graph") -> bool:
        logger.debug("[O1]: 正在消除互逆算子......")
        graph.cleanup().toposort()
        changed0 = self.process_squeeze_unsqueeze(graph)
        changed1 = self.process_split_concat(graph)
        changed = changed0 or changed1

        return changed
