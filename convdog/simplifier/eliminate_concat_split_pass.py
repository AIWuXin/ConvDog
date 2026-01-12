import numpy as np
import onnx_graphsurgeon as gs
from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger

class EliminateConcatSplitPass(BasePass):
    """
    通用拓扑优化：消除冗余的汇聚-分发(Concentration-Distribution)结构。

    场景：
    分支[i] -> Unsqueeze/Reshape -> Concat -> [中间算子链(如Slice)] -> Split -> Squeeze/Reshape -> 下游[i]

    优化为：
    分支[i] -> [分支级中间算子链] -> 下游[i]
    """

    # 可分发的点对点算子
    POINTWISE_OPS = ["Relu", "Sigmoid", "Tanh", "Abs", "Exp", "Log", "Clip", "Cast", "Mul", "Add"]
    # 需要特殊修正 Axis 的算子
    AXIS_DEPENDENT_OPS = ["Slice", "Softmax", "ReduceSum", "Transpose"]

    def process(self, graph: gs.Graph) -> bool:
        changed = False
        graph.cleanup().toposort()

        for concat_node in [n for n in graph.nodes if n.op == "Concat"]:
            if len(concat_node.inputs) < 2: continue

            # --- 1. 向上探测：收集包装节点 ---
            branch_inputs = []
            wrapping_nodes = [] # 用于日志统计
            is_valid_pattern = True
            wrapping_op_type = None

            for inp_var in concat_node.inputs:
                parent = inp_var.inputs[0] if inp_var.inputs else None
                if parent and parent.op in ["Unsqueeze", "Reshape"]:
                    branch_inputs.append(parent.inputs[0])
                    wrapping_nodes.append(parent)
                    wrapping_op_type = parent.op
                else:
                    is_valid_pattern = False
                    break
            if not is_valid_pattern: continue

            # --- 2. 向下探测：穿透中间算子寻找 Split ---
            curr_node = concat_node
            intermediate_chain = []
            split_node = None

            while True:
                out_var = curr_node.outputs[0]
                if len(out_var.outputs) != 1: break # 必须单路

                next_node = out_var.outputs[0]
                if next_node.op == "Split":
                    split_node = next_node
                    break
                elif next_node.op in self.POINTWISE_OPS or next_node.op in self.AXIS_DEPENDENT_OPS:
                    intermediate_chain.append(next_node)
                    curr_node = next_node
                else:
                    break

            if not split_node: continue

            # --- 3. 匹配出口：是否有对应的逆向算子 ---
            exit_op_type = "Squeeze" if wrapping_op_type == "Unsqueeze" else "Reshape"
            exit_nodes = []
            is_exit_valid = True

            if len(split_node.outputs) != len(branch_inputs): continue

            for out_var in split_node.outputs:
                if len(out_var.outputs) == 1 and out_var.outputs[0].op == exit_op_type:
                    exit_nodes.append(out_var.outputs[0])
                else:
                    is_exit_valid = False
                    break
            if not is_exit_valid: continue

            # --- 4. 执行图手术：分发中间算子并修正 Axis ---
            for i in range(len(branch_inputs)):
                curr_var = branch_inputs[i]
                final_target_var = exit_nodes[i].outputs[0]

                for inter_node in intermediate_chain:
                    orig_dtype = inter_node.outputs[0].dtype
                    new_attrs = inter_node.attrs.copy()
                    new_inputs = [curr_var]

                    # 修正 Axis 偏差（从 4D 降阶到 3D）
                    if inter_node.op == "Slice":
                        if "axes" in new_attrs:
                            new_attrs["axes"] = [a - 1 if a > 0 else a for a in new_attrs["axes"]]
                        if len(inter_node.inputs) >= 4:
                            axes_const = inter_node.inputs[3]
                            if isinstance(axes_const, gs.Constant):
                                corrected_val = np.maximum(np.array(axes_const.values) - 1, 0).astype(np.int64)
                                axes_var = gs.Constant(name=f"{inter_node.name}_b{i}_axes", values=corrected_val)
                                new_inputs = [curr_var, inter_node.inputs[1], inter_node.inputs[2], axes_var]
                                if len(inter_node.inputs) > 4:
                                    new_inputs.append(inter_node.inputs[4])
                    elif inter_node.op in ["Softmax", "ReduceSum"]:
                        if "axis" in new_attrs:
                            new_attrs["axis"] = max(0, new_attrs["axis"] - 1)
                    else:
                        new_inputs += inter_node.inputs[1:]

                    branch_out_var = gs.Variable(name=f"{inter_node.name}_branch_{i}", dtype=orig_dtype)
                    graph.nodes.append(gs.Node(
                        op=inter_node.op,
                        name=f"{inter_node.name}_b{i}",
                        attrs=new_attrs,
                        inputs=new_inputs,
                        outputs=[branch_out_var]
                    ))
                    curr_var = branch_out_var

                # 重定向下游
                for consumer in list(final_target_var.outputs):
                    for idx, c_inp in enumerate(consumer.inputs):
                        if c_inp == final_target_var:
                            consumer.inputs[idx] = curr_var

                if final_target_var in graph.outputs:
                    idx = graph.outputs.index(final_target_var)
                    graph.outputs[idx] = curr_var

            # --- 5. 生成详细日志 (手术前提取名称) ---
            wrap_names = " + ".join([n.name for n in wrapping_nodes])
            inter_names = (" + ".join([n.name for n in intermediate_chain]) + " + ") if intermediate_chain else ""
            exit_names = " + ".join([n.name for n in exit_nodes])

            logger.debug(
                f"[O1/O2] 成功消除汇聚冗余结构: "
                f"({wrap_names}) -> {concat_node.name} -> {inter_names}{split_node.name} -> ({exit_names}) "
                f"-> 已简化为各分支独立路径"
            )

            # 6. 置空引用，触发自动化清理
            nodes_to_clear = [concat_node, split_node] + intermediate_chain + exit_nodes + wrapping_nodes
            for n in nodes_to_clear:
                n.outputs, n.inputs = [], []

            changed = True

        if changed:
            graph.cleanup().toposort()
        return changed
