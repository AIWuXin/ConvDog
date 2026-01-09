import onnx_graphsurgeon as gs
from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger

class FuseAttentionPass(BasePass):
    Custom_Op_Map = {"Attention": "com.microsoft:Attention"}

    def process(self, graph: "gs.Graph") -> bool:
        changed = False
        graph.cleanup().toposort()

        for softmax_node in [n for n in graph.nodes if n.op == "Softmax"]:
            try:
                # --- 1. 核心算子定位 (同前) ---
                score_matmul = softmax_node.inputs[0].inputs[0] # Q * K
                context_matmul = softmax_node.outputs[0].outputs[0] # Prob * V
                if score_matmul.op != "MatMul" or context_matmul.op != "MatMul": continue

                # --- 2. 深度溯源函数：通过形状和最大搜索范围定位真正的 Input ---
                def find_true_origin(start_node, depth_limit=12):
                    curr = start_node
                    collected_in_branch = []

                    # 定义真正源头的特征：
                    # 1. 形状包含了 [..., 3, Heads, ...] 或者是图的输入
                    # 2. 或者是那个 3 分支的起始点 (Gather 的上一层)
                    for _ in range(depth_limit):
                        if curr not in collected_in_branch:
                            collected_in_branch.append(curr)

                        # 如果当前节点是 Gather 且向上是 Transpose(3,...)，这通常就是源头
                        if curr.op == "Gather":
                            parent = curr.inputs[0].inputs[0]
                            return parent, collected_in_branch

                        # 检查是否有输入，没有则到头了
                        if not curr.inputs or not curr.inputs[0].inputs:
                            break
                        curr = curr.inputs[0].inputs[0]
                    return curr, collected_in_branch

                # 分别从 Q, K, V 的输入点向上追溯
                origin_q, branch_q = find_true_origin(score_matmul.inputs[0].inputs[0])
                origin_k, branch_k = find_true_origin(score_matmul.inputs[1].inputs[0])

                # 寻找 V 分支输入
                v_inputs = [inp.inputs[0] for inp in context_matmul.inputs if inp != softmax_node.outputs[0]]
                if not v_inputs: continue
                origin_v, branch_v = find_true_origin(v_inputs[0])

                # --- 3. 强校验：Q, K, V 必须汇聚于同一个源头 ---
                if not (origin_q == origin_k == origin_v):
                    # 如果不相等，说明有的分支还没追溯够，或者这根本不是 Self-Attention
                    continue

                common_source = origin_q

                # 汇总所有要清理的节点
                all_fused_nodes = [score_matmul, softmax_node, context_matmul]
                all_fused_nodes.extend(branch_q)
                all_fused_nodes.extend(branch_k)
                all_fused_nodes.extend(branch_v)

                # 向下找最后的输出转置（可选，有的模型没有）
                final_node = context_matmul.outputs[0].outputs[0]
                if final_node.op == "Transpose":
                    all_fused_nodes.append(final_node)
                    final_out_var = final_node.outputs[0]
                else:
                    final_out_var = context_matmul.outputs[0]

                # 从QKV源头向上寻找投影部分
                prev_node = common_source.inputs[0].inputs[0]
                found_bias, found_weights = False, False
                if prev_node.op == "Reshape":
                    all_fused_nodes.append(prev_node)
                    add_node = prev_node.inputs[0].inputs[0]
                    next_var = [var for var in add_node.inputs if isinstance(var, gs.Variable)][0]
                    matmul_node = next_var.inputs[0]
                else:
                    if prev_node.op != "Add":
                        continue
                    add_node = prev_node
                    matmul_node = add_node.inputs[0].inputs[0]
                if add_node.op == "Add":
                    found_bias = True
                    all_fused_nodes.append(add_node)
                if matmul_node.op == "MatMul":
                    found_weights = True
                    all_fused_nodes.append(matmul_node)
                if not found_bias or not found_weights:
                    continue

                # --- 4. 执行融合手术 ---
                # 统一获取 num_heads (从 common_source 那个汇合转置节点提取)
                num_heads = 1
                if common_source.op == "Transpose" and "perm" in common_source.attrs:
                    if len(common_source.outputs[0].shape) >= 3:
                        num_heads = common_source.outputs[0].shape[2]

                input_x = matmul_node.inputs[0] # Reshape 之前的原始数据
                attention_weight = [var for var in matmul_node.inputs if isinstance(var, gs.Constant)][0]
                attention_bias = [var for var in add_node.inputs if isinstance(var, gs.Constant)][0]

                new_attn_node = gs.Node(
                    op="Attention",
                    name=f"Fused_Attention_at_{softmax_node.name}",
                    inputs=[input_x, attention_weight, attention_bias],
                    outputs=[gs.Variable(name=f"{final_out_var.name}_fused")],
                    attrs={"num_heads": int(num_heads)},
                    domain="convdog.ai"
                )
                graph.nodes.append(new_attn_node)

                # 下游接管
                for next_node in list(final_out_var.outputs):
                    for i, inp in enumerate(next_node.inputs):
                        if inp == final_out_var:
                            next_node.inputs[i] = new_attn_node.outputs[0]

                # --- 5. 日志与清理 ---
                # 去重日志节点名
                unique_names = []
                unique_nodes = []
                for n in all_fused_nodes:
                    if n.name not in unique_names:
                        unique_names.append(n.name)
                        unique_nodes.append(n)

                logger.debug(f"[O2] 成功识别并融合 Attention 算子链: " +
                             f"{' + '.join(sorted([f'{n.name}({n.op})' for n in unique_nodes]))}")

                for n in unique_nodes:
                    n.inputs.clear()
                    n.outputs.clear()

                changed = True

            except (IndexError, AttributeError, TypeError):
                continue

        if changed:
            self.replace_custom_ops(graph)
            graph.cleanup().toposort()

        return changed

    def replace_custom_ops(self, graph: gs.Graph) -> None:
        """ 保持原有的 Custom_Op_Map 转换逻辑 """
        for node in graph.nodes:
            if node.op in self.Custom_Op_Map:
                node_op = self.Custom_Op_Map[node.op]
                domain, op = node_op.split(":")
                node.op = op
                node.domain = domain
                node.version = 1
