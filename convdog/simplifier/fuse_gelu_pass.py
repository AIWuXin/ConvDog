import numpy as np
import onnx_graphsurgeon as gs

from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger


class FuseGeluPass(BasePass):
    """
    针对常见深度学习框架导出的 GELU 模式进行融合。
    匹配模式: x -> Div(sqrt(2)) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5) -> Gelu(x)
    """
    Custom_Op_Map = {
        "Gelu": "com.microsoft:Gelu"
    }

    def _is_constant_val(self, input_tensor, target, tolerance=1e-3):
        """ 辅助函数：检查张量是否为特定常数 """
        if isinstance(input_tensor, gs.Constant):
            val = input_tensor.values
            return np.all(np.abs(val - target) < tolerance)
        return False

    def process_gelu_fusion(self, graph: gs.Graph) -> bool:
        changed = False
        # 为了避免遍历时修改导致的迭代器问题，先固定节点列表
        # 我们以 Erf 算子作为锚点（Anchor），因为它在 Gelu 结构中特征最明显
        for node in [n for n in graph.nodes if n.op == "Erf"]:
            # --- 结构匹配逻辑 ---
            # 1. 回溯 Div: Erf <- Div
            div_node = node.inputs[0].inputs[0] if node.inputs[0].inputs else None
            if not div_node or div_node.op != "Div":
                continue

            # 校验 Div 的常数是否接近 sqrt(2) (约 1.414)
            if not any(self._is_constant_val(inp, 1.4142, 1e-2) for inp in div_node.inputs):
                continue

            # 2. 向下寻找 Add: Erf -> Add
            if len(node.outputs[0].outputs) != 1: continue
            add_node = node.outputs[0].outputs[0]
            if add_node.op != "Add" or not any(self._is_constant_val(inp, 1.0) for inp in add_node.inputs):
                continue

            # 3. 向下寻找第一个 Mul (合并分支): Add -> Mul_0
            if len(add_node.outputs[0].outputs) != 1: continue
            mul0_node = add_node.outputs[0].outputs[0]
            if mul0_node.op != "Mul": continue

            # 验证 Mul0 的另一个输入是否是最初的 x (即 Div 的输入)
            input_x = div_node.inputs[0]
            if input_x not in mul0_node.inputs:
                continue

            # 4. 向下寻找最后一个 Mul (缩放 0.5): Mul0 -> Mul_final
            if len(mul0_node.outputs[0].outputs) != 1: continue
            mul_final_node = mul0_node.outputs[0].outputs[0]
            if mul_final_node.op != "Mul" or not any(self._is_constant_val(inp, 0.5) for inp in mul_final_node.inputs):
                continue

            # --- 命中优化条件，开始执行手术 ---
            # 创建新的 Gelu 节点
            # 使用 Microsoft 的 Gelu 定义，输入是最初的 x
            gelu_out_var = gs.Variable(name=f"{mul_final_node.name}_fused_gelu_out")
            new_gelu_node = gs.Node(
                op="Gelu",
                name=f"{node.name}_fused_gelu",
                inputs=[input_x],
                outputs=[gelu_out_var],
                domain="convdog.ai" # 标记为自定义算子
            )
            graph.nodes.append(new_gelu_node)

            # 接管输出
            final_out_var = mul_final_node.outputs[0]
            for next_node in list(final_out_var.outputs):
                for i, inp in enumerate(next_node.inputs):
                    if inp == final_out_var:
                        next_node.inputs[i] = gelu_out_var

            # 清理旧节点连接
            for n in [div_node, node, add_node, mul0_node, mul_final_node]:
                n.inputs, n.outputs = [], []

            logger.debug(
                f"[O2] 成功发现并融合标准Gelu结构: "
                f"{input_x.name} + {div_node.name} + "
                f"{node.name} + {add_node.name} + "
                f"{mul0_node.name} + {mul_final_node.name} "
                f"-> {new_gelu_node.name}"
            )
            changed = True

        return changed

    def replace_custom_ops(self, graph: gs.Graph) -> None:
        """
        统一将内部标记的 Gelu 节点转换为目标后端支持的 domain
        """
        for node in graph.nodes:
            if node.op in self.Custom_Op_Map:
                target = self.Custom_Op_Map[node.op]
                domain, op = target.split(":")
                node.op = op
                node.domain = domain

    def process(self, graph: "gs.Graph") -> bool:
        """
        融合Gelu层。
        """
        graph.cleanup().toposort()

        # 1. 执行模式匹配与融合
        changed = self.process_gelu_fusion(graph)

        # 2. 如果发生了变化，执行替换和清理
        if changed:
            self.replace_custom_ops(graph)
            graph.cleanup().toposort()

        return changed
