import numpy as np
import onnx_graphsurgeon as gs

from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger

class FuseLayerNormPass(BasePass):
    """
    O2 级优化：按照非线性拓扑结构融合 LayerNorm。
    拓扑结构：
       [Input X] ----+-----> [ReduceMean] (RM1)
                     |            |
                     +-----> [Sub] <------- (分子计算)
                               |
                +--------------+------ concrete result (分子)
                |              |
              [Pow]            |
                |              |
          [ReduceMean] (RM2)   |
                |              |
            [Add eps]          |
                |              |
             [Sqrt]            |
                |              |
                +-----> [Div] <+------- (归一化)
                         |
                       [Mul] (Gamma)
                         |
                       [Add] (Beta)
    """

    def _is_constant_val(self, input_tensor, target, tolerance=1e-6):
        if isinstance(input_tensor, gs.Constant):
            return np.all(np.abs(input_tensor.values - target) < tolerance)
        return False

    def process(self, graph: "gs.Graph") -> bool:
        changed = False
        logger.debug("[O3]: 正在融合LayerNorm......")
        graph.cleanup().toposort()

        # 1. 以第一个 ReduceMean 为锚点尝试匹配
        for rm1_node in [n for n in graph.nodes if n.op == "ReduceMean"]:
            try:
                # --- [拓扑校验阶段] ---
                # 获取原始输入 X
                input_x = rm1_node.inputs[0]

                # 2. 寻找 Sub：必须同时连接 input_x 和 rm1_node 的输出
                sub_node = None
                for consumer in rm1_node.outputs[0].outputs:
                    if consumer.op == "Sub" and input_x in consumer.inputs:
                        sub_node = consumer
                        break
                if not sub_node: continue

                # 3. 寻找 Pow 和 Div：它们都是 sub_node 的消费者
                pow_node = None
                div_node = None
                for consumer in sub_node.outputs[0].outputs:
                    if consumer.op == "Pow": pow_node = consumer
                    if consumer.op == "Div": div_node = consumer

                if not pow_node or not div_node: continue
                if not self._is_constant_val(pow_node.inputs[1], 2.0): continue

                # 4. 沿着 Pow 向下寻找 RM2 -> Add(eps) -> Sqrt
                if len(pow_node.outputs[0].outputs) != 1: continue
                rm2_node = pow_node.outputs[0].outputs[0]
                if rm2_node.op != "ReduceMean": continue

                if len(rm2_node.outputs[0].outputs) != 1: continue
                add_eps_node = rm2_node.outputs[0].outputs[0]
                if add_eps_node.op != "Add": continue

                if len(add_eps_node.outputs[0].outputs) != 1: continue
                sqrt_node = add_eps_node.outputs[0].outputs[0]
                if sqrt_node.op != "Sqrt": continue

                # 5. 校验 Sqrt 的输出是否回到了刚才找到的那个 Div
                if div_node not in sqrt_node.outputs[0].outputs: continue

                # 6. 继续寻找最后的 Mul 和 Add
                if len(div_node.outputs[0].outputs) != 1: continue
                mul_node = div_node.outputs[0].outputs[0]
                if mul_node.op != "Mul": continue

                if len(mul_node.outputs[0].outputs) != 1: continue
                add_beta_node = mul_node.outputs[0].outputs[0]
                if add_beta_node.op != "Add": continue

                # --- [参数提取阶段] ---
                eps_val = float(add_eps_node.inputs[1].values.item()) if isinstance(add_eps_node.inputs[1], gs.Constant) else 1e-5
                gamma = mul_node.inputs[1]
                beta = add_beta_node.inputs[1]

                # --- [熔炼阶段] ---
                logger.debug(f"[O2] 发现非线性 LayerNorm 结构，开始融合...")

                ln_node = gs.Node(
                    op="LayerNormalization",
                    name=f"Fused_LN_at_{rm1_node.name}",
                    inputs=[input_x, gamma, beta],
                    outputs=[add_beta_node.outputs[0]],
                    attrs={"epsilon": eps_val, "axis": -1}
                )
                graph.nodes.append(ln_node)

                # --- [日志与清理] ---
                logger.debug(
                    f"[O2] 成功融合算子链: "
                    f"{rm1_node.name} + {sub_node.name} + {pow_node.name} + {rm2_node.name} + "
                    f"{add_eps_node.name} + {sqrt_node.name} + {div_node.name} + "
                    f"{mul_node.name} + {add_beta_node.name} -> {ln_node.name}"
                )

                # 断开所有参与融合节点的连接，以便 cleanup 清理
                for n in [
                    rm1_node, sub_node, pow_node,
                    rm2_node, add_eps_node, sqrt_node,
                    div_node, mul_node, add_beta_node
                ]:
                    n.outputs = []
                    n.inputs = []

                changed = True

            except (IndexError, AttributeError):
                continue

        if changed:
            graph.opset = 17 if graph.opset < 17 else graph.opset
            for entry in graph.import_domains:
                if entry.domain == "" or entry.domain == "ai.onnx":
                    if entry.version < 17:
                        logger.warning(f"[O2] 检测到 Opset {entry.version} 不支持 LayerNorm，已自动强升至 17")
                        entry.version = 17
            graph.cleanup().toposort()

        return changed
