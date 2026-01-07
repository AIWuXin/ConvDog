from typing import Optional
from convdog.core.graph import ConvDogModel
from convdog.utils.logger import logger
from convdog.simplifier.fuse_consecutive_node_pass import FuseConsecutiveNodePass
from convdog.simplifier.dead_code_elimination_pass import DeadCodeEliminationPass
from convdog.simplifier.inverse_op_node_pass import InverseOpCancellationPass
from convdog.simplifier.fuse_gemm_pass import GemmFusionPass
from convdog.simplifier.base_pass import BasePass


class O1Optimizer(object):
    def __init__(self, model: ConvDogModel, safe_mode=False):
        self.model = model
        self.safe_mode = safe_mode
        self.fuse_consecutive_node: Optional[BasePass] = None
        self.dead_code_elimination_pass: Optional[DeadCodeEliminationPass] = None
        self.gemm_pass: Optional[GemmFusionPass] = None
        self.inverse_pass: Optional[InverseOpCancellationPass] = None
        self.initialize_pass()

    def initialize_pass(self):
        self.fuse_consecutive_node = FuseConsecutiveNodePass()
        self.dead_code_elimination_pass = DeadCodeEliminationPass()
        self.gemm_pass = GemmFusionPass()
        self.inverse_pass = InverseOpCancellationPass()

    def apply(self) -> ConvDogModel:
        logger.info("[O1]: 开始执行初步算子优化和拓扑结构优化...")

        # 记录优化前的初始节点数
        # 这里的 self.model.graph.node 是 ONNX Proto 的底层节点列表
        prev_node_count = len(self.model.graph.nodes)
        iteration = 0

        while True:
            iteration += 1

            # --- 执行优化 Pass ---
            self.fuse_consecutive_node.run(self.model)
            self.dead_code_elimination_pass.run(self.model)
            self.gemm_pass.run(self.model)
            self.inverse_pass.run(self.model)

            self.model.reset_value_info()
            self.model.fold_tensors()

            # 获取优化后的节点数量
            current_node_count = len(self.model.graph.nodes)
            diff = prev_node_count - current_node_count
            diff_color = "green" if diff > 0 else "white"
            logger.info(
                f"[第 {iteration} 轮优化]："
                f"节点数 {prev_node_count} -> {current_node_count} "
                f"[{diff_color}]已削减:[/] {diff}"
            )

            # [收敛收工判断]
            # 如果节点数量不再发生变化，说明图中已经没有连续的节点了
            if current_node_count == prev_node_count:
                logger.success(f"[O1]: 优化收敛！总共迭代 {iteration} 轮。")
                break

            # 更新计数，继续下一轮嗅探
            prev_node_count = current_node_count

            # 防止死循环（保险起见，设置一个最大迭代次数）
            if iteration > 100:
                logger.warning("[O1]: 迭代次数过多，强制跳出，请检查图中是否存在环路。")
                break

        return self.model
