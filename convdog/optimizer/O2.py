from typing import Optional

from convdog.core.graph import ConvDogModel
from convdog.utils.logger import logger
from convdog.simplifier.fuse_conv_pass import FuseConvPass
from convdog.simplifier.fuse_gelu_pass import FuseGeluPass


class O2Optimizer(object):
    def __init__(self, model: ConvDogModel, safe_mode=False):
        self.model = model
        self.safe_mode = safe_mode
        self.conv_pass: Optional[FuseConvPass] = None
        self.gelu_pass: Optional[FuseGeluPass] = None
        self.initialize_pass()

    def initialize_pass(self):
        self.conv_pass = FuseConvPass()
        self.gelu_pass = FuseGeluPass()

    def replace_custom_ops(self):
        """
        替换自定义算子为标准算子。
        """
        self.conv_pass.replace_custom_ops(self.model.graph)
        self.gelu_pass.replace_custom_ops(self.model.graph)

    def apply(self) -> ConvDogModel:
        """
        应用O2优化等级。
        """
        logger.info("[O2]: 开始进行深度算子融合......")
        # 记录优化前的初始节点数
        prev_node_count = len(self.model.graph.nodes)
        iteration = 0

        while True:
            iteration += 1

            # --- 执行优化 Pass ---
            self.model.reset_value_info()
            self.model.fold_tensors()

            self.conv_pass.run(self.model)
            self.gelu_pass.run(self.model)

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
                logger.success(f"[O2]: 优化收敛！总共迭代 {iteration} 轮。")
                break

            # 更新计数，继续下一轮嗅探
            prev_node_count = current_node_count

            # 防止死循环（保险起见，设置一个最大迭代次数）
            if iteration > 100:
                logger.warning("[O2]: 迭代次数过多，强制跳出，请检查图中是否存在环路。")
                break

        return self.model
