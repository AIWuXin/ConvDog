from convdog.core.graph import ConvDogModel
from convdog.utils.logger import logger, OrtWarningPrintFilter


class O0Optimizer(object):
    """
    O0 等级优化器: 统一管理静态化、常量折叠与收敛控制
    """
    def __init__(self, graph: ConvDogModel, input_shapes: dict = None, safe_mode=False):
        self.graph = graph
        self.input_shapes = input_shapes
        self.safe_mode = safe_mode

    def apply(self) -> ConvDogModel:
        logger.info("[O0] 正在启动部署就绪优化 (Deployment Ready Pass)...")

        # 1. 执行静态化
        if self.input_shapes:
            logger.info("[O0] Step 1/2 - 正在执行静态化图手术...")
            logger.info(f" -> 注入静态形状: [bold cyan]{self.input_shapes}[/]")
            if self.input_shapes is not None:
                self.graph.resize_input_shape(self.input_shapes)
            else:
                logger.info("[O0] 无需静态化")

        # 2. 执行张量折叠 (带收敛监控)
        logger.info("[O0] Step 2/2 - 正在执行迭代张量折叠...")

        max_iters = 100
        prev_node_count = len(self.graph.graph.nodes) # 获取初始节点数

        with OrtWarningPrintFilter():
            for i in range(max_iters):
                # 执行一轮折叠
                self.graph.fold_tensors()

                # 统计当前节点数
                current_nodes = self.graph.graph.nodes
                current_count = len(current_nodes)
                diff = prev_node_count - current_count

                # 美化打印进度
                status_color = "green" if diff > 0 else "white"
                logger.info(
                    f" [折叠第 {i+1:02d} 轮] 节点数: [bold yellow]{prev_node_count}[/] -> "
                    f"[bold cyan]{current_count}[/] "
                    f"[{status_color}]已削减: {diff}[/]"
                )

                # 收敛判断：如果节点数不再减少，说明已经折叠完毕
                if diff <= 0:
                    logger.info(f" -> [bold green]图结构已收敛 (于第 {i+1} 轮)，停止折叠。[/]")
                    break

                prev_node_count = current_count

        logger.success("[O0] 模型优化处理完毕，准备交付！ 🐕")
        return self.graph
