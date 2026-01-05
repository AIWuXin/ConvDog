from convdog.core.graph import ConvDogModel
from convdog.utils.logger import logger

class O0Optimizer:
    """
    O0 等级优化器: 统一管理 FP32->FP16 转换和静态化
    """
    def __init__(self, graph: ConvDogModel, input_shapes: dict = None, safe_mode=False):
        self.graph = graph
        self.input_shapes = input_shapes
        self.safe_mode = safe_mode

    def apply(self) -> ConvDogModel:
        logger.info("[O0] 正在启动部署就绪优化 (Deployment Ready Pass)...")

        # 1. 执行静态化 (如果用户指定了形状)
        logger.info("[O0] Step 1/2 - 执行静态化图手术...")
        if self.input_shapes:
            logger.info(f" -> 执行静态化图手术: {self.input_shapes}")
            self.graph.resize_input_shape(self.input_shapes)

        # 2. 执行张量折叠
        logger.info("[O0] Step 2/2 - 执行张量折叠...")
        self.graph.fold_tensors()

        logger.success("[O0] 模型优化处理完毕")
        return self.graph
