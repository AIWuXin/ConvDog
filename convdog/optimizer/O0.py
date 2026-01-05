from convdog.core.graph import ConvDogGraph
from convdog.quantizer.fp16 import FP16Quantizer
from convdog.utils.logger import logger

class O0Optimizer:
    """
    O0 等级优化器: 统一管理 FP32->FP16 转换和静态化
    """
    def __init__(self, graph: ConvDogGraph, input_shapes: dict = None, safe_mode=False):
        self.graph = graph
        self.input_shapes = input_shapes
        self.safe_mode = safe_mode

    def apply(self) -> ConvDogGraph:
        logger.info("[O0] 正在启动部署就绪优化 (Deployment Ready Pass)...")

        # 1. 执行静态化 (如果用户指定了形状)
        if self.input_shapes:
            logger.info(f" -> 执行静态化图手术: {self.input_shapes}")
            self.graph.resize_input_shape(self.input_shapes)

        # 2. 执行 FP16 浮点权重转换 (包含之前写的语义保护逻辑)
        # 这里可以直接复用之前的 FP16Quantizer 逻辑
        fp16_tool = FP16Quantizer(self.graph, safe_mode=self.safe_mode)
        self.graph = fp16_tool.apply()

        logger.success("[O0] 模型优化处理完毕")
        return self.graph
