import numpy as np
import onnx
import onnx_graphsurgeon as gs
from convdog.utils.logger import logger
from convdog.core.graph import ConvDogModel

class FP16Quantizer:
    def __init__(self, model: ConvDogModel, safe_mode=True):
        """
        FP16 深度量化器 (基于 GraphSurgeon)
        :param model: ConvDogModel 对象，内部维护 gs.Graph
        :param safe_mode: 是否开启安全模式，保护 Resize/Clip 等算子的关键输入
        """
        self.convdog_model = model
        self.safe_mode = safe_mode
        self.protected_names = set()

    def apply(self):
        logger.info("[O1] 开始语义感知的FP16深度转换")

        # 1. 扫描受限张量 (根据算子规范保护必须为 FP32 的输入)
        if self.safe_mode:
            self.protected_names = self._get_required_fp32_tensors()

        # 2. 统一处理所有张量 (权重 + 中间变量 + 输入输出)
        self._convert_all_tensors()

        # 3. 处理常数节点 (ONNX 中的 Constant Op，在 gs 中可能表现为独立节点或 Constant 对象)
        self._fix_constant_nodes()

        # 4. 图清理与同步
        self.convdog_model.graph.cleanup().toposort()
        self.convdog_model.sync_model()

        logger.success("[O2] FP16转换及图语义修复完成")
        return self.convdog_model

    def _get_required_fp32_tensors(self):
        """
        扫描 GS 图，返回必须保持为 float32 的张量名称集合
        """
        protected = set()
        for node in self.convdog_model.graph.nodes:
            # Resize / Upsample 的 scales 和 roi 必须是 float32
            if node.op in ["Resize", "Upsample"]:
                # index 1: roi, index 2: scales
                for i in [1, 2]:
                    if i < len(node.inputs) and isinstance(node.inputs[i], (gs.Variable, gs.Constant)):
                        protected.add(node.inputs[i].name)

            # Clip 的 min/max 若存在，保持 FP32 稳定性更好
            elif node.op == "Clip":
                for i in range(1, len(node.inputs)):
                    if node.inputs[i]:
                        protected.add(node.inputs[i].name)

            # 如果是某些特定的后端算子（如 TopK 的值），也可以在此添加
        return protected

    def _convert_all_tensors(self):
        """
        遍历图中所有张量，统一切换 dtype
        gs.Graph.tensors() 返回的是 Dict[str, Tensor]，包含所有的 Variable 和 Constant
        """
        count_weights = 0
        count_vars = 0

        for name, tensor in self.convdog_model.graph.tensors().items():
            if name in self.protected_names:
                continue

            # 处理权重 (Constant)
            if isinstance(tensor, gs.Constant) and tensor.dtype == np.float32:
                # np.clip 避免超出 FP16 表示范围 (inf)
                tensor.values = np.clip(tensor.values, -65504, 65504).astype(np.float16)
                count_weights += 1

            # 处理变量 (Variable: 包含输入、输出和中间张量)
            elif isinstance(tensor, gs.Variable) and tensor.dtype == np.float32:
                tensor.dtype = np.float16
                count_vars += 1

        logger.debug(f"已转换 {count_weights} 个权重张量和 {count_vars} 个变量张量至 FP16")

    def _fix_constant_nodes(self):
        """
        额外处理那些以 Node 形式存在的 Constant 算子
        """
        for node in self.convdog_model.graph.nodes:
            if node.op == "Constant":
                # Constant 算子的属性通常是 'value'
                val = node.attrs.get("value")
                if val is not None and val.dtype == np.float32:
                    node.attrs["value"].values = val.values.astype(np.float16)
                    # 也可以考虑将 Constant 算子直接折叠进 Initializer (gs.Constant)
                    # gs 的 cleanup 往往会自动处理

    def _refresh_model(self):
        """
        同步并导出模型，强制进行 ONNX 层面推断
        """
        try:
            # 调用 ConvDogModel 封装的同步逻辑：gs.export_onnx -> shape_inference
            self.convdog_model.fold_tensors()

            # 手动提升 IR 版本以支持更现代的 FP16 表达
            export_model = self.convdog_model.model
            if export_model.ir_version < 7:
                export_model.ir_version = 7

            onnx.checker.check_model(export_model, full_check=True)
            logger.debug("[Core]: 形状推断与类型传导刷新成功")
        except Exception as e:
            logger.warning(f"FP16 图检查存在非致命警告 (可能是自定义算子): {e}")
