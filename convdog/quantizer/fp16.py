import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from convdog.utils.logger import logger
from convdog.core.graph import ConvDogModel


class FP16Quantizer:
    def __init__(self, graph: ConvDogModel, safe_mode=False):
        self.graph = graph
        self.model = self.graph.model
        self.safe_mode = safe_mode
        self.protected_names = set()

    def apply(self):
        logger.info("[O0] 开始语义感知的 FP16 深度量化")

        # 1. 扫描受限张量
        self.protected_names = self._get_required_fp32_tensors()

        # 2. 原地修改 Initializer (权重)
        self._apply_weights()

        # 3. 修改node io
        self._apply_value_infos()

        # 3. 同步计算图
        self._sync_graph()

        # 4. 安全的形状推断
        self._safe_inference()

        self.graph.update_indexes()

        logger.success("[O0] FP16 转换及图语义修复完成")
        return self.graph

    def _apply_weights(self):
        # 直接遍历 Protobuf 列表，不经过中间字典，确保修改生效
        for init in self.model.graph.initializer:
            if init.name in self.protected_names:
                continue

            if init.data_type == TensorProto.FLOAT:
                # 提取数据
                arr = numpy_helper.to_array(init)
                # 检查溢出并转换
                arr_fp16 = np.clip(arr, -65504, 65504).astype(np.float16)

                # 重新构造这个 Initializer 节点
                self.graph.add_initializer(init.name, arr_fp16)
                logger.debug(f"已强制转换权重: {init.name}")

    def _apply_value_infos(self):
        for idx, value_info in enumerate(self.model.graph.value_info):
            value_type = value_info.type.tensor_type.elem_type
            if value_type == TensorProto.FLOAT:
                if value_info.name not in self.protected_names:
                    self.model.graph.value_info[idx].type.tensor_type.elem_type = TensorProto.FLOAT16

    def _sync_graph(self):
        for idx, node in enumerate(self.model.graph.node):
            if node.op_type == "Constant":
                self._apply_constant(idx)

        # 类型对齐
        for x in list(self.model.graph.input) + list(self.model.graph.output):
            if x.name not in self.protected_names:
                x.type.tensor_type.elem_type = TensorProto.FLOAT16

    def _safe_inference(self):
        """执行推断并确保结果不回退"""
        from onnx import shape_inference
        # 提升 IR 版本以支持更高的 FP16 兼容性
        if self.model.ir_version < 7:
            self.model.ir_version = 7

        try:
            # 这里的推断必须严谨。如果推断失败，我们至少保留了已经修改好类型的 model
            inferred = shape_inference.infer_shapes(self.model, check_type=True)
            self.graph.model = inferred
            self.model = inferred
            logger.debug("Core: 形状推断与类型传导刷新成功")
        except Exception as e:
            logger.error(f"Core: 形状推断发生错误: {e}")

        try:
            onnx.checker.check_model(self.model, full_check=True)
        except Exception as e:
            logger.error(e)
            logger.warning("fp16图检查失败!!!")

    def _get_required_fp32_tensors(self):
        """
        [逻辑分离] 扫描所有节点，返回必须保持为 float32 的输入张量名单
        参考自 ONNX 算子规范。
        """
        protected = set()
        for node in self.graph.model.graph.node:
            # Resize / Upsample 的特定输入必须是 float32
            if node.op_type in ["Resize", "Upsample"]:
                # index 1: roi, index 2: scales (必须为 float32)
                for i in [1, 2]:
                    if len(node.input) > i and node.input[i]:
                        protected.add(node.input[i])

            # Clip 的 min/max 建议保持 float32 保证数值稳定性
            elif node.op_type == "Clip":
                for i in range(1, len(node.input)):
                    if node.input[i]:
                        protected.add(node.input[i])

            # 在此处可以扩展其他必须为 FP32 的算子，如 TopK, MeanVarianceNormalization 等
        return protected

    def _apply_constant(self, idx):
        node = self.model.graph.node[idx]
        for attr in node.attribute:
            if attr.name == "value":
                # 直接修改 TensorProto 内部数据类型
                attr.t.data_type = TensorProto.FLOAT16 # FLOAT16
                # 重新填入 fp16 字节流
                original_arr = numpy_helper.to_array(attr.t)
                attr.t.raw_data = original_arr.astype(np.float16).tobytes()
