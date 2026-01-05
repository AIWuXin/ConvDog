import collections
from typing import Dict, Optional
import ast
import math

import numpy as np
import onnx
from onnx import numpy_helper

from convdog.utils.logger import logger


class ConvDogGraph(object):
    def __init__(self, model_path: str):
        logger.info(f"正在嗅探模型: [bold white]{model_path}[/]", extra={"markup": True})
        self.model = onnx.load(model_path)
        # 初始化索引
        self.nodes = {}
        self.initializers: Dict[str, np.ndarray] = {}
        self.producers = {}
        self.consumers = collections.defaultdict(list)
        self.update_indexes()
        logger.info(f"成功嗅探模型: [bold white]{model_path}[/]")

    @property
    def graph(self) -> onnx.GraphProto:
        return self.model.graph

    def inject_convdog_info(self, author_name="ConvDog🐕"):
        """
        向 ONNX 模型中注入“转换狗”特有的导出信息
        """
        # 准备元数据字典
        meta_info = {
            "producer_name": "ConvDog🐕 (转换汪)",
            "producer_version": "0.1.0",
            "description": "This model was hunted and optimized by ConvDog.",
            "author": author_name,
            "status": "Injected_For_Testing"
        }

        # 清理旧的同名元数据（防止重复注入）
        existing_props = {prop.key for prop in self.model.metadata_props}

        for key, value in meta_info.items():
            if key not in existing_props:
                meta_prop = self.model.metadata_props.add()
                meta_prop.key = key
                meta_prop.value = value

        # [x] 同时修改 producer 字段
        # 保留原始导出信息
        # self.model.producer_name = "ConvDog🐕"
        # self.model.producer_version = "0.1.0"

        logger.debug(f"成功注入元数据，留下爪印：{author_name}")

    def update_indexes(self):
        """
        重置并重新构建全图索引。
        修复点：确保 self.initializers 始终存储 NumPy 数组而非 TensorProto。
        """
        self.nodes.clear()
        self.initializers.clear()
        self.producers.clear()
        self.consumers.clear()

        # 1. 建立节点、生产者、消费者逻辑
        for node in self.graph.node:
            if node.name:
                self.nodes[node.name] = node

            for output in node.output:
                self.producers[output] = node

            for input_name in node.input:
                self.consumers[input_name].append(node)

        # 2. 【修复关键】：建立权重索引时，强制转换 Proto 为 Numpy
        for init in self.graph.initializer:
            # 这样保证了其他 Pass（如 FP16Quantizer）拿到的永远是 array
            self.initializers[init.name] = numpy_helper.to_array(init)

        logger.debug("图索引已刷新：[cyan]%d[/] 节点, [cyan]%d[/] 权重数组",
                     len(self.graph.node), len(self.initializers), extra={"markup": True})

    def get_initializer_array(self, name: str) -> Optional[np.ndarray]:
        """
        获取权重数组。因为索引里存的就是 Numpy，直接返回即可。
        """
        return self.initializers.get(name)

    def add_initializer(self, name: str, array: np.ndarray):
        """
        向图中添加或更新权重。
        """
        # 1. 生成底层 Proto 对象用于 ONNX 序列化
        new_init = numpy_helper.from_array(array, name=name)

        # 2. 同步更新 ModelProto
        # 优先查找并替换，不存在则追加
        found = False
        for i, init in enumerate(self.graph.initializer):
            if init.name == name:
                self.graph.initializer.remove(init)
                self.graph.initializer.insert(i, new_init)
                found = True
                break
        if not found:
            self.graph.initializer.append(new_init)

        # 3. 同步更新内部 NumPy 索引
        self.initializers[name] = array

    @staticmethod
    def _parser_symbolic_shape(
            expression: str,
            symbolic_shape: Dict[str, int]
    ):
        """
        安全地计算数学表达式，支持简单的数学函数
        """

        # 定义允许的数学函数
        allowed_functions = {
            'floor': math.floor,
            'ceil': math.ceil,
            'round': round,
            'sqrt': math.sqrt,
            'abs': abs,
            'int': int,
            'float': float,
            'math': math
        }
        allowed_symbols = {
            'floor': "math.floor",
            'ceil': "math.ceil",
            'round': "round",
            'sqrt': "math.sqrt",
            'abs': "abs",
            'int': "int",
            'float': "float"
        }

        for key, value in symbolic_shape.items():
            expression = expression.replace(key, str(value))
        for key, value in allowed_symbols.items():
            expression = expression.replace(key, str(value))
        expression = expression.strip()

        # 1. 首先尝试直接计算简单表达式
        try:
            # 使用 ast 解析确保安全
            tree = ast.parse(expression, mode='eval')

            # 检查 AST 中是否只包含安全的节点类型
            safe_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                ast.Pow, ast.USub, ast.UAdd, ast.Num, ast.Constant,
                ast.Call, ast.Name, ast.Attribute, ast.Load
            )

            for node in ast.walk(tree):
                if isinstance(node, safe_nodes):
                    if isinstance(node, ast.Call):
                        # 检查函数调用是否在允许列表中
                        if isinstance(node.func, ast.Name):
                            if node.func.id not in allowed_functions:
                                raise ValueError(f"不允许的函数调用: {node.func.id}")
                    elif isinstance(node, ast.Attribute):
                        if node.attr not in allowed_functions.keys():
                            raise ValueError(f"不安全的节点类型: {type(node).__name__}")
                else:
                    raise ValueError(f"不安全的节点类型: {type(node).__name__}")

            # 编译并执行
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, allowed_functions)

            if isinstance(result, (int, float)):
                return int(result)  # 形状维度应该是整数
        except Exception as e:
            logger.debug(e)
            logger.debug("形状推断失败!!!退回原符号形状")
            return symbolic_shape

    def formalize_graph(self):
        """
        工业级图规范化：将所有 Constant 节点转为 Initializer。
        解决 ORT 优化器在大模型融合时对 Constant 节点的索引查找失败问题。
        """
        new_nodes = []
        for node in self.model.graph.node:
            if node.op_type == "Constant":
                # 提取常量值并转为 Initializer
                tensor_proto = node.attribute[0].t
                tensor_proto.name = node.output[0]
                self.model.graph.initializer.append(tensor_proto)
            else:
                new_nodes.append(node)

        # 重新刷新节点列表
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(new_nodes)
        self.update_indexes()

    def resize_input_shape(self, input_shapes: dict):
        """
        底层工具：修改输入节点的 Proto 并刷新全图形状
        """
        symbolic_shape = {}  # 缓存符号形状映射

        for input_proto in self.model.graph.input:
            if input_proto.name in input_shapes:
                target_shape = input_shapes[input_proto.name]
                # 修改 Opaque 的 TensorTypeProto
                for i, dim in enumerate(input_proto.type.tensor_type.shape.dim):
                    if i < len(target_shape):
                        if len(dim.dim_param) > 0:
                            symbolic_shape[dim.dim_param] = target_shape[i]
                        dim.ClearField("dim_param")
                        dim.dim_value = target_shape[i]
                logger.debug(f"GraphCore: 已修改输入 {input_proto.name} 的尺寸数据")

        for idx, value_info in enumerate(self.graph.value_info):
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    cur_symbolic_shape = dim.dim_param
                    static_shape = self._parser_symbolic_shape(
                        cur_symbolic_shape, symbolic_shape
                    )
                    if not isinstance(static_shape, dict):
                        dim.ClearField("dim_param")
                        dim.dim_value = static_shape

        self.formalize_graph()

        # 核心步骤：重新推理形状以确保中间 ValueInfo 逻辑一致
        import onnx.shape_inference
        self.model = onnx.shape_inference.infer_shapes(
            self.model,
            strict_mode=True,
            check_type=True
        )
        try:
            onnx.checker.check_model(self.model, full_check=True)
        except Exception as e:
            logger.error(e)
            logger.warning("静态图检查失败!!!")
        self.update_indexes()

    def save(self, output_path: str):
        """执行最终检查并保存"""
        try:
            # 注意：保存前如果有大量 add_initializer，其实不需要 full update_indexes
            # 除非你修改了节点的 input/output 拓扑。
            # 为了保险起见保留它，但也确保了其中的转换逻辑是正确的。
            self.update_indexes()

            onnx.checker.check_model(self.model)
            onnx.save(self.model, output_path)
            logger.success(f"导出成功: [underline]{output_path}[/]", extra={"markup": True})
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
