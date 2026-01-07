import argparse
import sys
import time
from typing import Dict, Optional

from convdog.core.graph import ConvDogModel
from convdog.optimizer.O0 import O0Optimizer
from convdog.optimizer.O1 import O1Optimizer
from convdog.optimizer.O2 import O2Optimizer
from convdog.utils.logger import logger
from convdog.utils.stats import (
    ModelStats,
    print_comparison_table,
    print_quant_summary
)


def parse_shape_arg(shape_str) -> Optional[Dict[str, int]]:
    """解析 image:1,3,224,224 格式"""
    if not shape_str: return None
    try:
        res = {}
        for item in shape_str.split(";"):
            name, dims = item.split(":")
            res[name] = [int(d) for d in dims.split(",")]
        return res
    except Exception as e:
        logger.error(e)
        raise argparse.ArgumentTypeError("形状格式必须为 'name:1,3,224,224'")


def parse_level_arg(level_str) -> int:
    """解析优化等级"""
    if level_str not in ["O0", "O1", "O2", "O3"]:
        logger.error("[x] 暂不支持的优化等级!!!")
        sys.exit(-1)
    return {"O0": 0, "O1": 1, "O2": 2, "O3": 3}[level_str]


def optimize_model(
        input_path: str,
        output_path: str,
        opt_level: int = 0,
        input_shapes: Optional[Dict[str, int]] = None
):
    # 加载原始模型
    graph = ConvDogModel(input_path)
    original_stats = ModelStats(graph, input_path)
    o0_optimizer = O0Optimizer(graph, input_shapes)
    o1_optimizer, o2_optimizer, o3_optimizer = None, None, None

    # 注入信息
    graph.inject_convdog_info()

    start_time = time.time()
    optimized_graph = o0_optimizer.apply()
    logger.success(f"[*] O0等级优化完毕!")

    if opt_level >= 1:
        o1_optimizer = O1Optimizer(optimized_graph)
        optimized_graph = o1_optimizer.apply()
        logger.success(f"[*] O1等级优化完毕!")
    if opt_level >= 2:
        o2_optimizer = O2Optimizer(optimized_graph)
        optimized_graph = o2_optimizer.apply()
        logger.success(f"[*] O2等级优化完毕!")

    if opt_level >= 3:
        pass

    if opt_level >= 2 and input_path is not None:
        o2_optimizer.replace_custom_ops()
        graph.sync_model()

    elapsed = time.time() - start_time

    # 保存优化后模型
    optimized_stats = ModelStats(graph, output_path)
    logger.info("正在统计计算图优化情况......")
    print_comparison_table(original_stats, optimized_stats, elapsed)
    logger.info("正在统计推理指标......")
    print_quant_summary(original_stats, optimized_stats)
    optimized_graph.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="ConvDog🐕 模型优化工具")
    parser.add_argument("input", help="输入 ONNX 路径")
    parser.add_argument("output", help="输出 ONNX 路径")
    parser.add_argument("level", type=parse_level_arg, help="优化等级")
    parser.add_argument("--shapes", type=parse_shape_arg, help="静态化形状, 格式 'name:1,3,224,224'")
    parser.add_argument("--fp16", action="store_false", help="fp16量化, 默认在O0阶段开启")
    args = parser.parse_args()

    optimize_model(
        args.input,
        args.output,
        args.level,
        args.shapes
    )


if __name__ == '__main__':
    sys.exit(main())
