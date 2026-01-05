import os
from collections import Counter, defaultdict

import numpy as np
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.console import Console, Group, Text
import onnxruntime as ort

from convdog.core.graph import ConvDogGraph
from convdog.utils.logger import logger


class ModelStats:
    """保存并提取模型统计信息的结构体"""

    def __init__(self, graph: ConvDogGraph, name: str):
        self.name = name
        self.graph = graph
        self.opset = graph.model.opset_import[0].version
        self.ir_version = graph.model.ir_version
        self.size_mb = graph.model.ByteSize() / (1024 * 1024)

        # 1. 统计算子数量 (Nodes)
        self.op_counts = Counter([node.op_type for node in graph.model.graph.node])

        # 2. 统计权重精度 (Initializers)
        self.weight_counts = Counter([str(arr.dtype) for arr in graph.initializers.values()])

        # 3. 统计对齐性 (TensorCore Alignment: dims % 8 == 0)
        self.aligned_count = 0
        self.total_weight_layers = len(graph.initializers)
        for arr in graph.initializers.values():
            if all(d % 8 == 0 for d in arr.shape if d > 1):
                self.aligned_count += 1

        # 4. 总参数量
        self.total_params = sum([arr.size for arr in graph.initializers.values()])
        self.dtype_params = defaultdict(int)
        for arr in graph.initializers.values():
            self.dtype_params[str(arr.dtype)] += arr.size

        # 5. 统计输入输出
        self.inputs = self._parse_io(graph.model.graph.input)
        self.outputs = self._parse_io(graph.model.graph.output)

    @staticmethod
    def _parse_io(io_list):
        info = []
        for x in io_list:
            shape = []
            if x.type.tensor_type.HasField("shape"):
                for dim in x.type.tensor_type.shape.dim:
                    shape.append(str(dim.dim_value) if dim.dim_value > 0 else dim.dim_param)
            info.append(f"{x.name}: ({', '.join(shape)})")
        return info


def get_diff_str(old_val, new_val, is_size=False):
    """根据数值变化生成带箭头的字符串"""
    if new_val == old_val:
        return f"{new_val:.2f} MB" if is_size else str(new_val)

    increased = new_val > old_val
    if is_size:
        color = "red" if increased else "green"
        arrow = "↑" if increased else "↓"
        return f"{new_val:.2f} MB [bold {color}]{arrow}[/]"
    else:
        color = "yellow" if increased else "green"
        arrow = "↑" if increased else "↓"
        return f"{new_val} [bold {color}]{arrow}[/]"


def run_inference(model_proto_bytes):
    """
    运行推理。改进版：优先从 Session 获取输入要求，实现类型自适应。
    """
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess = ort.InferenceSession(model_proto_bytes, providers=['CPUExecutionProvider'], sess_options=sess_options)

        input_dict = {}
        # 1. 动态获取 Session 需要的输入
        for input_meta in sess.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            dtype_str = input_meta.type  # e.g. 'tensor(float16)'

            # 处理动态 Shape (None 或 str)
            fixed_shape = []
            for s in shape:
                if isinstance(s, int) and s > 0:
                    fixed_shape.append(s)
                else:
                    # 如果用户在 O0 没指定 shapes，这里兜底用 1
                    fixed_shape.append(1)

            # 类型映射
            if "float16" in dtype_str:
                target_dtype = np.float16
            elif "int64" in dtype_str:
                target_dtype = np.int64
            else:
                target_dtype = np.float32

            # 生成随机数据
            data = np.random.randn(*fixed_shape).astype(target_dtype)
            input_dict[name] = data

        outputs = sess.run(None, input_dict)
        return outputs, None
    except Exception as e:
        logger.error(e)
        return None, str(e)


def calculate_rel_error(original_proto, optimized_proto):
    """
    计算相对误差。不再依赖外部传入的 inputs_info，实现全自动收敛。
    """
    # 转换 bytes
    orig_bytes = original_proto.model.SerializeToString()
    opt_bytes = optimized_proto.model.SerializeToString()

    # 1. 运行原始模型
    ref_outs, err_ref = run_inference(orig_bytes)
    if err_ref:
        return "ERR_TYPE"

    # 2. 运行优化模型
    opt_outs, err_opt = run_inference(opt_bytes)
    if err_opt:
        return "ERR_TYPE"

    # 3. 计算误差
    errors = []
    for r, o in zip(ref_outs, opt_outs):
        r_f32 = r.astype(np.float32)
        o_f32 = o.astype(np.float32)

        # 针对 FP16 计算余弦相似度（Cosine Similarity）比相对误差更科学
        inner_prod = np.sum(r_f32 * o_f32)
        iter_norm = np.linalg.norm(r_f32) * np.linalg.norm(o_f32)
        cosine_sim = inner_prod / (iter_norm + 1e-7)

        # 转换成 1 - sim 作为误差项
        errors.append(1.0 - cosine_sim)

    avg_error = np.mean(errors)
    return avg_error * 100


def print_comparison_table(original: ModelStats, optimized: ModelStats, elapsed_time: float):
    """
    基础优化对比表：展示算子变化明细
    """
    console = Console()
    table = Table(title="🐕 ConvDog 算子优化对比", header_style="bold magenta", border_style="cyan", box=box.ROUNDED)

    table.add_column("算子/属性 (Op/Attr)", justify="right", style="bold")
    table.add_column("原始 (Original)", justify="center")
    table.add_column("优化后 (Optimized)", justify="center")

    # 1. 基础信息
    table.add_row("Model Name", os.path.basename(original.name), os.path.basename(optimized.name))
    table.add_row("Op Set / IR", f"v{original.opset} / {original.ir_version}",
                  f"v{optimized.opset} / {optimized.ir_version}")

    # 2. IO 信息 (取第一个作为代表)
    table.add_section()
    table.add_row("Input Info", original.inputs[0] if original.inputs else "N/A",
                  optimized.inputs[0] if optimized.inputs else "N/A")
    table.add_row("Output Info", original.outputs[0] if original.outputs else "N/A",
                  optimized.outputs[0] if optimized.outputs else "N/A")

    # 3. 算子列表 (带箭头对比)
    table.add_section()
    all_ops = sorted(list(set(original.op_counts.keys()) | set(optimized.op_counts.keys())))
    for op in all_ops:
        orig_cnt = original.op_counts.get(op, 0)
        opt_cnt = optimized.op_counts.get(op, 0)

        # 使用 diff 函数美化输出
        styled_opt = get_diff_str(orig_cnt, opt_cnt)
        table.add_row(op, str(orig_cnt), styled_opt)

    # 4. 模型体积
    table.add_section()
    table.add_row("Model Size", f"{original.size_mb:.2f} MB",
                  get_diff_str(original.size_mb, optimized.size_mb, is_size=True))

    # 5. 时间统计
    table.add_section()
    table.add_row("Elapsed Time", "-", f"[bold yellow]{elapsed_time:.3f} s[/]")

    console.print("\n")
    console.print(table)


def print_every_layer_quant_table(graph: ConvDogGraph):
    """单独的量化状态表：展示权重类型分布"""
    console = Console()
    table = Table(title="💎 量化细节", header_style="bold blue")
    table.add_column("Weight Name", justify="left")
    table.add_column("Shape", justify="center")
    table.add_column("Precision", justify="center")

    for name, arr in graph.initializers.items():
        table.add_row(name, str(arr.shape), str(arr.dtype))

    console.print(table)


def print_quant_summary(original: ModelStats, optimized: ModelStats, elapsed_time: float):
    """
    专家级看板：量化对比 + 回退分析 + 部署诊断
    """
    console = Console()

    # --- 1. 左侧：转换指标 ---
    comp_table = Table(box=box.SIMPLE, header_style="bold magenta")
    comp_table.add_column("Item", justify="right")
    comp_table.add_column("Change", justify="left")

    all_dts = sorted(list(set(original.weight_counts.keys()) | set(optimized.weight_counts.keys())))
    for dt in all_dts:
        o_c = original.weight_counts.get(dt, 0)
        n_c = optimized.weight_counts.get(dt, 0)
        if o_c != n_c:
            arrow = "[bold green]↑[/]" if n_c > o_c else "[bold red]↓[/]"
            comp_table.add_row(f"{dt} Weights", f"{o_c} -> {n_c} {arrow}")

    comp_table.add_row("Model Size",
                       f"{original.size_mb:.2f} -> {get_diff_str(original.size_mb, optimized.size_mb, is_size=True)}")

    # --- 2. 中间：回退层 (Fallback) ---
    fallback_layers = []
    for name, arr in optimized.graph.initializers.items():
        if str(arr.dtype) == "float32":
            fallback_layers.append((name, arr.size * arr.itemsize / 1024))
    fallback_layers.sort(key=lambda x: x[1], reverse=True)

    fb_table = Table(box=box.SIMPLE, header_style="bold red")
    fb_table.add_column("Remaining FP32", max_width=25)
    fb_table.add_column("KB", justify="right")
    if not fallback_layers:
        fb_table.add_row("[green]Success: 0 fallback[/]", "")
    else:
        for name, size in fallback_layers[:8]:
            fb_table.add_row(name, f"{size:.0f}")

    # --- 3. 右侧：部署诊断 (Diagnostics) ---
    diag_table = Table(box=box.SIMPLE, header_style="bold cyan")
    diag_table.add_column("Metric", justify="right")
    diag_table.add_column("Status/Value", justify="left")

    # 对齐检查 (TensorCore 友好度)
    alignment_rate = (
                optimized.aligned_count / optimized.total_weight_layers * 100) if optimized.total_weight_layers > 0 else 0
    adj_status = "[green]Excellent[/]" if alignment_rate > 90 else "[yellow]Fair[/]"
    diag_table.add_row("TC Alignment", f"{alignment_rate:.1f}% {adj_status}")

    # 算子消减率
    o_ops = sum(original.op_counts.values())
    n_ops = sum(optimized.op_counts.values())
    reduction = (1 - n_ops / o_ops) * 100 if o_ops > 0 else 0
    diag_table.add_row("Op Reduction", f"{reduction:.1f}%")

    # 误差分析
    rel_error = calculate_rel_error(original.graph, optimized.graph)
    if rel_error is not None:
        if rel_error == "ERR_TYPE":
            diag_table.add_row("Rel Error Δ", "[red bold]ORT 运行失败!!![/]")
        else:
            if rel_error < 0.8:
                color, status = "green", "(Perfect)"
            elif rel_error < 1.8:
                color, status = "yellow", "(Fair)"
            else:
                color, status = "red", "(Degraded)"
            diag_table.add_row("Rel Error Δ", f"[{color}]{rel_error:.3f}% {status}[/]")
    else:
        diag_table.add_row("Rel Error Δ", "[dim]ORT not found[/]")

    # 精度分布条 (Visual Progress Bar)
    # 计算 FP16 占比
    fp16_ratio = optimized.dtype_params.get("float16", 0) / optimized.total_params if optimized.total_params > 0 else 0

    # 4. 组装
    main_group = Group(
        Columns([
            Panel(comp_table, title="📊 Precision Shift", border_style="magenta"),
            Panel(fb_table, title="⚠️ Fallback Details", border_style="red"),
            Panel(diag_table, title="🚀 Deployment Ready", border_style="cyan")
        ]),
        Panel(
            Text.assemble(
                ("Precision Distribution: ", "bold"),
                (f"FP16 {fp16_ratio * 100:.1f}% ", "yellow"),
                ("|" * int(fp16_ratio * 40), "yellow"),
                ("|" * (40 - int(fp16_ratio * 40)), "cyan"),
                (f" FP32 {(1 - fp16_ratio) * 100:.1f}%", "cyan"),
            ),
            border_style="dim"
        )
    )

    console.print(Panel(
        main_group,
        title=f"💎 ConvDog [bold]量化报告[/] - {os.path.basename(original.name)}",
        subtitle=f"Total: {original.total_params / 1e6:.2f}M Params | Efficiency: [bold green]{(1 - optimized.size_mb / original.size_mb) * 100:.1f}%[/]",
        border_style="bold blue", padding=(1, 2)
    ))
