import numpy as np
import onnx_graphsurgeon as gs

from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger

class FuseMultipleGatherPass(BasePass):
    """
    O1 优化：将同一源张量上多次对连续索引 (例如 0,1,2,...) 的 Gather 合并为一个 Split。
    修正要点：
      - 仅合并对同一 gs.Variable 的 Gather（identity check）
      - 如果原 Gather 是 scalar indices（会降维），则在 Split 后插入 Squeeze，使语义与 Gather 保持一致
      - 为 split 与 squeeze 的输出变量尽量填充 dtype/shape 以通过 shape inference
    """
    def process(self, graph: gs.Graph) -> bool:
        changed = False
        fusion_count = 0

        logger.debug("[O1]: 正在融合分组Gather......")
        graph.cleanup().toposort()

        # 收集所有 Gather 节点（使用快照）
        all_gathers = [n for n in graph.nodes if n.op == "Gather"]

        # 按 (data_var id, axis) 分组 —— 使用变量对象 identity 而不是仅 name
        groups = {}
        for g in all_gathers:
            if len(g.inputs) < 2:
                continue
            data = g.inputs[0]
            idx = g.inputs[1]
            if not isinstance(idx, gs.Constant):
                continue
            axis = int(g.attrs.get("axis", 0))
            key = (id(data), axis)  # 用 id(data) 保证同一对象
            groups.setdefault(key, []).append(g)

        for (data_id, axis), gnodes in groups.items():
            if len(gnodes) < 2:
                continue

            # 收集 (index_value, gather_node) 并检测 indices 是否为 scalar
            entries = []
            skip_group = False
            any_non_scalar = False
            for g in gnodes:
                idx_const = g.inputs[1]
                try:
                    vals = np.array(idx_const.values).astype(int).reshape(-1)
                except Exception:
                    skip_group = True
                    break
                if vals.size != 1:
                    any_non_scalar = True
                entries.append((int(vals[0]) if vals.size == 1 else tuple(vals.tolist()), g))
            if skip_group:
                continue

            # 只处理所有 indices 都是 scalar 的情况（更安全）
            if any_non_scalar:
                continue

            indices = sorted([e[0] for e in entries])
            # 暂定策略：只处理从0开始的连续序列 0..k-1
            if indices[0] != 0 or indices != list(range(indices[-1] + 1)):
                continue

            k = indices[-1] + 1

            # 源张量（取第一个 gather 的输入）
            data_var = gnodes[0].inputs[0]

            # 如果有静态 shape，则做额外校验并归一化 axis（支持负轴）
            axis_norm = axis
            if data_var.shape:
                rank = len(data_var.shape)
                if axis < 0:
                    axis_norm = axis + rank
                if axis_norm < 0 or axis_norm >= rank:
                    logger.debug(f"[O1] 跳过融合 Split: axis {axis} 越界 for {data_var.name}")
                    continue
                dim_size = data_var.shape[axis_norm]
                if dim_size is not None and dim_size < k:
                    logger.debug(f"[O1] 跳过融合 Split: 源张量 {data_var.name} 在 axis {axis_norm} 的长度 {dim_size} 小于 k={k}")
                    continue
            else:
                # 若没有 shape 信息，仍使用 axis 原值作为 split axis
                axis_norm = axis

            # 判断原始 Gather 是否会降维：取任一 gnode 的输出 shape（若有）和 data_var.shape 比较
            gather_reduces_dim = False
            sample_g_out = None
            if gnodes[0].outputs and gnodes[0].outputs[0].shape and data_var.shape:
                sample_g_out = gnodes[0].outputs[0].shape
                if len(sample_g_out) == len(data_var.shape) - 1:
                    gather_reduces_dim = True

            # 构造 split 输出 Variable（先为 split 的原始输出 —— 未 squeeze）
            split_outs = []
            for i in range(k):
                out_shape = None
                if data_var.shape:
                    out_shape = list(data_var.shape)
                    try:
                        out_shape[axis_norm] = 1  # Split 每块把该维设为 1
                    except Exception:
                        out_shape = None
                out_dtype = getattr(data_var, "dtype", None)
                split_var = gs.Variable(name=f"{data_var.name}_split_{i}", dtype=out_dtype, shape=out_shape)
                split_outs.append(split_var)

            # 创建 Split 节点，带 split & num_outputs
            split_var = gs.Constant(
                name=f"{data_var.name}_split",
                values=np.array([1] * k, dtype=np.int64)
            )
            split_node = gs.Node(
                op="Split",
                name=f"Fused_Split_{data_var.name}",
                inputs=[data_var, split_var],
                outputs=split_outs,
                attrs={"axis": int(axis_norm)}
            )
            graph.nodes.append(split_node)

            # 如果原 Gather 会降维，则为每个 split 输出插入 Squeeze 节点并将消费者指向 squeeze 输出
            squeeze_outs = {}
            if gather_reduces_dim:
                for i, split_var in enumerate(split_outs):
                    squeeze_var = gs.Variable(
                        name=f"{split_var.name}_sq",
                        dtype=split_var.dtype
                    )

                    # 修正点：将 axes 作为 Constant 输入，而不是 Attribute
                    # 这能确保在 Opset 13+ 中 Squeeze 只针对 axis_norm 操作
                    axes_const = gs.Constant(
                        name=f"{split_var.name}_axes",
                        values=np.array([axis_norm], dtype=np.int64)
                    )

                    squeeze_node = gs.Node(
                        op="Squeeze",
                        name=f"Fused_Squeeze_{data_var.name}_{i}",
                        inputs=[split_var, axes_const], # 两个输入
                        outputs=[squeeze_var]
                    )
                    graph.nodes.append(squeeze_node)
                    squeeze_outs[i] = squeeze_var

            # 将每个 Gather 的输出消费者重定向到对应 split_out（或 squeeze_out）
            for idx_val, gnode in entries:
                if not gnode.outputs:
                    continue
                out_var = gnode.outputs[0]
                target_var = squeeze_outs.get(idx_val, split_outs[idx_val])

                # 替换 downstream 的输入项
                for consumer in list(out_var.outputs):
                    for i, inp in enumerate(consumer.inputs):
                        if inp == out_var:
                            consumer.inputs[i] = target_var
                            logger.debug(f"[O1] 重定向: {consumer.name} 的输入 {out_var.name} -> {target_var.name}")

                # 记录日志并断开旧节点引用，等待 cleanup 删除
                logger.debug(f"[O1] 合并 Gather -> {split_node.name} 映射: {gnode.name} -> {target_var.name}")
                gnode.inputs = []
                gnode.outputs = []

            fusion_count += 1
            changed = True

        if changed:
            graph.cleanup().toposort()
            logger.debug(f"[O1] 总共执行了 {fusion_count} 次 Gather->Split(+Squeeze) 融合")

        return changed
