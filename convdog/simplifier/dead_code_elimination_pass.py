from typing import TYPE_CHECKING

from convdog.simplifier.base_pass import BasePass
from convdog.utils.logger import logger
if TYPE_CHECKING:
    import onnx_graphsurgeon as gs


class DeadCodeEliminationPass(BasePass):
    def __init__(self):
        super().__init__()

    def process(self, graph: "gs.Graph") -> bool:
        """
        执行死代码消除：
        1. 移除输出未被使用的孤立节点（非 Graph Outputs）。
        2. 移除未被任何节点引用的 Initializers (权重)。
        3. 移除未被使用的子图输入/输出。
        """
        # 记录清理前的状态
        prev_node_count = len(graph.nodes)
        prev_const_count = len(graph.tensors().keys())

        # 核心：调用 graphsurgeon 的 cleanup 魔法
        # recursive=True 会递归删除。例如：如果 A->B，B被删了，
        # 且 A 也没有其他输出，那么 A 也会被连带删除。
        graph.toposort()
        graph.cleanup()

        # 检查是否有变化
        curr_node_count = len(graph.nodes)
        curr_const_count = len(graph.tensors().keys())

        changed = (curr_node_count != prev_node_count) or (curr_const_count != prev_const_count)

        if changed:
            logger.info(f"[O1] 清理完成: 节点 {prev_node_count}->{curr_node_count}, "
                        f"参数 {prev_const_count}->{curr_const_count}")

        return changed
