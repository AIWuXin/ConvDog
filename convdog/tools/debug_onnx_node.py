import onnx


def debug_onnx_node(model_path, node_keyword):
    model = onnx.load(model_path)
    graph = model.graph

    print(f"🔍 正在检索包含关键词 '{node_keyword}' 的节点及其上下文...\n")

    # 建立查找字典
    inits = {i.name: i for i in graph.initializer}
    v_infos = {v.name: v for v in list(graph.value_info) + list(graph.input) + list(graph.output)}

    # 定义类型映射
    type_map = {1: "FLOAT32", 10: "FLOAT16", 7: "INT64", 2: "UINT8"}

    target_nodes = [n for n in graph.node if node_keyword in n.name]

    if not target_nodes:
        print(f"❌ 未找到包含 '{node_keyword}' 的节点")
        return

    for node in target_nodes:
        print(f"【节点】: {node.name} (OpType: {node.op_type})")

        for i, input_name in enumerate(node.input):
            # 1. 检查是否是权重 (Initializer)
            if input_name in inits:
                init = inits[input_name]
                dtype = type_map.get(init.data_type, str(init.data_type))
                print(f"  └─ 输入[{i}] (Weight): {input_name}")
                print(f"     ➔ 实际数据类型: {dtype}")

            # 2. 检查是否是中间张量 (ValueInfo)
            elif input_name in v_infos:
                vi = v_infos[input_name]
                dtype_val = vi.type.tensor_type.elem_type
                dtype = type_map.get(dtype_val, str(dtype_val))

                # 检查 Shape
                shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
                print(f"  └─ 输入[{i}] (Tensor): {input_name}")
                print(f"     ➔ 元数据声明类型: {dtype}")
                print(f"     ➔ 元数据声明形状: {shape}")

            else:
                print(f"  └─ 输入[{i}] (Unknown): {input_name} (不在 ValueInfo 或 Initializers 中！)")

        print("-" * 60)

if __name__ == "__main__":
    # 修改为你生成的那个问题的模型路径
    MODEL_FILE = "tests/res/onnx/dpts_sim.onnx"
    # 报错信息里提到的节点关键词
    KEYWORD = "/blocks.0/norm1/"

    debug_onnx_node(MODEL_FILE, KEYWORD)
