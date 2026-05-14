import torch
from pathlib import Path
from flower_fine_tuning import load_model

model_output_path = Path(__file__).parent.parent.joinpath("assets/model")

def load_model_from_state_dict():
    model = load_model()
    # 暂存模型
    model_dict_path = model_output_path.joinpath("flower_model.pth")

    # 加载 state_dict
    state_dict = torch.load(model_dict_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 成功加载模型参数")

    # 测试前向传播
    # 准备示例输入（VGG 需要 224x224 输入）
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"📊 测试输出形状: {test_output.shape}")  # 应该是 [1, 17]
    return model


def export_to_onnx(model):
    model.eval()  # 导出前务必设置为评估模式
    dummy_input = torch.randn(1, 3, 224, 224)
    model_onnx_path = model_output_path.joinpath("flower_model.onnx")

    # 定义动态维度：batch 维度可变，其他维度固定
    dynamic_shapes = {
        'x': {0: torch.export.Dim("batch")}  # 'x' 是模型 forward 方法的参数名
    }

    torch.onnx.export(
        model,
        dummy_input,  # 直接传递张量，不要转换为列表
        model_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_shapes=dynamic_shapes
    )

    print(f"✅ ONNX 模型已导出: {model_onnx_path}")
    return model_onnx_path


def verify_onnx_model(onnx_path, dummy_input, original_model):
    """验证导出的 ONNX 模型"""
    import onnxruntime as ort
    import numpy as np

    # 原始模型输出
    with torch.no_grad():
        original_output = original_model(dummy_input).numpy()

    # ONNX 模型输出
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_output = ort_session.run(['output'], {'input': dummy_input.numpy()})[0]

    # 比较差异
    max_diff = np.abs(original_output - onnx_output).max()
    print(f"🔍 原始模型 vs ONNX 最大差异: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("✅ 验证通过，模型导出正确")
    else:
        print("⚠️ 差异较大，请检查")
        print(f"原始输出前5个值: {original_output[0][:5]}")
        print(f"ONNX输出前5个值: {onnx_output[0][:5]}")


if __name__ == '__main__':
    model = load_model_from_state_dict()
    # export_to_onnx(model)
    verify_onnx_model(model_output_path.joinpath("flower_model.onnx"), torch.randn(1, 3, 224, 224), model)

