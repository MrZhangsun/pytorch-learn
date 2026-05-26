from thop import profile, clever_format
import torch.nn as nn
import torch
import torchvision.models as models

input_tensor = torch.randn(1, 3, 224, 224)
for model_name in models.list_models():
    try:
        # 获取模型
        fn = getattr(models, model_name)
        model = fn(weights=None)

        # 计算模型参数和FLOPs
        flops, params = profile(
            model,
            inputs=(input_tensor,),
            verbose=False
        )

        # 格式化
        flops, params = clever_format(
            [flops, params],
            "%.3f"
        )

        print(f"{model_name:<30} "
              f"Params: {params:<10} "
              f"FLOPs: {flops}")

    except Exception as e:
        print(f"{model_name:<30} ERROR: {e}")