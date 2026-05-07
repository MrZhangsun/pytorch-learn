import onnxruntime as ort
import numpy as np

# 1️⃣ 加载模型
session = ort.InferenceSession("assets/fraud_model.onnx")

# 2️⃣ 构造输入（必须 float32）
x = np.random.randn(1, 10).astype(np.float32)

print(x)
# 3️⃣ 推理
outputs = session.run(
    None,                 # 输出全部
    {"input": x}          # ⚠️ input 要和你导出时一致
)

print(outputs)