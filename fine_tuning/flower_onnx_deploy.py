import requests
import numpy as np
from PIL import Image
from pathlib import Path
from flower_fine_tuning import build_transforms

"""
部署方案： ONNX Runtime for Java + Spring Boot 的企业级生产部署方案
方案落地过程：
1. 编写Vgg19_bn微调模型，冻结卷积层，替换最后一层特征；
2. 加载企业自由数据，进行数据预处理；
3. 进行模型微调训练，观察损失，Early Stopping，保存模型；
4. 模型导出为ONNX格式（包含推理图和/模型参数数据两个文件）
5. 推理引擎使用ONNX Runtime，需要在部署的机器上安装好
6. 采用Java + Spring Boot作为API推理服务；
7. Python复用训练阶段的数据预处理transformer对预测数据进行预处理；
8. 调用Java Http API进行预测。
"""

def predict(image_path, url="http://localhost:8080/api/v1/predict"):
    # 1. 加载图片
    img = Image.open(image_path).convert('RGB')

    # 2. ✅ 直接用训练时的 transform_val 处理
    _, transform_val = build_transforms()
    img_tensor = transform_val(img)  # 形状: [3, 448, 448]

    # 3. 添加批次维度: [3, 448, 448] → [1, 3, 448, 448]
    img_tensor = img_tensor.unsqueeze(0)

    # 4. 转为 list 传给接口
    input_data = img_tensor.numpy().tolist()

    payload = {"data": input_data}

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        probabilities = result['prediction'][0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # 类别映射
        labels = ['c1', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17',
                  'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

        print(f"预测类别: {predicted_class} ({labels[predicted_class]})")
        print(f"置信度: {confidence:.4f}")
        print(f"推理耗时: {result['inferenceTimeMs']}ms")

        # 打印 Top-3 预测
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        print("\nTop-3 预测:")
        for idx in top3_idx:
            print(f"  {labels[idx]}: {probabilities[idx]:.4f}")
    else:
        print(f"预测失败: {response.status_code}, {response.text}")


if __name__ == "__main__":
    flower_root_dir = Path(__file__).parent.parent.joinpath("202601_CV/20260412/17flowers/c14/image_1041.jpg")
    # 替换为你的测试图片路径
    predict(flower_root_dir)