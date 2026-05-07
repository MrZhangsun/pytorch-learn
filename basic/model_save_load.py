import torch
import torchvision.models as models
from pathlib import Path

dataset_save_path = Path(__file__).parent.parent.joinpath("assets/data")
"""
模型参数保存
"""
def save_model_params():
    # 加载预训练的 VGG16 模型
    model = models.vgg16(weights='IMAGENET1K_V1')
    # 保存模型参数
    torch.save(model.state_dict(), dataset_save_path.joinpath('model_weights.pth'))


"""
模型参数加载
"""
def load_model_params():
    # 不指定weights参数，则加载的模型参数为随机初始化，即：没有训练的模型
    untrain_model = models.vgg16()
    untrain_model.load_state_dict(torch.load(dataset_save_path.joinpath('model_weights.pth')))
    untrain_model.eval()


def load_random_model():
    """创建随机初始化的未训练模型"""
    model = models.vgg16()  # weights=None，随机初始化
    model.eval()
    print("创建了随机初始化的模型（未训练）")
    return model


def load_pretrained_model():
    """加载官方预训练模型（在ImageNet上训练好的）"""
    # PyTorch官方提供的预训练权重
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    print("加载了官方预训练模型")
    return model


def load_custom_trained_model():
    """加载自己训练保存的模型"""
    model = models.vgg16()

    # 加载自定义权重
    weight_path = dataset_save_path.joinpath('model_weights.pth')
    model.load_state_dict(torch.load(weight_path))
    model.eval() # 将模型切换到评估/推理模式
    print("加载了自定义训练的模型")
    return model