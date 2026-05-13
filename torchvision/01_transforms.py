import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


def image_classification(image):
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias= True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transforms(image)
    return image

def image_detection(image: torch.Tensor):
    """
    模拟目标检测（Object Detection）任务中的数据增强过程：当你对图片进行裁剪、翻转时，
    图片里物体的标注框（Bounding Boxes）也会自动跟着调整位置。
    """
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),  # 随机裁剪并缩放到 224x224
        v2.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
        v2.ToDtype(torch.float32, scale=True),  # 转为浮点数并归一化到 0-1
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化（减去均值，除以方差）
    ])
    c, h, w = image.shape
    # 随机生成 3 个框的左上角坐标 (x1, y1, x2, y2)
    boxes = torch.randint(0, h // 2, size=(3, 4))
    boxes[:, 2:] += boxes[:, :2]  # 确保右下角坐标 (x2, y2) 大于左上角
    print(boxes)
    """
    这是最关键的一步！
    普通的 Tensor 只是数字，但 tv_tensors.BoundingBoxes 告诉 PyTorch：“这些数字代表坐标，并且它们是依附于这张 (h, w) 大小的图片的。”
    有了这个“身份认证”，后面的 transforms 才知道在裁剪图片时，该怎么去修改这些坐标。
    """
    boxes = tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=(h, w))
    print(boxes)

    """
    你把图片和框作为一个字典传进去。
    神奇的效果：
    如果 RandomResizedCrop 切掉了框的一部分，框的坐标会自动缩小或偏移。
    如果 RandomHorizontalFlip 翻转了图片，框的左右坐标（x1, x2）也会自动对调。
    """
    outputs = transforms({"image": image, "boxes": boxes})
    print(outputs)

if __name__ == '__main__':
    H, W = 32, 32
    image_random = torch.randint(0, 255, size=(3, H, W), dtype=torch.uint8)
    image_detection(image_random)