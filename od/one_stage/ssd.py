from torchvision.models import detection

def ssd_model():
    ssd = detection.ssd300_vgg16(weights=None)
    print(ssd)


if __name__ == '__main__':
    ssd_model()