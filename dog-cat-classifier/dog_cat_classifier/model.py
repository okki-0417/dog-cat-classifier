from torchvision import models

# ImageNetにおける犬・猫のクラスID範囲
DOG_CLASS_RANGE = range(151, 269)
CAT_CLASS_RANGE = range(281, 286)


def load_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, categories
