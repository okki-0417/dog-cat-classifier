from typing import cast

import torch
import torchvision.transforms as transforms
from PIL import Image

# ImageNet正規化パラメータ
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = cast(torch.Tensor, transform(image))
    return tensor.unsqueeze(0)
