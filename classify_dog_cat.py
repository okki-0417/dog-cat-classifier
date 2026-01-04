import sys
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


DOG_CLASS_INDEXES_IN_IMAGENET = range(151, 269)
CAT_CLASS_INDEXES_IN_IMAGENET = range(281, 286)


def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def classify_dog_or_cat(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    total_dog_probability = sum(probabilities[i].item() for i in DOG_CLASS_INDEXES_IN_IMAGENET)
    total_cat_probability = sum(probabilities[i].item() for i in CAT_CLASS_INDEXES_IN_IMAGENET)

    return total_dog_probability, total_cat_probability


def main():
    if len(sys.argv) < 2:
        print("使い方: python classify_dog_cat.py <画像ファイルパス>")
        print("例: python classify_dog_cat.py dog.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print("モデルを読み込み中...")
    model = load_model()

    print(f"画像を分析中: {image_path}")
    image_tensor = preprocess_image(image_path)

    dog_probability, cat_probability = classify_dog_or_cat(model, image_tensor)

    print("\n=== 判定結果 ===")
    print(f"犬の確率: {dog_probability * 100:.1f}%")
    print(f"猫の確率: {cat_probability * 100:.1f}%")

    if dog_probability > cat_probability:
        print("\n判定: 犬 🐕")
    elif cat_probability > dog_probability:
        print("\n判定: 猫 🐱")
    else:
        print("\n判定: どちらとも言えません")


if __name__ == "__main__":
    main()
