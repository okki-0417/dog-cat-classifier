import sys

from .model import load_model
from .preprocess import preprocess_image
from .classifier import classify
from .presenter import print_result


def main():
    if len(sys.argv) < 2:
        print("使い方: python -m dog_cat_classifier <画像ファイルパス>")
        print("例: python -m dog_cat_classifier dog.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    model, categories = load_model()
    image_tensor = preprocess_image(image_path)
    result = classify(model, image_tensor, categories)

    print_result(result)


if __name__ == "__main__":
    main()
