"""
ResNet18モデルの中身を覗いてみるスクリプト
"""

import torch


def main():
    # モデルの重みを読み込む
    checkpoint_path = torch.hub.get_dir() + "/checkpoints/resnet18-f37072fd.pth"
    weights = torch.load(checkpoint_path, weights_only=True)

    print("=== レイヤー一覧 ===")
    for key in weights.keys():
        print(f"{key}: {weights[key].shape}")

    print(f"\n合計レイヤー数: {len(weights)}")

    # 最初の畳み込み層の重みを少し見る
    print("\n=== conv1.weight の中身（一部） ===")
    print(weights["conv1.weight"][0, 0, :3, :3])

    # パラメータ総数
    total = sum(w.numel() for w in weights.values())
    print(f"\n総パラメータ数: {total:,} 個")


if __name__ == "__main__":
    main()
