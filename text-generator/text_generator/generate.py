import torch

from .model import CharRNN
from .data import TextDataset


def generate(
    model: CharRNN,
    dataset: TextDataset,
    start_text: str = "The ",
    length: int = 200,
    temperature: float = 0.8,
) -> str:
    """
    学習済みモデルで文章を生成する

    temperature:
        低い(0.5) → 確実な予測、単調な文章
        高い(1.2) → ランダム性が増す、多様な文章
    """
    device = next(model.parameters()).device
    model.eval()

    # 開始テキストをインデックスに変換
    chars = dataset.encode(start_text).unsqueeze(0).to(device)
    hidden = None

    generated = start_text

    with torch.no_grad():
        # まず開始テキストを処理して文脈を学習
        for i in range(len(start_text) - 1):
            _, hidden = model(chars[:, i : i + 1], hidden)

        # 1文字ずつ生成
        current_char = chars[:, -1:]
        for _ in range(length):
            output, hidden = model(current_char, hidden)

            # temperatureで確率分布を調整
            probs = torch.softmax(output[0, 0] / temperature, dim=0)

            # 確率に従ってサンプリング
            next_idx = torch.multinomial(probs, 1)
            next_char = dataset.idx_to_char[next_idx.item()]

            generated += next_char
            current_char = next_idx.unsqueeze(0)

    return generated
