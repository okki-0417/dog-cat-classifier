import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CharRNN
from .data import TextDataset


def train(
    text: str,
    epochs: int = 50,
    batch_size: int = 64,
    seq_length: int = 100,
    hidden_size: int = 128,
    lr: float = 0.002,
    resume_from: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # データ準備
    dataset = TextDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"語彙数: {dataset.vocab_size}文字")
    print(f"データ長: {len(dataset.data)}文字")
    print(f"バッチ数: {len(dataloader)}/エポック")

    # モデル作成または読み込み
    if resume_from:
        print(f"モデル読み込み: {resume_from}")
        checkpoint = torch.load(resume_from, weights_only=False)
        model = CharRNN(dataset.vocab_size, checkpoint["hidden_size"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = CharRNN(dataset.vocab_size, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学習ループ
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx % 500 == 0:
                print(f"\r  Epoch {epoch+1}: {batch_idx}/{len(dataloader)}", end="", flush=True)
            x, y = x.to(device), y.to(device)

            # 順伝播
            output, _ = model(x)

            # 損失計算（出力と正解の差）
            loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))

            # 逆伝播（勾配計算）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}      ")

    # モデルを保存
    save_path = "model.pth"
    torch.save({
        "model_state": model.state_dict(),
        "chars": dataset.chars,
        "hidden_size": hidden_size,
    }, save_path)
    print(f"モデルを保存: {save_path}")

    return model, dataset
