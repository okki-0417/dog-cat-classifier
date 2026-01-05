import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """
    文字レベルRNN

    仕組み:
    1. 文字をベクトルに変換（embedding）
    2. LSTMで順番に処理して文脈を学習
    3. 次の文字の確率を出力
    """

    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 文字 → ベクトル
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # LSTM: 時系列データを処理する層
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # ベクトル → 次の文字の確率
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) 文字インデックスの列
        hidden: LSTMの隠れ状態（前の文脈を覚えている）
        """
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """隠れ状態をゼロで初期化"""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
