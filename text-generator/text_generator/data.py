import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """テキストを学習用データに変換する"""

    def __init__(self, text: str, seq_length: int = 100):
        self.seq_length = seq_length

        # 文字の一覧を作成
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        # 文字 ↔ インデックスの変換辞書
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # テキスト全体をインデックス列に変換
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text])

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # 入力: seq_length文字
        # 正解: 1文字ずらした seq_length文字（次の文字を予測させる）
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """文字列 → インデックス列"""
        return torch.tensor([self.char_to_idx[ch] for ch in text])

    def decode(self, indices: torch.Tensor) -> str:
        """インデックス列 → 文字列"""
        return "".join(self.idx_to_char[i.item()] for i in indices)
