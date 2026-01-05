# text-generator

文字レベルRNNによるテキスト生成プロジェクト。

## 仕組み

1. テキストを1文字ずつ読み込む
2. 「この文字の次は何か」を学習
3. 学習したパターンで新しい文章を生成

## セットアップ

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使い方

```bash
# 学習 & 生成
python -m text_generator sample.txt --epochs 50

# オプション
python -m text_generator sample.txt --epochs 100 --start "The " --length 500
```

## ロードマップ

- [x] Level 1: 文字レベルRNN (LSTM)
- [ ] Level 2: 単語レベルRNN
- [ ] Level 3: Attention機構の追加
- [ ] Level 4: 簡易Transformer
