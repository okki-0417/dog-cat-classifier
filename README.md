# dog-cat-classifier

犬か猫かを判定する画像分類プロジェクト。PyTorchを使用。

モデルの仕組みを学びながら、段階的にレベルアップしていく。

## 現在の実装

事前学習済みResNet18（ImageNet）を使用した犬猫判定。

## ロードマップ

- [x] Level 1: 事前学習済みモデルをそのまま使う
- [ ] Level 2: 転移学習（ResNetをベースに犬猫データで再学習）
- [ ] Level 3: 犬種・猫種を細かく分類
- [ ] Level 4: シンプルなCNNをゼロから作成
- [ ] Level 5: 自作CNNの精度改善
- [ ] Level 6: 最新アーキテクチャに挑戦（Vision Transformerなど）

## セットアップ

```bash
# 仮想環境を作成・有効化
python3 -m venv venv
source venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

## 使い方

### 犬猫判定

```bash
python classify_dog_cat.py <画像ファイル>
```

出力例:
```
=== 判定結果 ===
犬の確率: 89.3%
猫の確率: 0.2%

判定: 犬
```

### モデルの中身を確認

```bash
python inspect_model.py
```

## 仕組み（Level 1）

1. 画像を前処理（リサイズ、正規化）
2. ResNet18で1000クラスの確率を計算
3. 犬クラス（ID: 151-268）と猫クラス（ID: 281-285）の確率を合計
4. 高い方を判定結果として出力
