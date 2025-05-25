# 空き地検出プロジェクト

このプロジェクトは、画像から空き地を検出するためのツールセットです。データの前処理、分析、学習、推論の機能を提供します。

## プロジェクト構造

```
src/
├── utils.py          # ユーティリティ機能
├── data_check.py     # データ分析機能
├── model.py          # モデル定義
├── train.py          # 学習処理
└── infer.py          # 推論処理
```

## 機能概要

### 1. ユーティリティ機能 (`utils.py`)

#### JsonUtils クラス
- マスク画像をJSONフォーマットに変換する機能を提供
- 単一または複数のマスク画像を処理可能
- アノテーションデータの形式を統一

#### DatasetSplitter クラス
- データセットを学習用と検証用に分割
- 画像とアノテーションファイルを適切なディレクトリ構造で保存
- ランダムな分割をサポート

### 2. データ分析機能 (`data_check.py`)

#### BaseAnalyzer クラス
- すべての分析クラスの基底クラス
- 共通のインターフェースを定義
- データの読み込み、分析、結果表示の基本機能を提供

#### ImageSizeAnalyzer クラス
- 画像サイズの統計分析
- 幅と高さの分布を可視化
- 箱ひげ図と散布図による詳細な分析
- サイズの種類と頻度の集計

#### PositivePixelAnalyzer クラス
- マスク画像のポジティブピクセル比率の分析
- 累積分布の可視化
- オーバーレイ画像の生成
- 統計情報の出力

### 3. モデル関連

#### model.py
- 検出モデルの定義
- アーキテクチャの設定

#### train.py
- モデルの学習処理
- 学習パラメータの設定
- 学習の進捗管理

#### infer.py
- 学習済みモデルを使用した推論
- 検出結果の出力

## 使用方法

### データの前処理
```python
from src.utils import DatasetSplitter

# データセットの分割
splitter = DatasetSplitter(
    image_dir="path/to/images",
    annotation_file="path/to/annotations.json",
    output_dir="path/to/output"
)
splitter.split(test_size=0.2)
```

### データの分析
```python
from src.data_check import ImageSizeAnalyzer, PositivePixelAnalyzer

# 画像サイズの分析
size_analyzer = ImageSizeAnalyzer("path/to/images")
size_analyzer.analyze()
size_analyzer.print_summary()
size_analyzer.plot_histograms()

# ポジティブピクセルの分析
pixel_analyzer = PositivePixelAnalyzer(
    annotation_path="path/to/annotations.json",
    image_dir="path/to/images"
)
pixel_analyzer.analyze()
pixel_analyzer.print_summary()
pixel_analyzer.plot_histogram()
```

## 依存パッケージ

- numpy
- opencv-python (cv2)
- Pillow (PIL)
- matplotlib
- seaborn
- scikit-learn
- tqdm

## 注意事項

- 画像は`.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff`形式に対応
- アノテーションデータは特定のJSONフォーマットに従う必要があります
- メモリ使用量に注意（特に大量の画像を処理する場合）