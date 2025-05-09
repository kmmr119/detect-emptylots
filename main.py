import os
import cv2
import numpy as np
from src.data.dataset import EmptyLotDataset
from src.models.detector import EmptyLotDetector

def main():
    # データセットの初期化
    data_dir = "raw_data"  # データセットのディレクトリ
    dataset = EmptyLotDataset(data_dir)
    
    # モデルの初期化
    model = EmptyLotDetector()
    
    # サンプル画像の処理
    for image_path in dataset.image_paths[:5]:  # 最初の5枚の画像を処理
        # 画像の読み込み
        image = dataset.load_image(image_path)
        
        # 画像の前処理
        processed_image = dataset.preprocess_image(image)
        
        # 空き地の検出
        empty_lots = model.detect_empty_lots(processed_image)
        
        # 結果の表示
        print(f"\n画像: {os.path.basename(image_path)}")
        print(f"検出された空き地の数: {len(empty_lots)}")
        for i, lot in enumerate(empty_lots, 1):
            print(f"空き地 {i}:")
            print(f"  位置: {lot['bbox']}")
            print(f"  確率: {lot['probability']:.2f}")

if __name__ == "__main__":
    main()
