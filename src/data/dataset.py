import os
import cv2
import numpy as np
from typing import Tuple, List, Optional

class EmptyLotDataset:
    def __init__(self, data_dir: str):
        """
        空き地データセットを管理するクラス
        
        Args:
            data_dir (str): データセットのルートディレクトリ
        """
        self.data_dir = data_dir
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self) -> List[str]:
        """画像ファイルのパスを取得"""
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
                    
        return image_paths
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        画像を読み込む
        
        Args:
            image_path (str): 画像ファイルのパス
            
        Returns:
            np.ndarray: 読み込んだ画像
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        画像の前処理を行う
        
        Args:
            image (np.ndarray): 入力画像
            target_size (Tuple[int, int]): リサイズ後のサイズ
            
        Returns:
            np.ndarray: 前処理済みの画像
        """
        # リサイズ
        resized = cv2.resize(image, target_size)
        # 正規化
        normalized = resized.astype(np.float32) / 255.0
        return normalized 