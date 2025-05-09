import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmptyLotDetector(nn.Module):
    def __init__(self):
        """
        空き地検出モデル
        """
        super().__init__()
        
        # CNNの基本構造
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # 全結合層
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x (torch.Tensor): 入力画像テンソル
            
        Returns:
            torch.Tensor: 予測結果
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
    def predict(self, image: np.ndarray) -> float:
        """
        画像から空き地の確率を予測
        
        Args:
            image (np.ndarray): 入力画像
            
        Returns:
            float: 空き地である確率
        """
        self.eval()
        with torch.no_grad():
            # 画像の前処理
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
            
            # 予測
            output = self(image)
            return output.item()
    
    def detect_empty_lots(self, image: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        画像内の空き地を検出
        
        Args:
            image (np.ndarray): 入力画像
            threshold (float): 検出閾値
            
        Returns:
            List[Dict[str, Any]]: 検出された空き地の情報
        """
        # 画像をグリッドに分割
        grid_size = 224
        height, width = image.shape[:2]
        
        empty_lots = []
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # グリッドの切り出し
                grid = image[y:y+grid_size, x:x+grid_size]
                if grid.shape[:2] != (grid_size, grid_size):
                    continue
                
                # 予測
                prob = self.predict(grid)
                if prob > threshold:
                    empty_lots.append({
                        'bbox': (x, y, x+grid_size, y+grid_size),
                        'probability': prob
                    })
        
        return empty_lots 