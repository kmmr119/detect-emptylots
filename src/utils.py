import os
import json
import shutil
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F

def calculate_metrics(pred, target, num_classes, smooth=1e-6):
    """
    セグメンテーションの評価指標を計算する
    
    Args:
        pred (torch.Tensor): 予測値 (B, C, H, W)
        target (torch.Tensor): 正解値 (B, H, W)
        num_classes (int): クラス数
        smooth (float): 数値安定性のための平滑化項
    
    Returns:
        dict: 各指標の値
    """
    # 予測値をクラスインデックスに変換
    pred = torch.argmax(pred, dim=1)
    
    # 各クラスごとの指標を計算
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'pixel_accuracy': []
    }
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # 共通部分の計算
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        pred_sum = pred_cls.sum().float()
        target_sum = target_cls.sum().float()
        
        # Dice係数
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        
        # IoU
        iou = (intersection + smooth) / (union + smooth)
        
        # Precision
        precision = (intersection + smooth) / (pred_sum + smooth)
        
        # Recall
        recall = (intersection + smooth) / (target_sum + smooth)
        
        # Pixel Accuracy
        pixel_acc = (pred_cls == target_cls).float().mean()
        
        # 結果を保存
        metrics['dice'].append(dice)
        metrics['iou'].append(iou)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['pixel_accuracy'].append(pixel_acc)
    
    # 全クラスの平均を計算
    for key in metrics:
        metrics[key] = torch.mean(torch.tensor(metrics[key]))
    
    return metrics

class JsonUtils:
    @staticmethod
    def mask_to_json(masks, file_names, output_path, widths=None, heights=None):
        """
        複数のマスク画像を1つのJSONフォーマットに変換して保存する
        
        Args:
            masks (list): マスク画像のリスト [(H, W) numpy配列, ...]
            file_names (list): 画像ファイル名のリスト
            output_path (str): 出力JSONファイルのパス
            widths (list, optional): 画像の幅のリスト
            heights (list, optional): 画像の高さのリスト
        """
        images_data = []
        
        for i, (mask, file_name) in enumerate(zip(masks, file_names)):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []
            for contour in contours:
                points = contour.reshape(-1, 2)
                segmentation.extend(points.flatten().tolist())
            
            image_data = {
                "file_name": file_name,
                "width": widths[i] if widths is not None else mask.shape[1],
                "height": heights[i] if heights is not None else mask.shape[0],
                "annotations": [{
                    "class": "vacant_lot",
                    "segmentation": segmentation
                }]
            }
            images_data.append(image_data)
        
        result = {"images": images_data}
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

    @staticmethod
    def single_mask_to_json(mask, file_name, output_path, width=None, height=None):
        """
        単一のマスク画像をJSONフォーマットに変換して保存する
        
        Args:
            mask (numpy.ndarray): マスク画像 (H, W)
            file_name (str): 画像ファイル名
            output_path (str): 出力JSONファイルのパス
            width (int, optional): 画像の幅
            height (int, optional): 画像の高さ
        """
        JsonUtils.mask_to_json(
            [mask], 
            [file_name], 
            output_path, 
            [width] if width is not None else None, 
            [height] if height is not None else None
        )

class DatasetSplitter:
    def __init__(self, image_dir, annotation_file, output_dir, image_ext=".tif"):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.output_dir = output_dir
        self.image_ext = image_ext

    def split(self, test_size=0.2, random_state=42):
        with open(self.annotation_file, 'r') as f:
            annotations_data = json.load(f)

        images = annotations_data['images']
        train_images, val_images = train_test_split(
            images, test_size=test_size, random_state=random_state
        )

        # ディレクトリ構造の作成
        dirs = {
            'train': {'img': 'train/images', 'ann': 'train/annotations'},
            'val': {'img': 'val/images', 'ann': 'val/annotations'}
        }
        
        for split in dirs.values():
            for path in split.values():
                os.makedirs(os.path.join(self.output_dir, path), exist_ok=True)

        # データの分割と保存
        for split_name, split_data in [('train', train_images), ('val', val_images)]:
            # 画像のコピー
            for entry in split_data:
                src_path = os.path.join(self.image_dir, entry['file_name'])
                dst_path = os.path.join(self.output_dir, dirs[split_name]['img'], entry['file_name'])
                shutil.copy(src_path, dst_path)

            # アノテーションの保存
            ann_path = os.path.join(self.output_dir, dirs[split_name]['ann'], "annotations.json")
            with open(ann_path, 'w') as f:
                json.dump({"images": split_data}, f, indent=4)

        print(f"✅ Split completed. Train: {len(train_images)}, Val: {len(val_images)}")
