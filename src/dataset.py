import torch
from torch.utils.data import Dataset
import rasterio
import json
import numpy as np
import os
import cv2
import torchvision.transforms.functional as TF
from PIL import Image

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, label_map, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = label_map  # 例: {"vacant_lot": 1}
        self.target_size = target_size

        # アノテーションJSON読み込み
        with open(annotation_file, 'r') as f:
            self.annotations_data = json.load(f)['images']

    def __len__(self):
        return len(self.annotations_data)

    def __getitem__(self, idx):
        entry = self.annotations_data[idx]
        file_name = entry['file_name']
        width = entry['width']
        height = entry['height']
        annotations = entry.get('annotations', [])

        # 画像読み込み
        image_path = os.path.join(self.image_dir, file_name)
        with rasterio.open(image_path) as src:
            image = src.read().transpose(1, 2, 0).astype(np.uint8)

        # マスク作成
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations:
            class_name = ann['class']
            class_id = self.label_map.get(class_name, 0)
            segmentation = ann['segmentation']
            points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [points.astype(np.int32)], class_id)

        # オーグメンテーション or リサイズ
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # PILに変換してリサイズ（maskはNEAREST）
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            image = TF.resize(image, self.target_size, interpolation=Image.BILINEAR)
            mask = TF.resize(mask, self.target_size, interpolation=Image.NEAREST)

            image = TF.to_tensor(image)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_size=(256, 256)):
        """
        テストデータを読み込むためのデータセットクラス
        
        Args:
            image_dir (str): テスト画像が格納されているディレクトリのパス
            transform: 画像の変換処理（オプション）
            target_size (tuple): リサイズ後の画像サイズ (height, width)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        
        # 画像ファイルのリストを取得
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff'))]
        self.image_files.sort()  # ファイル名でソート

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        
        # 画像読み込み
        image_path = os.path.join(self.image_dir, file_name)
        with rasterio.open(image_path) as src:
            image = src.read().transpose(1, 2, 0).astype(np.uint8)
            height, width = image.shape[:2]

        # オーグメンテーション or リサイズ
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # PILに変換してリサイズ
            image = Image.fromarray(image)
            image = TF.resize(image, self.target_size, interpolation=Image.BILINEAR)
            image = TF.to_tensor(image)

        return image, file_name