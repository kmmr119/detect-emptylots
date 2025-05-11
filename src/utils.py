import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self, image_dir, annotation_file, output_dir, image_ext=".tif"):
        self.image_dir = image_dir
        self.annotation_file = annotation_file  # 単一のJSONファイル
        self.output_dir = output_dir
        self.image_ext = image_ext

    def split(self, test_size=0.2, random_state=42):
        # アノテーションファイルを読み込む
        with open(self.annotation_file, 'r') as f:
            annotations_data = json.load(f)

        images = annotations_data['images']
        image_filenames = [entry['file_name'] for entry in images]

        # 学習用と評価用に分割
        train_images, val_images = train_test_split(
            images, test_size=test_size, random_state=random_state
        )

        # 保存ディレクトリ作成
        train_img_dir = os.path.join(self.output_dir, "train", "images")
        val_img_dir = os.path.join(self.output_dir, "val", "images")
        train_ann_dir = os.path.join(self.output_dir, "train", "annotations")
        val_ann_dir = os.path.join(self.output_dir, "val", "annotations")
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_ann_dir, exist_ok=True)
        os.makedirs(val_ann_dir, exist_ok=True)

        train_ann_path = os.path.join(train_ann_dir, "annotations.json")
        val_ann_path = os.path.join(val_ann_dir, "annotations.json")

        # 画像をコピー
        for entry in train_images:
            shutil.copy(
                os.path.join(self.image_dir, entry['file_name']),
                os.path.join(train_img_dir, entry['file_name'])
            )

        for entry in val_images:
            shutil.copy(
                os.path.join(self.image_dir, entry['file_name']),
                os.path.join(val_img_dir, entry['file_name'])
            )

        # 対応するアノテーションを書き出し
        with open(train_ann_path, 'w') as f:
            json.dump({"images": train_images}, f, indent=4)

        with open(val_ann_path, 'w') as f:
            json.dump({"images": val_images}, f, indent=4)

        print(f"✅ Split completed. Train: {len(train_images)}, Val: {len(val_images)}")
