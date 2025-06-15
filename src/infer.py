import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset import CustomSegmentationDataset
import albumentations as A
from PIL import Image
import cv2


def inference(model, image, original_size=None, device='cpu'):
    model.eval()
    with torch.no_grad():
        # imageがNumPy配列の場合: Tensorに変換
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

        # imageが3次元 (C, H, W) の場合、バッチ次元 (N=1) を追加
        if image.ndim == 3:
            image = image.unsqueeze(0).float()

        image = image.to(device)
        output = model(image)['out']
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # 元のサイズに戻す
        if original_size is not None:
            # マスクをuint8に変換
            pred = pred.astype(np.uint8)
            
            # albumentationsのリサイズ処理
            transform = A.Compose([
                A.LongestMaxSize(max_size=max(original_size)),
                A.PadIfNeeded(
                    min_height=original_size[0],
                    min_width=original_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            ])
            transformed = transform(image=pred)
            pred = transformed['image']

    return pred

class InferenceVisualizer:
    def __init__(self, dataset):
        """
        推論結果の可視化と保存を行うクラス
        
        Args:
            dataset: 検証データセット
        """
        self.dataset = dataset

    def visualize_and_save(self, pred_masks, output_dir, save_masks_only=False):
        """
        推論結果を可視化して保存する
        
        Args:
            pred_masks (list): 推論結果のマスク画像のリスト
            output_dir (str): 出力ディレクトリのパス
            save_masks_only (bool): マスクのみを保存する場合はTrue
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx, pred_mask in enumerate(pred_masks):
            # 画像と正解マスクを取得
            val_img, val_mask = self.dataset[idx]
            file_name = self.dataset.annotations_data[idx]['file_name']
            
            # 元のサイズにリサイズ
            original_size = (self.dataset.annotations_data[idx]['height'], 
                           self.dataset.annotations_data[idx]['width'])
            
            # マスクをuint8に変換
            pred_mask = pred_mask.astype(np.uint8)
            
            transform = A.Compose([
                A.LongestMaxSize(max_size=max(original_size)),
                A.PadIfNeeded(
                    min_height=original_size[0],
                    min_width=original_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )
            ])
            transformed = transform(image=pred_mask)
            pred_mask = transformed['image']
            
            if save_masks_only:
                # マスクのみを保存
                plt.figure(figsize=(5,5))
                plt.imshow(pred_mask)
                plt.axis('off')
                
                output_path = os.path.join(output_dir, f"mask_{file_name}")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            else:
                # 画像、正解マスク、予測マスクを並べて保存
                plt.figure(figsize=(15,5))
                plt.subplot(1,3,1)
                plt.title("Image")
                plt.imshow(val_img.permute(1, 2, 0).cpu().numpy())
                plt.subplot(1,3,2)
                plt.title("Ground Truth")
                plt.imshow(val_mask)
                plt.subplot(1,3,3)
                plt.title("Prediction")
                plt.imshow(pred_mask)
                
                output_path = os.path.join(output_dir, f"prediction_{file_name}")
                plt.savefig(output_path)
            
            plt.close()

        print(f"推論結果を {output_dir} に保存しました。") 