import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset import CustomSegmentationDataset


def inference(model, image, device='cpu'):
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