import json
import numpy as np
import os
import cv2

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
    # 結果を格納するリスト
    images_data = []
    
    # 各マスクを処理
    for i, (mask, file_name) in enumerate(zip(masks, file_names)):
        # マスクからポリゴンを抽出
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ポリゴンの座標を1次元配列に変換
        segmentation = []
        for contour in contours:
            # 輪郭の座標を1次元配列に変換
            points = contour.reshape(-1, 2)
            segmentation.extend(points.flatten().tolist())
        
        # 画像データを作成
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
    
    # 最終的なJSONフォーマットに変換
    result = {
        "images": images_data
    }
    
    # JSONファイルを保存
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

def single_mask_to_json(mask, file_name, output_path, width=None, height=None):
    """
    単一のマスク画像をJSONフォーマットに変換して保存する（後方互換性のため）
    
    Args:
        mask (numpy.ndarray): マスク画像 (H, W)
        file_name (str): 画像ファイル名
        output_path (str): 出力JSONファイルのパス
        width (int, optional): 画像の幅
        height (int, optional): 画像の高さ
    """
    mask_to_json([mask], [file_name], output_path, [width] if width is not None else None, [height] if height is not None else None)