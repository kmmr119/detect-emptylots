import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

import json
import cv2
from tqdm import tqdm

class ImageSizeAnalyzer:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.widths = []
        self.heights = []
        self.size_counter = {}
    
    def analyze(self):
        """画像フォルダ内の全画像のサイズを取得し、統計を準備"""
        self.widths = []
        self.heights = []
        self.size_counter = {}
        
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                path = os.path.join(self.image_dir, filename)
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        self.widths.append(w)
                        self.heights.append(h)
                        key = (w, h)
                        self.size_counter[key] = self.size_counter.get(key, 0) + 1
                except Exception as e:
                    print(f'読み込みエラー: {filename}, エラー: {e}')
    
    def print_summary(self, top_n=5):
        """画像サイズの統計情報を出力"""
        print(f'総画像数: {len(self.widths)}')
        print(f'画像サイズの種類数: {len(self.size_counter)}')
        print(f'最も多いサイズ TOP{top_n}:')
        for size, count in sorted(self.size_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f'  サイズ: {size}, 枚数: {count}')
    
    def plot_histograms(self):
        """Plot histograms of image widths and heights"""
        if not self.widths or not self.heights:
            print("Please run analyze() first.")
            return

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(self.widths, bins=30, kde=True)
        plt.title('Distribution of Image Widths')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        sns.histplot(self.heights, bins=30, kde=True)
        plt.title('Distribution of Image Heights')
        plt.xlabel('Height (pixels)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    
    def plot_size_scatter(self):
        """Scatter plot of image sizes with marginal boxplots and annotated quartiles"""
        if not self.widths or not self.heights:
            print("Please run analyze() first.")
            return

        widths = np.array(self.widths)
        heights = np.array(self.heights)

        # 箱ひげ図の統計量を計算
        def get_box_stats(arr):
            return {
                'min': np.min(arr),
                'q1': np.percentile(arr, 25),
                'median': np.median(arr),
                'q3': np.percentile(arr, 75),
                'max': np.max(arr)
            }

        w_stats = get_box_stats(widths)
        h_stats = get_box_stats(heights)

        # グリッドの設定
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[1, 5],
                            wspace=0.05, hspace=0.05)

        ax_box_x = plt.subplot(gs[0, 1])
        ax_box_y = plt.subplot(gs[1, 0])
        ax_scatter = plt.subplot(gs[1, 1])

        # 上部 boxplot（Width）
        sns.boxplot(x=widths, ax=ax_box_x, color='lightblue')
        ax_box_x.set(xlabel='', yticks=[])
        ax_box_x.xaxis.tick_top()  # 上に目盛りを表示
        ax_box_x.xaxis.set_label_position('top')  # ラベル位置も上へ

        for key, value in w_stats.items():
            ax_box_x.text(value, 0.05, f'{int(value)}', rotation=90,
                        va='bottom', ha='center', fontsize=9)

        # 左 boxplot（Height）縦表示
        sns.boxplot(y=heights, ax=ax_box_y, color='lightgreen')
        ax_box_y.set(ylabel='', xticks=[])
        ax_box_y.set_xticks([])
        for key, value in h_stats.items():
            ax_box_y.text(0.05, value, f'{int(value)}',
                        va='center', ha='left', fontsize=9)

        # 散布図（サイズ）
        sns.scatterplot(x=widths, y=heights, alpha=0.6, ax=ax_scatter)
        ax_scatter.set_xlabel('Width (pixels)', fontsize=11)

        # 縦軸のラベルと目盛りを右に移動
        ax_scatter.yaxis.tick_right()
        ax_scatter.yaxis.set_label_position('right')
        ax_scatter.set_ylabel('Height (pixels)', fontsize=11)

        ax_scatter.grid(True)

        # min/max 補助線とラベル
        label_offset = 10
        ax_scatter.axvline(w_stats['min'], color='red', linestyle='--', linewidth=1)
        ax_scatter.axvline(w_stats['max'], color='blue', linestyle='--', linewidth=1)
        ax_scatter.axhline(h_stats['min'], color='orange', linestyle='--', linewidth=1)
        ax_scatter.axhline(h_stats['max'], color='green', linestyle='--', linewidth=1)

        ax_scatter.text(w_stats['min'] + label_offset, h_stats['max'], f"{int(w_stats['min'])}",
                        color='red', fontsize=9, va='bottom', ha='left')
        ax_scatter.text(w_stats['max'] - label_offset, h_stats['max'], f"{int(w_stats['max'])}",
                        color='blue', fontsize=9, va='bottom', ha='right')
        ax_scatter.text(w_stats['max'], h_stats['min'] + label_offset, f"{int(h_stats['min'])}",
                        color='orange', fontsize=9, va='top', ha='right')
        ax_scatter.text(w_stats['max'], h_stats['max'] - label_offset, f"{int(h_stats['max'])}",
                        color='green', fontsize=9, va='bottom', ha='right')

        # 上部タイトル
        plt.suptitle('Scatter Plot of Image Sizes with Marginal Boxplots and Quartile Labels', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

class PositivePixelAnalyzer:
    def __init__(self, annotation_path, image_dir=None):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.data = None
        self.ratios = []

    def load_annotation(self):
        """Load annotation JSON file"""
        with open(self.annotation_path, 'r') as f:
            self.data = json.load(f)

    def compute_ratios(self):
        """Compute positive pixel ratios from polygon annotations"""
        if self.data is None:
            self.load_annotation()

        self.ratios = []  # Reset

        for image_info in tqdm(self.data['images'], desc="Processing images"):
            width = image_info['width']
            height = image_info['height']
            annotations = image_info.get('annotations', [])

            # Create empty mask
            mask = np.zeros((height, width), dtype=np.uint8)

            for ann in annotations:
                seg = ann.get('segmentation', [])
                if not seg:
                    continue
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

            total_pixels = height * width
            positive_pixels = np.sum(mask)
            ratio = positive_pixels / total_pixels
            self.ratios.append(ratio)

    def print_stats(self):
        """Print summary statistics of positive pixel ratios"""
        if not self.ratios:
            print("No ratios calculated. Run compute_ratios() first.")
            return

        ratios_np = np.array(self.ratios)
        print(f"Total images: {len(self.ratios)}")
        print(f"Average positive ratio: {ratios_np.mean():.4f}")
        print(f"Median: {np.median(ratios_np):.4f}")
        print(f"Max: {ratios_np.max():.4f}")
        print(f"Min: {ratios_np.min():.4f}")

    def plot_histogram(self, bins=30):
        """
        Plot histogram of positive pixel ratios with:
        - cumulative % on secondary axis
        - vertical lines at 50% and 80% thresholds
        - fine x-axis ticks
        - bar annotations (count and percentage)
        """
        if not self.ratios:
            print("No ratios calculated. Run compute_ratios() first.")
            return

        ratios_np = np.array(self.ratios)
        counts, bin_edges = np.histogram(ratios_np, bins=bins)
        total_images = len(ratios_np)

        # 累積パーセンテージ
        cumulative_counts = np.cumsum(counts)
        cumulative_percent = cumulative_counts / total_images * 100
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 閾値の位置を取得（50%, 80%）
        def find_threshold(percent_target):
            for i, cp in enumerate(cumulative_percent):
                if cp >= percent_target:
                    return bin_centers[i], cp
            return bin_centers[-1], cumulative_percent[-1]

        thresh_50_x, _ = find_threshold(50)
        thresh_80_x, _ = find_threshold(80)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # ヒストグラム（左Y軸）
        bar_width = bin_edges[1] - bin_edges[0]
        bars = ax1.bar(bin_centers, counts, width=bar_width, color='skyblue',
                    edgecolor='black', label='Histogram')
        ax1.set_xlabel("Positive Pixel Ratio", fontsize=12)
        ax1.set_ylabel("Number of Images", fontsize=12, color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # 横軸目盛りを細かく
        ax1.set_xticks(np.round(np.linspace(0, 1, 21), 2))  # 0.05間隔の目盛り

        # 第2軸：累積%
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, cumulative_percent, color='red', marker='o',
                linestyle='-', linewidth=2, label='Cumulative %')
        ax2.set_ylabel("Cumulative Percentage (%)", fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 棒の上に枚数と比率を表示
        for x, count in zip(bin_centers, counts):
            if count > 0:
                percent = count / total_images * 100
                ax1.text(x, count + max(counts) * 0.01,
                        f'{count}\n({percent:.1f}%)',
                        ha='center', va='bottom', fontsize=9)

        # 閾値ラインと注釈
        ax1.axvline(thresh_50_x, color='gray', linestyle='--', linewidth=1)
        ax1.axvline(thresh_80_x, color='gray', linestyle='--', linewidth=1)
        ax1.text(thresh_50_x, max(counts) * 0.9, '50% threshold',
                rotation=90, va='bottom', ha='right', color='gray')
        ax1.text(thresh_80_x, max(counts) * 0.9, '80% threshold',
                rotation=90, va='bottom', ha='right', color='gray')

        # タイトルと凡例
        fig.suptitle("Histogram & Cumulative % of Positive Pixel Ratios", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
        plt.show()
    
    def save_positive_overlay_images(self, output_dir, alpha=0.6, color=(0, 0, 255), positive_ratio=1.0):
        """
        Save overlay images where positive ratio >= given threshold.
        
        Args:
            output_dir (str): Directory to save overlaid images
            alpha (float): Transparency level for overlay (0~1)
            color (tuple): BGR color of mask overlay
            positive_ratio (float): Minimum positive pixel ratio threshold (e.g., 1.0 or 0.95)
        """
        if not self.ratios or not self.data:
            print("Run compute_ratios() first.")
            return

        os.makedirs(output_dir, exist_ok=True)
        count = 0

        for image_info, ratio in zip(self.data['images'], self.ratios):
            if ratio < positive_ratio:
                continue

            fname = image_info['file_name']
            img_path = os.path.join(self.image_dir, fname)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            height, width = image_info['height'], image_info['width']

            # Create binary mask
            mask = np.zeros((height, width), dtype=np.uint8)
            for ann in image_info.get('annotations', []):
                seg = ann.get('segmentation', [])
                if not seg:
                    continue
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

            # Color mask
            mask_rgb = np.zeros_like(image)
            mask_rgb[mask == 1] = color

            # Overlay
            overlay = cv2.addWeighted(mask_rgb, alpha, image, 1 - alpha, 0)

            # Save
            out_path = os.path.join(output_dir, f"overlay_{fname}")
            cv2.imwrite(out_path, overlay)
            count += 1

        print(f"{count} overlaid images saved to {output_dir}")