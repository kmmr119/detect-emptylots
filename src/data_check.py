import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import json
import cv2
from tqdm import tqdm

class BaseAnalyzer:
    """分析クラスの基底クラス"""
    def __init__(self, data_path, image_dir=None):
        self.data_path = data_path
        self.image_dir = image_dir
        self.data = None

    def load_data(self):
        """データの読み込み（サブクラスで実装）"""
        raise NotImplementedError

    def analyze(self):
        """データの分析（サブクラスで実装）"""
        raise NotImplementedError

    def print_summary(self):
        """分析結果の表示（サブクラスで実装）"""
        raise NotImplementedError

class ImageSizeAnalyzer(BaseAnalyzer):
    def __init__(self, image_dir):
        super().__init__(image_dir, image_dir)
        self.widths = []
        self.heights = []
        self.size_counter = {}
    
    def load_data(self):
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
    
    def analyze(self):
        self.load_data()
    
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
        ax_box_x.xaxis.tick_top()
        ax_box_x.xaxis.set_label_position('top')

        for key, value in w_stats.items():
            ax_box_x.text(value, 0.05, f'{int(value)}', rotation=90,
                        va='bottom', ha='center', fontsize=9)

        # 左 boxplot（Height）
        sns.boxplot(y=heights, ax=ax_box_y, color='lightgreen')
        ax_box_y.set(ylabel='', xticks=[])
        ax_box_y.set_xticks([])
        for key, value in h_stats.items():
            ax_box_y.text(0.05, value, f'{int(value)}',
                        va='center', ha='left', fontsize=9)

        # 散布図
        sns.scatterplot(x=widths, y=heights, alpha=0.6, ax=ax_scatter)
        ax_scatter.set_xlabel('Width (pixels)', fontsize=11)
        ax_scatter.yaxis.tick_right()
        ax_scatter.yaxis.set_label_position('right')
        ax_scatter.set_ylabel('Height (pixels)', fontsize=11)
        ax_scatter.grid(True)

        # 補助線とラベル
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

        plt.suptitle('Scatter Plot of Image Sizes with Marginal Boxplots and Quartile Labels', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

class PositivePixelAnalyzer(BaseAnalyzer):
    def __init__(self, annotation_path, image_dir=None):
        super().__init__(annotation_path, image_dir)
        self.ratios = []

    def load_data(self):
        """Load annotation JSON file"""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

    def analyze(self):
        """Compute positive pixel ratios from polygon annotations"""
        if self.data is None:
            self.load_data()

        self.ratios = []

        for image_info in tqdm(self.data['images'], desc="Processing images"):
            width = image_info['width']
            height = image_info['height']
            annotations = image_info.get('annotations', [])

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

    def print_summary(self):
        """Print summary statistics of positive pixel ratios"""
        if not self.ratios:
            print("No ratios calculated. Run analyze() first.")
            return

        ratios_np = np.array(self.ratios)
        print(f"Total images: {len(self.ratios)}")
        print(f"Average positive ratio: {ratios_np.mean():.4f}")
        print(f"Median: {np.median(ratios_np):.4f}")
        print(f"Max: {ratios_np.max():.4f}")
        print(f"Min: {ratios_np.min():.4f}")

    def plot_histogram(self, bins=30):
        """Plot histogram of positive pixel ratios with cumulative percentage"""
        if not self.ratios:
            print("No ratios calculated. Run analyze() first.")
            return

        ratios_np = np.array(self.ratios)
        counts, bin_edges = np.histogram(ratios_np, bins=bins)
        total_images = len(ratios_np)

        cumulative_counts = np.cumsum(counts)
        cumulative_percent = cumulative_counts / total_images * 100
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        def find_threshold(percent_target):
            for i, cp in enumerate(cumulative_percent):
                if cp >= percent_target:
                    return bin_centers[i], cp
            return bin_centers[-1], cumulative_percent[-1]

        thresh_50_x, _ = find_threshold(50)
        thresh_80_x, _ = find_threshold(80)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        bar_width = bin_edges[1] - bin_edges[0]
        bars = ax1.bar(bin_centers, counts, width=bar_width, color='skyblue',
                    edgecolor='black', label='Histogram')
        ax1.set_xlabel("Positive Pixel Ratio", fontsize=12)
        ax1.set_ylabel("Number of Images", fontsize=12, color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        ax1.set_xticks(np.round(np.linspace(0, 1, 21), 2))

        ax2 = ax1.twinx()
        ax2.plot(bin_centers, cumulative_percent, color='red', marker='o',
                linestyle='-', linewidth=2, label='Cumulative %')
        ax2.set_ylabel("Cumulative Percentage (%)", fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Distribution of Positive Pixel Ratios", fontsize=14)
        plt.tight_layout()
        plt.show()

    def save_positive_overlay_images(self, output_dir, alpha=0.6, color=(0, 0, 255), positive_ratio=1.0):
        """
        指定されたpositive_ratio以上の画像のみをオーバーレイして保存する
        
        Args:
            output_dir (str): 出力ディレクトリのパス
            alpha (float): オーバーレイの透明度 (0-1)
            color (tuple): オーバーレイの色 (BGR)
            positive_ratio (float): 保存する画像の最小positive_ratio
        """
        if self.data is None:
            self.load_data()

        os.makedirs(output_dir, exist_ok=True)
        
        # positive_ratioでフィルタリング
        filtered_images = [
            img_info for img_info in self.data['images']
            if img_info.get('positive_ratio', 0.0) >= positive_ratio
        ]
        
        print(f"保存対象画像数: {len(filtered_images)} (positive_ratio >= {positive_ratio})")

        for image_info in tqdm(filtered_images, desc="オーバーレイ画像を保存中"):
            if self.image_dir is None:
                continue

            image_path = os.path.join(self.image_dir, image_info['file_name'])
            if not os.path.exists(image_path):
                continue

            img = cv2.imread(image_path)
            if img is None:
                continue

            height, width = img.shape[:2]
            overlay = np.zeros_like(img)
            mask = np.zeros((height, width), dtype=np.uint8)

            for ann in image_info.get('annotations', []):
                seg = ann.get('segmentation', [])
                if not seg:
                    continue
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

            overlay[mask > 0] = color
            result = cv2.addWeighted(img, 1, overlay, alpha, 0)
            output_path = os.path.join(output_dir, f"overlay_{image_info['file_name']}")
            cv2.imwrite(output_path, result)