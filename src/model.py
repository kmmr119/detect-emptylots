import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from .utils import calculate_metrics

def create_deeplabv3(num_classes=2):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model

class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_deeplabv3(num_classes=num_classes)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)['out']
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks)
        
        # 評価指標の計算
        metrics = calculate_metrics(outputs, masks, self.num_classes)
        
        # ログ出力
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice', metrics['dice'], prog_bar=True)
        self.log('train_iou', metrics['iou'], prog_bar=True)
        self.log('train_precision', metrics['precision'])
        self.log('train_recall', metrics['recall'])
        self.log('train_pixel_accuracy', metrics['pixel_accuracy'])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks)
        
        # 評価指標の計算
        metrics = calculate_metrics(outputs, masks, self.num_classes)
        
        # ログ出力
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', metrics['dice'], prog_bar=True)
        self.log('val_iou', metrics['iou'], prog_bar=True)
        self.log('val_precision', metrics['precision'])
        self.log('val_recall', metrics['recall'])
        self.log('val_pixel_accuracy', metrics['pixel_accuracy'])
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]