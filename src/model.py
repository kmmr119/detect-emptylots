import torch
from torchvision import models

def get_deeplabv3plus(num_classes):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model