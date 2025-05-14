import torch
import numpy as np

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