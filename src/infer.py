import torch
import numpy as np

def inference(model, image, device='cpu'):
    model.eval()
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        image = image.to(device)
        output = model(image)['out']
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return pred 