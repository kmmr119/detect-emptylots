import json
import numpy as np

def mask_to_json(mask, output_path):
    # mask: (H, W) numpy配列
    result = {"segmentation": mask.tolist()}
    with open(output_path, "w") as f:
        json.dump(result, f)