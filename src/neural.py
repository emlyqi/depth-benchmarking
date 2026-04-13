import cv2
import numpy as np
import torch
from PIL import Image


def run_midas(image_path, model, transform, device):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()
    # INVERT - DPT outputs inverse depth (high = close, low = far)
    depth = 1.0 / (depth + 1e-8)
    return depth


def run_midas_finetuned(image_path, model, transform, device):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()
    # NO inversion - finetuned model outputs metric depth directly
    return depth


def run_depth_anything(image_path, pipeline):
    img = Image.open(str(image_path)).convert('RGB')
    result = pipeline(img)
    depth = np.array(result['predicted_depth'].squeeze()).astype(np.float32)
    depth = np.clip(depth, 0.1, None)  # clip to avoid div by zero and negatives
    depth = 1.0 / (depth + 1e-8)  # invert
    return depth


def median_scale_align(pred, gt):
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    scale = np.median(gt[mask]) / np.median(pred[mask])
    aligned = pred * scale
    aligned = np.clip(aligned, 0, 80)
    return aligned
