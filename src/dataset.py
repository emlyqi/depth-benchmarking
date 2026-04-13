from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.stereo import disp_to_depth


def get_kitti_paths(split='training'):
    img2_dir = Path(KITTI_ROOT) / split / 'image_2'
    img3_dir = Path(KITTI_ROOT) / split / 'image_3'
    disp_dir = Path(KITTI_ROOT) / split / 'disp_occ_0'

    # get all left images, only _10 files (first frame of each scene)
    left_imgs = sorted([f for f in img2_dir.glob('*_10.png')])

    triplets = []
    for left_path in left_imgs:
        name = left_path.name  # e.g. 000110_10.png
        right_path = img3_dir / name
        disp_path = disp_dir / name

        # only include if all three exist
        if right_path.exists() and disp_path.exists():
            triplets.append((left_path, right_path, disp_path))

    return triplets


def read_calib(calib_path):
    with open(calib_path) as f:
        lines = f.readlines()

    calib = {}
    for line in lines:
        key, val = line.strip().split(':', 1)
        try:
            calib[key] = np.array([float(x) for x in val.strip().split()])
        except ValueError:
            continue  # skip non-numeric lines like calib_time

    P0 = calib['P_rect_00'].reshape(3, 4)
    P1 = calib['P_rect_01'].reshape(3, 4)

    f = P0[0, 0]
    B = -P1[0, 3] / P1[0, 0]

    return f, B


def read_calib_full(calib_path):
    with open(calib_path) as f:
        lines = f.readlines()
    calib = {}
    for line in lines:
        key, val = line.strip().split(':', 1)
        try:
            calib[key] = np.array([float(x) for x in val.strip().split()])
        except ValueError:
            continue
    P0 = calib['P_rect_00'].reshape(3, 4)
    f = P0[0, 0]
    cx = P0[0, 2]
    cy = P0[1, 2]
    B = -calib['P_rect_01'].reshape(3, 4)[0, 3] / f
    return f, B, cx, cy


class KITTIDepthDataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        left_path, _, disp_path = self.triplets[idx]

        # load image
        img = cv2.imread(str(left_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load GT depth
        disp_gt = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        disp_gt[disp_gt == 0] = np.nan
        depth_gt = disp_to_depth(disp_gt, f, B)
        depth_gt = np.nan_to_num(depth_gt, nan=0.0)  # replace nan with 0
        depth_gt = cv2.resize(depth_gt, (384, 384), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            img = self.transform(img)

        depth_gt = torch.tensor(depth_gt, dtype=torch.float32)
        return img, depth_gt
