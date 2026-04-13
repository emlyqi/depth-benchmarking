import cv2
import numpy as np


def compute_stereo(left_path, right_path):
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # StereoBM
    bm = cv2.StereoBM_create(numDisparities=128, blockSize=11)
    bm.setSpeckleWindowSize(80)
    bm.setSpeckleRange(32)
    bm.setUniquenessRatio(5)
    bm.setMinDisparity(0)
    disp_bm = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp_bm[disp_bm <= 0] = np.nan


    # StereoSGBM
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=11,
        P1=8*3*11**2,
        P2=32*3*11**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=80,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp_sgbm = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp_sgbm[disp_sgbm <= 0] = np.nan

    return disp_bm, disp_sgbm


def disp_to_depth(disp, f, B):
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = f * B / disp
        depth[depth <= 0] = np.nan
        depth[depth > 80] = np.nan # KITTI max range is 80m
    return depth
