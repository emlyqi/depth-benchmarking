import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from configs.config import OUTPUT_PATH, SAVE_PATH
from src.stereo import compute_stereo, disp_to_depth
from src.neural import run_midas, run_midas_finetuned, run_depth_anything, median_scale_align
from src.metrics import compute_metrics, average_metrics, print_results_table


def evaluate_all(triplets, f, B, midas_large, transform, depth_anything, device):
    bm_metrics = []
    sgbm_metrics = []
    dpt_metrics = []
    da_metrics = []

    for i, (left_path, right_path, disp_path) in enumerate(triplets):
        # load GT
        disp_gt = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        disp_gt[disp_gt == 0] = np.nan
        depth_gt = disp_to_depth(disp_gt, f, B)

        # SGBM
        disp_bm, disp_sgbm = compute_stereo(left_path, right_path)
        depth_bm = disp_to_depth(disp_bm, f, B)
        depth_sgbm = disp_to_depth(disp_sgbm, f, B)
        bm_metrics.append(compute_metrics(depth_bm, depth_gt))
        sgbm_metrics.append(compute_metrics(depth_sgbm, depth_gt))

        # DPT Large
        midas_raw = run_midas(left_path, midas_large, transform, device)
        midas_aligned = median_scale_align(midas_raw, depth_gt)
        dpt_metrics.append(compute_metrics(midas_aligned, depth_gt))

        # DepthAnything
        da_raw = run_depth_anything(left_path, depth_anything)
        da_aligned = median_scale_align(da_raw, depth_gt)
        da_metrics.append(compute_metrics(da_aligned, depth_gt))

        if i % 10 == 0:
            print(f"{i}/200 done")

    # print final table
    results = {
        'BM': average_metrics(bm_metrics),
        'SGBM': average_metrics(sgbm_metrics),
        'DPT_Large': average_metrics(dpt_metrics),
        'DepthAnything': average_metrics(da_metrics)
    }
    print_results_table(results)


def evaluate_finetuned(triplets, f, B, transform, device):
    # load fine-tuned model
    # uses SAVE_PATH from training above; change if loading weights from elsewhere
    finetuned_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    finetuned_model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    finetuned_model.to(device)
    finetuned_model.eval()

    # run evaluation
    finetuned_metrics = []

    for i, (left_path, right_path, disp_path) in enumerate(triplets):
        disp_gt = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        disp_gt[disp_gt == 0] = np.nan
        depth_gt = disp_to_depth(disp_gt, f, B)

        finetuned_depth = run_midas_finetuned(left_path, finetuned_model, transform, device)
        # don't need to scale / align bc finetuned model learned to output metric depth
        finetuned_metrics.append(compute_metrics(finetuned_depth, depth_gt))
        if i % 10 == 0:
            print(f"{i}/200 done")

    results_finetuned = {
        'DPT_Large_finetuned': average_metrics(finetuned_metrics)
    }
    print_results_table(results_finetuned)


def visualize_comparison(triplets, indices, f, B, midas, transform, depth_anything, device):
    fig, axes = plt.subplots(len(indices), 6, figsize=(20, 2*len(indices)))

    for row, i in enumerate(indices):
        left_path, right_path, disp_path = triplets[i]

        # load GT
        disp_gt = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        disp_gt[disp_gt == 0] = np.nan
        depth_gt = disp_to_depth(disp_gt, f, B)

        # stereo
        disp_bm, disp_sgbm = compute_stereo(left_path, right_path)
        depth_bm = disp_to_depth(disp_bm, f, B)
        depth_sgbm = disp_to_depth(disp_sgbm, f, B)

        # neural
        midas_raw = run_midas(left_path, midas, transform, device)
        midas_aligned = median_scale_align(midas_raw, depth_gt)

        da_raw = run_depth_anything(left_path, depth_anything)
        da_aligned = median_scale_align(da_raw, depth_gt)

        left_img = cv2.cvtColor(cv2.imread(str(left_path)), cv2.COLOR_BGR2RGB)

        axes[row][0].imshow(left_img)
        axes[row][0].set_title(f'Left Image {i}')
        axes[row][1].imshow(depth_gt, cmap='plasma', vmin=0, vmax=80)
        axes[row][1].set_title('GT Depth')
        axes[row][2].imshow(depth_bm, cmap='plasma', vmin=0, vmax=80)
        axes[row][2].set_title('StereoBM')
        axes[row][3].imshow(depth_sgbm, cmap='plasma', vmin=0, vmax=80)
        axes[row][3].set_title('StereoSGBM')
        axes[row][4].imshow(midas_aligned, cmap='plasma', vmin=0, vmax=80)
        axes[row][4].set_title('DPT Large')
        axes[row][5].imshow(da_aligned, cmap='plasma', vmin=0, vmax=80)
        axes[row][5].set_title('DepthAnything')

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'depth_comparison_5scenes.png'), dpi=150, bbox_inches='tight')
    plt.show()
