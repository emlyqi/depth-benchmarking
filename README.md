# Depth Estimation Benchmark: Stereo vs Neural Methods

> **Full technical writeup with methodology, parameter decisions, and analysis:** [WRITEUP.md](WRITEUP.md)

Benchmarking stereo and monocular depth estimation on KITTI Stereo 2015. Implements classical stereo (StereoBM, StereoSGBM), neural depth (DPT-Large, DepthAnything V2), fine-tuning DPT-Large on metric depth, ONNX export + INT8 quantization, and bird's eye view occupancy mapping.

## Results

| Method | MAE ↓ | RMSE ↓ | AbsRel ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|--------|-------|--------|----------|----------|-----------|-----------|
| StereoBM | 1.201 | 3.465 | 0.055 | 0.971 | 0.985 | 0.990 |
| StereoSGBM | 1.175 | 3.759 | 0.059 | 0.965 | 0.983 | 0.990 |
| DPT-Large (pretrained) | 2.541 | 5.475 | 0.127 | 0.861 | 0.962 | 0.986 |
| DepthAnything V2 Small | 2.154 | 5.414 | 0.105 | 0.912 | 0.976 | 0.989 |
| DPT-Large (fine-tuned) | 2.053 | 4.366 | 0.103 | 0.887 | 0.972 | 0.992 |

Evaluated on all 200 KITTI Stereo 2015 training scenes. Neural methods use median scale alignment. Fine-tuned model outputs metric depth directly (no alignment needed).

## Key Findings

- Stereo methods outperform neural methods on metric accuracy — geometric constraints give natural metric scale without alignment
- Fine-tuning DPT-Large on KITTI improved AbsRel by 19% (0.127 → 0.103) and beats pretrained DepthAnything V2
- INT8 quantization reduces model size 4x (1368MB → 352MB) but requires GPU for speed benefits — CPU inference is slower due to dequantization overhead
- Fine-tuning on sparse LiDAR GT degrades sky depth prediction — model receives no gradient signal for upper image regions

## Visualizations

<img src="assets/depth_comparison_5scenes.png" width="900">

*5-scene comparison across all methods*

<img src="assets/finetuned_comparison.png" width="700">

*Pretrained vs fine-tuned DPT-Large*

<img src="assets/l_sgbm_bev.png" width="700">

*Bird's eye view occupancy map from stereo depth*

## W&B Dashboard

[View training runs and evaluation metrics →](https://api.wandb.ai/links/emlyqi-team/01e1cfgd)

## Project Structure

```
depth-benchmarking/
├── README.md
├── requirements.txt
├── assets/                          # saved figures for README
├── configs/
│   └── config.py                    # all constants (MODEL_NAME, EPOCHS, BATCH_SIZE, etc.)
├── notebooks/
│   └── depth_benchmarking.ipynb     # full Kaggle notebook
└── src/
    ├── dataset.py                   # get_kitti_paths, read_calib, read_calib_full, KITTIDepthDataset
    ├── stereo.py                    # compute_stereo, disp_to_depth
    ├── neural.py                    # run_midas, run_midas_finetuned, run_depth_anything, median_scale_align
    ├── metrics.py                   # compute_metrics, average_metrics, print_results_table
    ├── train.py                     # training loop
    ├── evaluate.py                  # evaluate_all, evaluate_finetuned, visualize_comparison
    └── bev.py                       # depth_to_bev
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data and output paths

Configured via environment variables in `configs/config.py`, with Kaggle defaults:

```python
INPUT_PATH = os.environ.get('KITTI_DATA_DIR', '/kaggle/input/datasets')
OUTPUT_PATH = os.environ.get('OUTPUT_DIR', '/kaggle/working')
```

**On Kaggle:** these defaults work out of the box.

**Locally:**

```bash
export KITTI_DATA_DIR=./data
export OUTPUT_DIR=./output
```

### 3. KITTI dataset path

In `notebooks/depth_benchmarking.ipynb` (cell 3), edit `KITTI_ROOT` to match where your KITTI Stereo 2015 data lives:

```python
KITTI_ROOT = os.path.join(INPUT_PATH, 'kitti-stereo-2015', 'data', 'data_scene_flow')
```

On Kaggle you may need to prepend your dataset slug:

```python
KITTI_ROOT = os.path.join(INPUT_PATH, '<your-dataset-slug>', 'kitti-stereo-2015', 'data', 'data_scene_flow')
```

Expected directory structure:

```
data_scene_flow/
├── training/
│   ├── image_2/       # left images
│   ├── image_3/       # right images
│   └── disp_occ_0/    # ground truth disparity
└── data_scene_flow_calib/
    └── training/
        └── calib_cam_to_cam/
```

### 4. Weights & Biases

A [W&B](https://wandb.ai) account is needed for logging training runs and final results. The notebook defaults to `wandb.login()` (interactive prompt). Alternatives:

- **Environment variable:** `export WANDB_API_KEY=<your-key>`
- **Kaggle secrets:** store your key as `WANDB_API_KEY`

Set your W&B entity in **two** places (search for `<YOUR_WANDB_ENTITY>`):

1. The training cell in the notebook / `wandb_entity` param in `src/train.py`
2. The final results logging cell near the bottom of the notebook

### 5. Reproducing results

```bash
# run the full notebook
jupyter notebook notebooks/depth_benchmarking.ipynb
```

The notebook is the primary runnable artifact. The `src/` files are the same code organized into modules for readability.
