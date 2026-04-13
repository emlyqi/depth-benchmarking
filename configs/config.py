import os

INPUT_PATH = os.environ.get('KITTI_DATA_DIR', '/kaggle/input/datasets')
OUTPUT_PATH = os.environ.get('OUTPUT_DIR', '/kaggle/working')
SAVE_PATH = os.path.join(OUTPUT_PATH, 'dpt_large_finetuned.pth')

# training config
MODEL_NAME = "DPT_Large"
EPOCHS = 15
BATCH_SIZE = 2  # start with 2 for DPT_Large
LR = 1e-4
TRAIN_SIZE = 160
VAL_SIZE = 40
LOSS = "HuberLoss"
KITTI_MAX_DEPTH = 80.0
GRID_RES = 0.2
X_RANGE = (-25, 35)
Z_RANGE = (3, 80)
