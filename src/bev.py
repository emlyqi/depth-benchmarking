import numpy as np

from configs.config import GRID_RES, X_RANGE, Z_RANGE


def depth_to_bev(depth, f, cx, cy, grid_res=GRID_RES, x_range=X_RANGE, z_range=Z_RANGE):
    H, W = depth.shape

    # pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # valid pixels only
    valid = np.isfinite(depth) & (depth > 0)

    Z = depth[valid]
    X = (uu[valid] - cx) * Z / f
    Y = (vv[valid] - cy) * Z / f  # height

    # filter out ground plane and sky
    # in KITTI, camera is ~1.65m above ground
    # Y positive = below camera, Y negative = above camera
    height_mask = (Y > -5) & (Y < 0.5)  # keep points above ground level, removes road / sky
    Z = Z[height_mask]
    X = X[height_mask]

    # create grid
    x_bins = int((x_range[1] - x_range[0]) / grid_res)
    z_bins = int((z_range[1] - z_range[0]) / grid_res)
    grid = np.zeros((z_bins, x_bins))

    # bin points into grid
    xi = ((X - x_range[0]) / grid_res).astype(int)
    zi = ((Z - z_range[0]) / grid_res).astype(int)

    # keep only points within range
    mask = (xi >= 0) & (xi < x_bins) & (zi >= 0) & (zi < z_bins)
    xi, zi = xi[mask], zi[mask]

    # mark occupied cells
    grid[zi, xi] = 1

    return grid
