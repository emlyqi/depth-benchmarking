import numpy as np


def compute_metrics(pred, gt):
    """
    pred: predicted depth map (H, W) in meters
    gt: ground truth depth map (H, W) in meters
    returns: dict of metrics
    """
    # only evaluate where GT is valid
    mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)

    if mask.sum() == 0:
        return None

    pred_masked = pred[mask]
    gt_masked = gt[mask]

    # MAE
    mae = np.mean(np.abs(pred_masked - gt_masked))

    # RMSE
    rmse = np.sqrt(np.mean((pred_masked - gt_masked) ** 2))

    # AbsRel
    absrel = np.mean(np.abs(pred_masked - gt_masked) / gt_masked)

    # threshold accuracy
    ratio = np.maximum(pred_masked / gt_masked, gt_masked / pred_masked)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        'mae': mae,
        'rmse': rmse,
        'absrel': absrel,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }


def average_metrics(metrics_list):
    """average a list of metric dicts"""
    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        return None

    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list]) for k in keys}


def print_results_table(results_dict):
    """
    results_dict: {'method_name': avg_metrics_dict}
    """
    print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'AbsRel':>8} {'δ<1.25':>8} {'δ<1.25²':>8} {'δ<1.25³':>8}")
    print("-" * 76)
    for method, metrics in results_dict.items():
        print(
            f"{method:<20} "
            f"{metrics['mae']:>8.3f} "
            f"{metrics['rmse']:>8.3f} "
            f"{metrics['absrel']:>8.3f} "
            f"{metrics['delta1']:>8.3f} "
            f"{metrics['delta2']:>8.3f} "
            f"{metrics['delta3']:>8.3f}"
        )
