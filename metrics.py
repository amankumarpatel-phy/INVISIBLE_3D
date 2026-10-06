"""
Reconstruction metrics for INVISIBLE³D.

Quantitative comparison between reconstructed objects and ground truth.
All metrics operate on either complex fields, amplitudes, phases, or
refractive-index distributions.
"""

import numpy as np
from typing import Dict, Optional


def mse(ground_truth: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean Squared Error.
    
    MSE = (1/N) Σ |GT - Recon|²
    """
    return float(np.mean(np.abs(ground_truth - reconstruction)**2))


def rmse(ground_truth: np.ndarray, reconstruction: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(ground_truth, reconstruction)))


def nrmse(ground_truth: np.ndarray, reconstruction: np.ndarray) -> float:
    """Normalized Root Mean Squared Error.
    
    NRMSE = RMSE / (max(GT) - min(GT))
    """
    gt_range = np.abs(ground_truth).max() - np.abs(ground_truth).min()
    if gt_range == 0:
        return 0.0
    return rmse(ground_truth, reconstruction) / gt_range


def psnr(ground_truth: np.ndarray, reconstruction: np.ndarray,
         data_range: Optional[float] = None) -> float:
    """Peak Signal-to-Noise Ratio.
    
    PSNR = 10·log10(peak² / MSE)
    """
    if data_range is None:
        data_range = float(np.abs(ground_truth).max())
    mse_val = mse(ground_truth, reconstruction)
    if mse_val == 0:
        return np.inf
    return float(10 * np.log10(data_range**2 / mse_val))


def ssim(ground_truth: np.ndarray, reconstruction: np.ndarray,
         data_range: Optional[float] = None) -> float:
    """Structural Similarity Index.
    
    Uses a sliding window approach. For complex data, computes on magnitude.
    """
    # Use magnitude for complex data
    gt = np.abs(ground_truth).astype(np.float64)
    recon = np.abs(reconstruction).astype(np.float64)
    
    if data_range is None:
        data_range = gt.max() - gt.min()
    if data_range == 0:
        data_range = 1.0
    
    # Constants
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2
    
    # Global SSIM (simplified; for per-patch, use skimage)
    mu_x = gt.mean()
    mu_y = recon.mean()
    sigma_x_sq = np.var(gt)
    sigma_y_sq = np.var(recon)
    sigma_xy = np.mean((gt - mu_x) * (recon - mu_y))
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    return float(numerator / denominator)


def ssim_skimage(ground_truth: np.ndarray, reconstruction: np.ndarray,
                 data_range: Optional[float] = None) -> float:
    """SSIM using scikit-image for windowed computation (more accurate)."""
    try:
        from skimage.metrics import structural_similarity
        gt = np.abs(ground_truth).astype(np.float64)
        recon = np.abs(reconstruction).astype(np.float64)
        if data_range is None:
            data_range = gt.max() - gt.min()
            if data_range == 0:
                data_range = 1.0
        win_size = min(7, min(gt.shape))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        return float(structural_similarity(gt, recon, data_range=data_range,
                                            win_size=win_size))
    except ImportError:
        return ssim(ground_truth, reconstruction, data_range)


def voxelwise_relative_error(ground_truth: np.ndarray,
                              reconstruction: np.ndarray,
                              epsilon: float = 1e-10) -> np.ndarray:
    """Per-voxel relative error map.
    
    ε(r) = |GT(r) - Recon(r)| / (|GT(r)| + ε)
    
    Returns:
        Array of same shape with relative errors.
    """
    return np.abs(ground_truth - reconstruction) / (np.abs(ground_truth) + epsilon)


def refractive_index_error(gt_ri: np.ndarray, recon_ri: np.ndarray) -> Dict[str, float]:
    """Compute refractive-index-specific error metrics.
    
    Args:
        gt_ri: Ground truth complex RI distribution.
        recon_ri: Reconstructed complex RI distribution.
    
    Returns:
        Dict with real-part error, imaginary-part error, and magnitude error.
    """
    return {
        'real_mse': float(np.mean((gt_ri.real - recon_ri.real)**2)),
        'real_rmse': float(np.sqrt(np.mean((gt_ri.real - recon_ri.real)**2))),
        'imag_mse': float(np.mean((gt_ri.imag - recon_ri.imag)**2)),
        'imag_rmse': float(np.sqrt(np.mean((gt_ri.imag - recon_ri.imag)**2))),
        'magnitude_mse': float(np.mean((np.abs(gt_ri) - np.abs(recon_ri))**2)),
        'phase_mse': float(np.mean((np.angle(gt_ri) - np.angle(recon_ri))**2)),
    }


def support_iou(gt_support: np.ndarray, recon_support: np.ndarray) -> float:
    """Intersection over Union for binary support masks.
    
    IoU = |GT ∩ Recon| / |GT ∪ Recon|
    """
    gt_bool = gt_support.astype(bool)
    recon_bool = recon_support.astype(bool)
    intersection = np.sum(gt_bool & recon_bool)
    union = np.sum(gt_bool | recon_bool)
    if union == 0:
        return 1.0  # Both empty
    return float(intersection / union)


def support_overlap(gt_support: np.ndarray, recon_support: np.ndarray) -> float:
    """Support overlap ratio = |GT ∩ Recon| / |GT|."""
    gt_bool = gt_support.astype(bool)
    recon_bool = recon_support.astype(bool)
    gt_count = np.sum(gt_bool)
    if gt_count == 0:
        return 1.0
    return float(np.sum(gt_bool & recon_bool) / gt_count)


def all_metrics(ground_truth: np.ndarray, reconstruction: np.ndarray,
                support_gt: Optional[np.ndarray] = None,
                support_recon: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute all available metrics.
    
    Args:
        ground_truth: GT array (complex or real).
        reconstruction: Reconstructed array.
        support_gt: Optional GT support mask.
        support_recon: Optional reconstructed support mask.
    
    Returns:
        Dictionary with all metric values.
    """
    results = {
        'MSE': mse(ground_truth, reconstruction),
        'RMSE': rmse(ground_truth, reconstruction),
        'NRMSE': nrmse(ground_truth, reconstruction),
        'PSNR': psnr(ground_truth, reconstruction),
        'SSIM': ssim(ground_truth, reconstruction),
    }
    
    # Try skimage SSIM
    try:
        results['SSIM_windowed'] = ssim_skimage(ground_truth, reconstruction)
    except Exception:
        pass
    
    # RI-specific metrics for complex data
    if np.iscomplexobj(ground_truth):
        ri_metrics = refractive_index_error(ground_truth, reconstruction)
        results.update({f'RI_{k}': v for k, v in ri_metrics.items()})
    
    # Support metrics
    if support_gt is not None and support_recon is not None:
        results['IoU'] = support_iou(support_gt, support_recon)
        results['Support_Overlap'] = support_overlap(support_gt, support_recon)
    
    return results


def format_metrics(metrics: Dict[str, float], precision: int = 6) -> str:
    """Format metrics dictionary as a readable string."""
    lines = []
    for key, value in metrics.items():
        if np.isinf(value):
            lines.append(f"  {key}: ∞")
        else:
            lines.append(f"  {key}: {value:.{precision}f}")
    return "\n".join(lines)
