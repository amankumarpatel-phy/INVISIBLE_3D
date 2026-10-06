"""
Visualization module for INVISIBLE³D.

Publication-quality plots for diffraction patterns, reconstructions,
error maps, convergence curves, and 3D volume rendering.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from typing import Optional, List, Dict, Tuple, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io


# Publication style
STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def apply_style():
    plt.rcParams.update(STYLE)


def _add_colorbar(ax, im, label=''):
    """Add colorbar to axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    if label:
        cb.set_label(label)
    return cb


def _extent_from_params(shape, pixel_size, units='µm'):
    """Compute image extent for physical coordinates."""
    scale = {'m': 1, 'mm': 1e3, 'µm': 1e6, 'nm': 1e9}[units]
    Ny, Nx = shape[-2:]
    half_x = Nx // 2 * pixel_size * scale
    half_y = Ny // 2 * pixel_size * scale
    return [-half_x, half_x, -half_y, half_y]


# ============================================================================
# Diffraction patterns
# ============================================================================

def plot_diffraction(intensity: np.ndarray, pixel_size: float = 1e-6,
                     title: str = 'Diffraction Pattern',
                     log_scale: bool = False,
                     wavelength: float = 632.8e-9,
                     units: str = 'µm',
                     figsize: Tuple = (6, 5)) -> plt.Figure:
    """Plot 2D diffraction intensity pattern."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    extent = _extent_from_params(intensity.shape, pixel_size, units)
    
    if log_scale:
        norm = LogNorm(vmin=max(intensity[intensity > 0].min(), 1e-10),
                       vmax=intensity.max())
        im = ax.imshow(intensity, extent=extent, cmap='inferno', norm=norm)
        _add_colorbar(ax, im, 'log₁₀(Intensity)')
    else:
        im = ax.imshow(intensity, extent=extent, cmap='inferno')
        _add_colorbar(ax, im, 'Intensity [a.u.]')
    
    ax.set_xlabel(f'x [{units}]')
    ax.set_ylabel(f'y [{units}]')
    ax.set_title(f'{title}\nλ = {wavelength*1e9:.1f} nm')
    
    fig.tight_layout()
    return fig


def plot_diffraction_log(intensity: np.ndarray, **kwargs) -> plt.Figure:
    """Plot diffraction pattern in log scale."""
    return plot_diffraction(intensity, log_scale=True, **kwargs)


def plot_diffraction_comparison(intensities: List[np.ndarray],
                                 labels: List[str],
                                 pixel_size: float = 1e-6,
                                 title: str = 'Diffraction Comparison',
                                 figsize: Tuple = (14, 4)) -> plt.Figure:
    """Compare multiple diffraction patterns side by side."""
    apply_style()
    n = len(intensities)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    extent = _extent_from_params(intensities[0].shape, pixel_size)
    
    for ax, I, label in zip(axes, intensities, labels):
        im = ax.imshow(I, extent=extent, cmap='inferno')
        _add_colorbar(ax, im)
        ax.set_title(label)
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
# Reconstruction visualization
# ============================================================================

def plot_complex_field(field: np.ndarray, pixel_size: float = 1e-6,
                       title: str = 'Complex Field',
                       units: str = 'µm',
                       figsize: Tuple = (12, 5)) -> plt.Figure:
    """Plot amplitude and phase of a complex field."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    extent = _extent_from_params(field.shape, pixel_size, units)
    
    # Amplitude
    im1 = ax1.imshow(np.abs(field), extent=extent, cmap='gray')
    _add_colorbar(ax1, im1, 'Amplitude')
    ax1.set_title('Amplitude')
    ax1.set_xlabel(f'x [{units}]')
    ax1.set_ylabel(f'y [{units}]')
    
    # Phase
    im2 = ax2.imshow(np.angle(field), extent=extent, cmap='twilight_shifted',
                     vmin=-np.pi, vmax=np.pi)
    _add_colorbar(ax2, im2, 'Phase [rad]')
    ax2.set_title('Phase')
    ax2.set_xlabel(f'x [{units}]')
    ax2.set_ylabel(f'y [{units}]')
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_ri_distribution(ri: np.ndarray, pixel_size: float = 1e-6,
                          title: str = 'Refractive Index',
                          units: str = 'µm',
                          figsize: Tuple = (12, 5)) -> plt.Figure:
    """Plot real and imaginary parts of refractive-index distribution."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    extent = _extent_from_params(ri.shape, pixel_size, units)
    
    im1 = ax1.imshow(ri.real, extent=extent, cmap='viridis')
    _add_colorbar(ax1, im1, 'n (real RI)')
    ax1.set_title('Real Part n(x,y)')
    ax1.set_xlabel(f'x [{units}]')
    ax1.set_ylabel(f'y [{units}]')
    
    im2 = ax2.imshow(ri.imag, extent=extent, cmap='magma')
    _add_colorbar(ax2, im2, 'κ (extinction)')
    ax2.set_title('Imaginary Part κ(x,y)')
    ax2.set_xlabel(f'x [{units}]')
    ax2.set_ylabel(f'y [{units}]')
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_ri_slices_3d(volume: np.ndarray, pixel_size: float = 1e-6,
                       slice_thickness: float = 1e-6,
                       slice_indices: Optional[Dict[str, int]] = None,
                       title: str = '3D RI Slices',
                       figsize: Tuple = (15, 5)) -> plt.Figure:
    """Plot XY, XZ, YZ slices through 3D RI volume."""
    apply_style()
    Nz, Ny, Nx = volume.shape
    
    if slice_indices is None:
        slice_indices = {'z': Nz // 2, 'y': Ny // 2, 'x': Nx // 2}
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    vmin, vmax = volume.real.min(), volume.real.max()
    
    # XY slice
    xy_slice = volume[slice_indices.get('z', Nz//2)].real
    extent_xy = _extent_from_params((Ny, Nx), pixel_size)
    im1 = ax1.imshow(xy_slice, extent=extent_xy, cmap='viridis', vmin=vmin, vmax=vmax)
    _add_colorbar(ax1, im1, 'n')
    ax1.set_title(f'XY slice (z={slice_indices.get("z", Nz//2)})')
    ax1.set_xlabel('x [µm]')
    ax1.set_ylabel('y [µm]')
    
    # XZ slice
    xz_slice = volume[:, slice_indices.get('y', Ny//2), :].real
    extent_xz = [-(Nx//2)*pixel_size*1e6, (Nx//2)*pixel_size*1e6,
                 -(Nz//2)*slice_thickness*1e6, (Nz//2)*slice_thickness*1e6]
    im2 = ax2.imshow(xz_slice, extent=extent_xz, cmap='viridis', vmin=vmin, vmax=vmax,
                     aspect='auto')
    _add_colorbar(ax2, im2, 'n')
    ax2.set_title(f'XZ slice (y={slice_indices.get("y", Ny//2)})')
    ax2.set_xlabel('x [µm]')
    ax2.set_ylabel('z [µm]')
    
    # YZ slice
    yz_slice = volume[:, :, slice_indices.get('x', Nx//2)].real
    extent_yz = [-(Ny//2)*pixel_size*1e6, (Ny//2)*pixel_size*1e6,
                 -(Nz//2)*slice_thickness*1e6, (Nz//2)*slice_thickness*1e6]
    im3 = ax3.imshow(yz_slice, extent=extent_yz, cmap='viridis', vmin=vmin, vmax=vmax,
                     aspect='auto')
    _add_colorbar(ax3, im3, 'n')
    ax3.set_title(f'YZ slice (x={slice_indices.get("x", Nx//2)})')
    ax3.set_xlabel('y [µm]')
    ax3.set_ylabel('z [µm]')
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
# GT vs Reconstruction
# ============================================================================

def plot_gt_vs_reconstruction(gt: np.ndarray, recon: np.ndarray,
                               pixel_size: float = 1e-6,
                               title: str = 'Ground Truth vs Reconstruction',
                               metrics_dict: Optional[Dict] = None,
                               figsize: Tuple = (18, 5)) -> plt.Figure:
    """Side-by-side comparison: GT, Reconstruction, Error Map."""
    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    extent = _extent_from_params(gt.shape, pixel_size)
    
    # Use real part for RI, amplitude for fields
    if np.iscomplexobj(gt):
        gt_show = gt.real
        recon_show = recon.real
        label = 'n (RI)'
    else:
        gt_show = gt
        recon_show = recon
        label = 'Value'
    
    vmin = min(gt_show.min(), recon_show.min())
    vmax = max(gt_show.max(), recon_show.max())
    
    im1 = ax1.imshow(gt_show, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
    _add_colorbar(ax1, im1, label)
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('x [µm]')
    ax1.set_ylabel('y [µm]')
    
    im2 = ax2.imshow(recon_show, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
    _add_colorbar(ax2, im2, label)
    ax2.set_title('Reconstruction')
    ax2.set_xlabel('x [µm]')
    ax2.set_ylabel('y [µm]')
    
    error = np.abs(gt_show - recon_show)
    im3 = ax3.imshow(error, extent=extent, cmap='hot')
    _add_colorbar(ax3, im3, '|Error|')
    ax3.set_title('Error Map')
    ax3.set_xlabel('x [µm]')
    ax3.set_ylabel('y [µm]')
    
    # Add metrics text
    if metrics_dict:
        text = '\n'.join([f'{k}: {v:.4f}' for k, v in list(metrics_dict.items())[:5]])
        fig.text(0.99, 0.02, text, fontsize=8, ha='right', va='bottom',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
# Convergence
# ============================================================================

def plot_convergence(errors: Union[List[float], Dict[str, List[float]]],
                     title: str = 'Reconstruction Convergence',
                     ylabel: str = 'Error',
                     log_scale: bool = True,
                     figsize: Tuple = (8, 5)) -> plt.Figure:
    """Plot convergence curve(s)."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(errors, dict):
        for name, err in errors.items():
            ax.plot(err, label=name, linewidth=1.5)
        ax.legend()
    else:
        ax.plot(errors, 'b-', linewidth=1.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


# ============================================================================
# Fourier space
# ============================================================================

def plot_fourier_space(field: np.ndarray, pixel_size: float = 1e-6,
                       title: str = 'Fourier Space',
                       figsize: Tuple = (6, 5)) -> plt.Figure:
    """Plot Fourier-space magnitude (log scale)."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(field)))
    spectrum_log = np.log10(spectrum + 1e-10)
    
    Ny, Nx = field.shape
    fx_max = 0.5 / pixel_size / 1e6
    extent = [-fx_max, fx_max, -fx_max, fx_max]
    
    im = ax.imshow(spectrum_log, extent=extent, cmap='viridis')
    _add_colorbar(ax, im, 'log₁₀|F|')
    ax.set_xlabel('fx [µm⁻¹]')
    ax.set_ylabel('fy [µm⁻¹]')
    ax.set_title(title)
    
    fig.tight_layout()
    return fig


# ============================================================================
# Research experiment plots
# ============================================================================

def plot_metric_vs_parameter(param_values: List[float],
                              metric_values: Dict[str, List[float]],
                              param_name: str = 'Parameter',
                              param_unit: str = '',
                              title: str = 'Metric vs Parameter',
                              figsize: Tuple = (8, 5)) -> plt.Figure:
    """Plot reconstruction metric as function of experimental parameter."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric_name, values in metric_values.items():
        ax.plot(param_values, values, 'o-', label=metric_name, linewidth=1.5, markersize=5)
    
    xlabel = f'{param_name}' + (f' [{param_unit}]' if param_unit else '')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_algorithm_comparison(results: Dict[str, dict],
                               metric_name: str = 'PSNR',
                               figsize: Tuple = (8, 5)) -> plt.Figure:
    """Bar chart comparing algorithm performance."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(results.keys())
    values = [results[n]['metrics'].get(metric_name, 0) for n in names]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'Algorithm Comparison — {metric_name}')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return fig


# ============================================================================
# 3D Plotly volume
# ============================================================================

def create_3d_volume_plotly(volume: np.ndarray,
                            pixel_size: float = 1e-6,
                            slice_thickness: float = 1e-6,
                            title: str = '3D Volume',
                            opacity: float = 0.3,
                            surface_count: int = 10):
    """Create interactive 3D volume visualization using Plotly.
    
    Returns Plotly figure object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    Nz, Ny, Nx = volume.shape
    
    x = np.arange(Nx) * pixel_size * 1e6
    y = np.arange(Ny) * pixel_size * 1e6
    z = np.arange(Nz) * slice_thickness * 1e6
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    vol_data = np.abs(volume.real) if np.iscomplexobj(volume) else np.abs(volume)
    vol_transposed = np.transpose(vol_data, (2, 1, 0))  # Plotly expects [x, y, z]
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=vol_transposed.flatten(),
        opacity=opacity,
        surface_count=surface_count,
        colorscale='Viridis',
        colorbar_title='n (RI)',
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x [µm]',
            yaxis_title='y [µm]',
            zaxis_title='z [µm]',
        ),
        width=700,
        height=600,
    )
    
    return fig


# ============================================================================
# Utility: figure to bytes (for Streamlit)
# ============================================================================

def fig_to_bytes(fig: plt.Figure, format: str = 'png') -> bytes:
    """Convert matplotlib figure to bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    plt.close(fig)
    return arr
