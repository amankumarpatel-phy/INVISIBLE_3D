"""
Utility functions for INVISIBLE³D.
Coordinate grids, Fourier tools, and common numerical operations.
"""

import numpy as np
from typing import Tuple, Optional


def coordinate_grid_2d(N: int, pixel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create centered 2D spatial coordinate grids.
    
    Args:
        N: Grid size (assumes square NxN).
        pixel_size: Physical pixel size in meters.
    
    Returns:
        X, Y meshgrid arrays centered at origin.
    """
    coords = (np.arange(N) - N // 2) * pixel_size
    X, Y = np.meshgrid(coords, coords)
    return X, Y


def coordinate_grid_3d(Nz: int, Ny: int, Nx: int, pixel_size: float,
                        slice_thickness: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create centered 3D spatial coordinate grids.
    
    Args:
        Nz, Ny, Nx: Grid dimensions.
        pixel_size: Lateral pixel size in meters.
        slice_thickness: Axial step size. Defaults to pixel_size.
    
    Returns:
        X, Y, Z meshgrid arrays.
    """
    if slice_thickness is None:
        slice_thickness = pixel_size
    x = (np.arange(Nx) - Nx // 2) * pixel_size
    y = (np.arange(Ny) - Ny // 2) * pixel_size
    z = (np.arange(Nz) - Nz // 2) * slice_thickness
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    return X, Y, Z


def frequency_grid_2d(N: int, pixel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create 2D spatial frequency grids (fftfreq-based).
    
    Args:
        N: Grid size.
        pixel_size: Physical pixel size.
    
    Returns:
        FX, FY frequency meshgrids.
    """
    freq = np.fft.fftfreq(N, d=pixel_size)
    FX, FY = np.meshgrid(freq, freq)
    return FX, FY


def frequency_grid_2d_rect(Ny: int, Nx: int, pixel_size_x: float,
                            pixel_size_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create 2D frequency grids for rectangular arrays."""
    if pixel_size_y is None:
        pixel_size_y = pixel_size_x
    fx = np.fft.fftfreq(Nx, d=pixel_size_x)
    fy = np.fft.fftfreq(Ny, d=pixel_size_y)
    FX, FY = np.meshgrid(fx, fy)
    return FX, FY


def wavenumber(wavelength: float) -> float:
    """Compute wavenumber k = 2*pi/lambda."""
    return 2.0 * np.pi / wavelength


def add_poisson_noise(intensity: np.ndarray, photon_count: float = 1e4,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add Poisson shot noise to an intensity pattern.
    
    Args:
        intensity: Non-negative intensity array.
        photon_count: Mean total photon count scaling.
        rng: Random number generator for reproducibility.
    
    Returns:
        Noisy intensity with Poisson statistics.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Normalize and scale
    I_max = intensity.max()
    if I_max == 0:
        return intensity.copy()
    scaled = intensity / I_max * photon_count
    noisy = rng.poisson(lam=np.clip(scaled, 0, None)).astype(np.float64)
    return noisy * I_max / photon_count


def add_gaussian_noise(data: np.ndarray, sigma: float,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add Gaussian noise to data.
    
    Args:
        data: Input array.
        sigma: Standard deviation of noise (relative to data max).
        rng: Random number generator.
    
    Returns:
        Data with additive Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma * np.abs(data).max(), size=data.shape)
    if np.iscomplexobj(data):
        noise = noise + 1j * rng.normal(0, sigma * np.abs(data).max(), size=data.shape)
    return data + noise


def ensure_complex(arr: np.ndarray) -> np.ndarray:
    """Ensure array is complex-valued."""
    if not np.iscomplexobj(arr):
        return arr.astype(np.complex128)
    return arr


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def circular_aperture(N: int, radius_pixels: float) -> np.ndarray:
    """Create a circular binary aperture mask.
    
    Args:
        N: Grid size.
        radius_pixels: Radius of aperture in pixels.
    
    Returns:
        Binary mask (1 inside circle, 0 outside).
    """
    y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
    r2 = x**2 + y**2
    return (r2 <= radius_pixels**2).astype(np.float64)


def snr_db(signal: np.ndarray, noisy: np.ndarray) -> float:
    """Compute signal-to-noise ratio in dB."""
    noise = noisy - signal
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)
    if noise_power == 0:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


def zero_pad(arr: np.ndarray, factor: int = 2) -> np.ndarray:
    """Zero-pad a 2D array symmetrically.
    
    Args:
        arr: Input 2D array.
        factor: Padding factor (output is factor*N x factor*N).
    
    Returns:
        Zero-padded array.
    """
    Ny, Nx = arr.shape
    out_Ny, out_Nx = Ny * factor, Nx * factor
    out = np.zeros((out_Ny, out_Nx), dtype=arr.dtype)
    y0 = (out_Ny - Ny) // 2
    x0 = (out_Nx - Nx) // 2
    out[y0:y0+Ny, x0:x0+Nx] = arr
    return out


def crop_center(arr: np.ndarray, Ny: int, Nx: int) -> np.ndarray:
    """Crop the center of a 2D array."""
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[cy - Ny//2:cy - Ny//2 + Ny, cx - Nx//2:cx - Nx//2 + Nx]
