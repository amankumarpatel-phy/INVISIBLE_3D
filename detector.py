"""
Detector model for INVISIBLE³D.

Simulates realistic detector effects including noise, saturation,
finite resolution, missing pixels, and limited numerical aperture.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .utils import frequency_grid_2d_rect, circular_aperture


@dataclass
class DetectorParams:
    """Detector simulation parameters."""
    # Noise
    photon_count: float = 1e5          # Mean photon count for Poisson noise
    gaussian_sigma: float = 0.0        # Gaussian noise σ (relative to max)
    background: float = 0.0            # Additive background intensity
    
    # Detector effects
    saturation_level: float = np.inf   # Pixel saturation threshold
    bit_depth: int = 16                # ADC bit depth (0 = continuous)
    
    # Resolution
    psf_sigma: float = 0.0            # Detector PSF sigma [pixels]
    pixel_binning: int = 1             # Pixel binning factor
    
    # Aperture
    numerical_aperture: float = np.inf # NA limit (if finite, applies freq cutoff)
    
    # Defects
    missing_pixel_fraction: float = 0.0  # Fraction of dead pixels
    missing_pixel_seed: int = 42         # Seed for dead pixel positions
    
    # Systematic errors
    amplitude_scaling: float = 1.0     # Global amplitude scaling factor
    wavelength_error: float = 0.0      # Fractional wavelength uncertainty
    angular_error: float = 0.0         # Angular pointing error [rad]
    defocus: float = 0.0              # Defocus distance [m]
    
    # Random seed
    seed: int = 42


class Detector:
    """Simulates a realistic intensity detector with noise and artifacts.
    
    Pipeline: optical_field → |·|² → NA_filter → PSF_blur → Poisson →
              Gaussian → background → saturation → quantize → dead_pixels
    """

    def __init__(self, params: DetectorParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self._dead_pixel_mask = None

    def measure(self, field: np.ndarray, wavelength: float = 500e-9,
                pixel_size: float = 1e-6) -> np.ndarray:
        """Convert complex field to measured intensity with all detector effects.
        
        Args:
            field: 2D complex optical field at detector plane.
            wavelength: Wavelength [m] (for NA filtering).
            pixel_size: Pixel size [m].
        
        Returns:
            Measured intensity pattern (real, non-negative).
        """
        p = self.params
        U = field.copy()
        
        # 1. Apply defocus if specified
        if p.defocus != 0:
            from .propagation import angular_spectrum
            U = angular_spectrum(U, wavelength, p.defocus, pixel_size)
        
        # 2. Apply NA filtering in Fourier domain
        if np.isfinite(p.numerical_aperture):
            U = self._apply_na_filter(U, wavelength, pixel_size)
        
        # 3. Amplitude scaling
        U *= p.amplitude_scaling
        
        # 4. Intensity
        I = np.abs(U)**2
        
        # 5. PSF blur (finite pixel response)
        if p.psf_sigma > 0:
            I = self._apply_psf(I)
        
        # 6. Pixel binning
        if p.pixel_binning > 1:
            I = self._bin_pixels(I)
        
        # 7. Background
        I += p.background
        
        # 8. Poisson noise
        if p.photon_count < np.inf:
            I = self._add_poisson(I)
        
        # 9. Gaussian noise
        if p.gaussian_sigma > 0:
            I = self._add_gaussian(I)
        
        # 10. Saturation
        if np.isfinite(p.saturation_level):
            I = np.minimum(I, p.saturation_level)
        
        # 11. Quantization
        if p.bit_depth > 0 and p.bit_depth < 32:
            I = self._quantize(I)
        
        # 12. Dead pixels
        if p.missing_pixel_fraction > 0:
            I = self._apply_dead_pixels(I)
        
        # Ensure non-negative
        I = np.maximum(I, 0)
        
        return I

    def ideal_measure(self, field: np.ndarray) -> np.ndarray:
        """Ideal intensity measurement (no noise, no detector effects).
        
        Args:
            field: 2D complex field.
        
        Returns:
            Ideal intensity |U|².
        """
        return np.abs(field)**2

    def _apply_na_filter(self, field: np.ndarray, wavelength: float,
                          pixel_size: float) -> np.ndarray:
        """Apply numerical aperture frequency cutoff."""
        Ny, Nx = field.shape
        FX, FY = frequency_grid_2d_rect(Ny, Nx, pixel_size)
        f_cutoff = self.params.numerical_aperture / wavelength
        mask = (FX**2 + FY**2) <= f_cutoff**2
        spectrum = np.fft.fft2(field)
        return np.fft.ifft2(spectrum * mask)

    def _apply_psf(self, intensity: np.ndarray) -> np.ndarray:
        """Apply Gaussian point spread function blur."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(intensity, sigma=self.params.psf_sigma)

    def _bin_pixels(self, intensity: np.ndarray) -> np.ndarray:
        """Bin pixels by averaging blocks."""
        b = self.params.pixel_binning
        Ny, Nx = intensity.shape
        # Crop to divisible size
        Ny_crop = (Ny // b) * b
        Nx_crop = (Nx // b) * b
        I_crop = intensity[:Ny_crop, :Nx_crop]
        return I_crop.reshape(Ny_crop // b, b, Nx_crop // b, b).mean(axis=(1, 3))

    def _add_poisson(self, intensity: np.ndarray) -> np.ndarray:
        """Add Poisson shot noise."""
        I_max = intensity.max()
        if I_max == 0:
            return intensity
        scaled = intensity / I_max * self.params.photon_count
        noisy = self.rng.poisson(lam=np.clip(scaled, 0, 1e12)).astype(np.float64)
        return noisy * I_max / self.params.photon_count

    def _add_gaussian(self, intensity: np.ndarray) -> np.ndarray:
        """Add Gaussian readout noise."""
        sigma = self.params.gaussian_sigma * intensity.max()
        noise = self.rng.normal(0, sigma, size=intensity.shape)
        return intensity + noise

    def _quantize(self, intensity: np.ndarray) -> np.ndarray:
        """Quantize to bit depth."""
        levels = 2**self.params.bit_depth - 1
        I_max = intensity.max()
        if I_max == 0:
            return intensity
        normalized = intensity / I_max
        quantized = np.round(normalized * levels) / levels
        return quantized * I_max

    def _apply_dead_pixels(self, intensity: np.ndarray) -> np.ndarray:
        """Apply dead/missing pixels."""
        if self._dead_pixel_mask is None or self._dead_pixel_mask.shape != intensity.shape:
            dead_rng = np.random.default_rng(self.params.missing_pixel_seed)
            self._dead_pixel_mask = dead_rng.random(intensity.shape) > self.params.missing_pixel_fraction
        return intensity * self._dead_pixel_mask

    def get_snr(self, clean_intensity: np.ndarray,
                noisy_intensity: np.ndarray) -> float:
        """Compute SNR in dB between clean and noisy measurements."""
        signal_power = np.mean(clean_intensity**2)
        noise_power = np.mean((noisy_intensity - clean_intensity)**2)
        if noise_power == 0:
            return np.inf
        return 10 * np.log10(signal_power / noise_power)
