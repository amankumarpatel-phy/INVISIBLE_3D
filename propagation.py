"""
Optical propagation module for INVISIBLE³D.

Implements forward wave-optics models for coherent light propagation:
  1. Angular Spectrum Method (ASM)
  2. Fresnel propagation (transfer function)
  3. Fraunhofer / far-field diffraction
  4. Multislice beam propagation method (BPM)
  5. Born approximation (first-order weak scattering)
  6. Rytov approximation

All functions operate on complex optical fields U(x,y) or U(x,y,z).
"""

import numpy as np
from typing import Optional, Tuple, List
from .utils import frequency_grid_2d, frequency_grid_2d_rect, wavenumber


# ============================================================================
# 1. Angular Spectrum Method
# ============================================================================

def angular_spectrum(field: np.ndarray, wavelength: float, z: float,
                     pixel_size: float, n_medium: float = 1.0,
                     bandlimit: bool = True) -> np.ndarray:
    """Propagate optical field using Angular Spectrum Method.
    
    U_out = IFFT2{ FFT2{U_in} · H_AS }
    
    H_AS(fx, fy) = exp(i·2π·z · sqrt(n²/λ² - fx² - fy²))
    
    Args:
        field: 2D complex input field U(x, y).
        wavelength: Wavelength in vacuum [m].
        z: Propagation distance [m]. Positive = forward.
        pixel_size: Spatial sampling [m].
        n_medium: Refractive index of propagation medium.
        bandlimit: If True, apply anti-aliasing bandlimit filter.
    
    Returns:
        Propagated complex field.
    """
    Ny, Nx = field.shape
    FX, FY = frequency_grid_2d_rect(Ny, Nx, pixel_size)
    
    # Spatial frequency cutoff
    f_max = n_medium / wavelength
    f_sq = FX**2 + FY**2
    
    # Propagating vs evanescent
    propagating = f_sq < f_max**2
    
    # Transfer function
    H = np.zeros((Ny, Nx), dtype=np.complex128)
    kz = 2 * np.pi * np.sqrt(np.maximum(f_max**2 - f_sq, 0))
    H[propagating] = np.exp(1j * z * kz[propagating])
    
    # Bandlimit filter to prevent aliasing artifacts
    if bandlimit:
        fx_limit = 1.0 / (np.sqrt((2 * z * (1.0 / (Nx * pixel_size)))**2 + 1) * wavelength / n_medium)
        fy_limit = 1.0 / (np.sqrt((2 * z * (1.0 / (Ny * pixel_size)))**2 + 1) * wavelength / n_medium)
        bl_mask = (np.abs(FX) < fx_limit) & (np.abs(FY) < fy_limit)
        H *= bl_mask
    
    # Propagate
    spectrum = np.fft.fft2(field)
    propagated = np.fft.ifft2(spectrum * H)
    
    return propagated


# ============================================================================
# 2. Fresnel Propagation (Transfer Function approach)
# ============================================================================

def fresnel_propagation(field: np.ndarray, wavelength: float, z: float,
                        pixel_size: float, n_medium: float = 1.0) -> np.ndarray:
    """Propagate optical field using Fresnel approximation (TF approach).
    
    H_F(fx, fy) = exp(ikz) · exp(-iπλz(fx² + fy²)/n)
    
    This is the paraxial approximation of ASM.
    
    Args:
        field: 2D complex input field.
        wavelength: Vacuum wavelength [m].
        z: Propagation distance [m].
        pixel_size: Pixel size [m].
        n_medium: Medium refractive index.
    
    Returns:
        Propagated complex field.
    """
    Ny, Nx = field.shape
    k = wavenumber(wavelength) * n_medium
    FX, FY = frequency_grid_2d_rect(Ny, Nx, pixel_size)
    
    # Fresnel transfer function
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2) / n_medium)
    
    spectrum = np.fft.fft2(field)
    propagated = np.fft.ifft2(spectrum * H)
    
    return propagated


def fresnel_single_fft(field: np.ndarray, wavelength: float, z: float,
                       pixel_size: float, n_medium: float = 1.0) -> Tuple[np.ndarray, float]:
    """Fresnel propagation using single-FFT (impulse response) approach.
    
    U(x,y,z) = (exp(ikz)/(iλz)) · FFT{ U(x',y',0) · exp(ik(x'²+y'²)/(2z)) }
    
    NOTE: This changes the output sampling to Δx_out = λz/(NΔx_in).
    
    Returns:
        (propagated_field, output_pixel_size)
    """
    Ny, Nx = field.shape
    k = wavenumber(wavelength) * n_medium
    lam = wavelength / n_medium
    
    # Input coordinates
    x = (np.arange(Nx) - Nx // 2) * pixel_size
    y = (np.arange(Ny) - Ny // 2) * pixel_size
    XX, YY = np.meshgrid(x, y)
    
    # Quadratic phase in input plane
    chirp = np.exp(1j * k / (2 * z) * (XX**2 + YY**2))
    
    # Prefactor
    prefactor = np.exp(1j * k * z) / (1j * lam * z)
    
    # Output pixel size
    dx_out = lam * np.abs(z) / (Nx * pixel_size)
    
    # Single FFT
    propagated = prefactor * np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field * chirp)))
    
    return propagated, dx_out


# ============================================================================
# 3. Fraunhofer / Far-field Diffraction
# ============================================================================

def fraunhofer(field: np.ndarray, wavelength: float, z: float,
               pixel_size: float, n_medium: float = 1.0) -> Tuple[np.ndarray, float]:
    """Far-field (Fraunhofer) diffraction pattern.
    
    U(x,y) ∝ FFT{ U_in(x',y') }  evaluated at fx = x/(λz), fy = y/(λz)
    
    Valid when z >> k·a²/2 where a is the aperture size.
    
    Returns:
        (diffracted_field, output_pixel_size)
    """
    Ny, Nx = field.shape
    k = wavenumber(wavelength) * n_medium
    lam = wavelength / n_medium
    
    # Output pixel size
    dx_out = lam * z / (Nx * pixel_size)
    
    # Prefactor
    prefactor = np.exp(1j * k * z) / (1j * lam * z)
    
    # Output coordinates for quadratic phase
    x_out = (np.arange(Nx) - Nx // 2) * dx_out
    y_out = (np.arange(Ny) - Ny // 2) * dx_out
    XX, YY = np.meshgrid(x_out, y_out)
    quad_phase = np.exp(1j * k / (2 * z) * (XX**2 + YY**2))
    
    # FFT
    far_field = prefactor * quad_phase * np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(field))
    ) * pixel_size**2
    
    return far_field, dx_out


# ============================================================================
# 4. Multislice Beam Propagation Method (BPM)
# ============================================================================

def multislice_propagation(field: np.ndarray, object_3d: np.ndarray,
                           wavelength: float, pixel_size: float,
                           slice_thickness: float,
                           n_background: float = 1.0,
                           propagation_method: str = 'angular_spectrum',
                           return_all_slices: bool = False) -> np.ndarray:
    """Propagate field through a 3D object using the multislice method.
    
    For each slice z_i:
      1. Apply transmission: U *= exp(i·k·Δn(x,y,z_i)·Δz)
      2. Free-space propagate by Δz
    
    Args:
        field: 2D input field (illumination).
        object_3d: 3D complex refractive-index array [Nz, Ny, Nx].
        wavelength: Vacuum wavelength [m].
        pixel_size: Lateral pixel size [m].
        slice_thickness: Axial slice thickness [m].
        n_background: Background medium refractive index.
        propagation_method: 'angular_spectrum' or 'fresnel'.
        return_all_slices: If True, return field at every slice.
    
    Returns:
        Exit field (2D), or list of fields at all slices if return_all_slices=True.
    """
    Nz = object_3d.shape[0]
    k = wavenumber(wavelength)
    
    # Select propagation function
    if propagation_method == 'angular_spectrum':
        prop_fn = lambda u: angular_spectrum(u, wavelength, slice_thickness, pixel_size, n_background, bandlimit=True)
    elif propagation_method == 'fresnel':
        prop_fn = lambda u: fresnel_propagation(u, wavelength, slice_thickness, pixel_size, n_background)
    else:
        raise ValueError(f"Unknown method: {propagation_method}")
    
    U = field.copy().astype(np.complex128)
    slices = [] if return_all_slices else None
    
    for iz in range(Nz):
        # Transmission through slice
        delta_n = object_3d[iz] - n_background
        transmission = np.exp(1j * k * delta_n * slice_thickness)
        U = U * transmission
        
        # Free-space propagation to next slice
        if iz < Nz - 1:
            U = prop_fn(U)
        
        if return_all_slices:
            slices.append(U.copy())
    
    if return_all_slices:
        return np.array(slices)
    return U


# ============================================================================
# 5. Born Approximation (First-order weak scattering)
# ============================================================================

def born_forward(incident_field: np.ndarray, scattering_potential: np.ndarray,
                 wavelength: float, pixel_size: float,
                 z_detector: float, n_medium: float = 1.0) -> np.ndarray:
    """Compute scattered field using first Born approximation (2D).
    
    U_s(r) = ∫ G(r-r') · V(r') · U_inc(r') dr'
    
    For 2D, computed as convolution with Green's function in Fourier domain:
      Û_s = Ĝ · F{V · U_inc}
    
    Args:
        incident_field: 2D incident field.
        scattering_potential: 2D scattering potential V = k²(n²-n_bg²).
        wavelength: Wavelength [m].
        pixel_size: Pixel size [m].
        z_detector: Detector distance [m].
        n_medium: Background refractive index.
    
    Returns:
        Total field at detector plane (U_inc_propagated + U_s).
    """
    k = wavenumber(wavelength) * n_medium
    
    # Source term: V·U_inc
    source = scattering_potential * incident_field
    
    # Propagate source term to detector
    U_scattered = angular_spectrum(source, wavelength, z_detector, pixel_size, n_medium)
    
    # Scale by pixel area (discrete approximation of integral)
    U_scattered *= pixel_size**2
    
    # Propagate incident field to detector
    U_inc_det = angular_spectrum(incident_field, wavelength, z_detector, pixel_size, n_medium)
    
    return U_inc_det + U_scattered


def born_3d_forward(incident_field: np.ndarray, scattering_potential_3d: np.ndarray,
                    wavelength: float, pixel_size: float,
                    slice_thickness: float, z_detector: float,
                    n_medium: float = 1.0) -> np.ndarray:
    """3D first Born approximation using slice-by-slice computation.
    
    Sum scattered contributions from each slice and propagate to detector.
    """
    Nz = scattering_potential_3d.shape[0]
    k = wavenumber(wavelength) * n_medium
    Ny, Nx = incident_field.shape
    
    U_total_scattered = np.zeros((Ny, Nx), dtype=np.complex128)
    
    for iz in range(Nz):
        z_slice = (iz - Nz // 2) * slice_thickness
        z_to_det = z_detector - z_slice
        
        if z_to_det <= 0:
            continue
            
        # Incident field at this slice (free-space propagation)
        U_inc_slice = angular_spectrum(incident_field, wavelength, z_slice, pixel_size, n_medium)
        
        # Source at this slice
        source = scattering_potential_3d[iz] * U_inc_slice
        
        # Propagate to detector
        U_s_contrib = angular_spectrum(source, wavelength, z_to_det, pixel_size, n_medium)
        U_total_scattered += U_s_contrib * pixel_size**2 * slice_thickness
    
    # Incident field at detector
    U_inc_det = angular_spectrum(incident_field, wavelength, z_detector, pixel_size, n_medium)
    
    return U_inc_det + U_total_scattered


# ============================================================================
# 6. Rytov Approximation
# ============================================================================

def rytov_forward(incident_field: np.ndarray, scattering_potential: np.ndarray,
                  wavelength: float, pixel_size: float,
                  z_detector: float, n_medium: float = 1.0) -> np.ndarray:
    """Compute field using Rytov approximation (2D).
    
    The Rytov complex phase: ψ_s ≈ U_s_Born / U_inc
    Total field: U_total = U_inc · exp(ψ_s)
    
    Args:
        incident_field: 2D incident field.
        scattering_potential: 2D scattering potential.
        wavelength: Wavelength [m].
        pixel_size: Pixel size [m].
        z_detector: Detector distance [m].
        n_medium: Background refractive index.
    
    Returns:
        Total field at detector using Rytov model.
    """
    # Compute Born scattered field
    source = scattering_potential * incident_field
    U_scattered = angular_spectrum(source, wavelength, z_detector, pixel_size, n_medium)
    U_scattered *= pixel_size**2
    
    # Incident field at detector
    U_inc_det = angular_spectrum(incident_field, wavelength, z_detector, pixel_size, n_medium)
    
    # Rytov complex phase
    # Avoid division by zero
    eps = 1e-30
    psi_s = U_scattered / (U_inc_det + eps)
    
    # Total field via Rytov
    U_total = U_inc_det * np.exp(psi_s)
    
    return U_total


# ============================================================================
# Multi-measurement acquisition
# ============================================================================

def multi_distance_measurement(field: np.ndarray, wavelength: float,
                               distances: List[float], pixel_size: float,
                               n_medium: float = 1.0,
                               method: str = 'angular_spectrum') -> List[np.ndarray]:
    """Acquire intensity measurements at multiple propagation distances.
    
    Args:
        field: 2D exit field from object.
        wavelength: Wavelength [m].
        distances: List of propagation distances z1, z2, ..., zN [m].
        pixel_size: Pixel size [m].
        n_medium: Medium refractive index.
        method: Propagation method.
    
    Returns:
        List of 2D intensity patterns I_k = |U(x,y,z_k)|².
    """
    prop_fn = angular_spectrum if method == 'angular_spectrum' else fresnel_propagation
    intensities = []
    for z in distances:
        U_z = prop_fn(field, wavelength, z, pixel_size, n_medium)
        intensities.append(np.abs(U_z)**2)
    return intensities


def tilted_plane_wave(Ny: int, Nx: int, pixel_size: float,
                      wavelength: float, theta_x: float = 0.0,
                      theta_y: float = 0.0, n_medium: float = 1.0) -> np.ndarray:
    """Generate a tilted plane wave for multi-angle illumination.
    
    U_inc = exp(i·(kx·x + ky·y))
    where kx = k·sin(theta_x), ky = k·sin(theta_y).
    
    Args:
        theta_x, theta_y: Tilt angles in radians.
    
    Returns:
        2D complex field.
    """
    k = wavenumber(wavelength) * n_medium
    kx = k * np.sin(theta_x)
    ky = k * np.sin(theta_y)
    
    x = (np.arange(Nx) - Nx // 2) * pixel_size
    y = (np.arange(Ny) - Ny // 2) * pixel_size
    XX, YY = np.meshgrid(x, y)
    
    return np.exp(1j * (kx * XX + ky * YY))


def multi_angle_measurement(object_2d_or_3d: np.ndarray,
                            wavelength: float, pixel_size: float,
                            angles: List[Tuple[float, float]],
                            z_detector: float,
                            slice_thickness: Optional[float] = None,
                            n_background: float = 1.0,
                            method: str = 'multislice') -> List[np.ndarray]:
    """Acquire intensity measurements under multiple illumination angles.
    
    Args:
        object_2d_or_3d: Object RI distribution (2D or 3D).
        wavelength: Wavelength [m].
        pixel_size: Pixel size [m].
        angles: List of (theta_x, theta_y) angle pairs in radians.
        z_detector: Object-to-detector distance [m].
        slice_thickness: For 3D objects.
        n_background: Background RI.
        method: 'multislice' or 'born'.
    
    Returns:
        List of 2D intensity patterns.
    """
    is_3d = object_2d_or_3d.ndim == 3
    if is_3d:
        Nz, Ny, Nx = object_2d_or_3d.shape
        if slice_thickness is None:
            slice_thickness = pixel_size
    else:
        Ny, Nx = object_2d_or_3d.shape
    
    intensities = []
    for theta_x, theta_y in angles:
        # Generate tilted illumination
        inc_field = tilted_plane_wave(Ny, Nx, pixel_size, wavelength,
                                      theta_x, theta_y, n_background)
        
        if is_3d and method == 'multislice':
            exit_field = multislice_propagation(
                inc_field, object_2d_or_3d, wavelength, pixel_size,
                slice_thickness, n_background
            )
            # Propagate to detector
            det_field = angular_spectrum(exit_field, wavelength, z_detector,
                                         pixel_size, n_background)
        elif is_3d and method == 'born':
            from .objects import ObjectGenerator, ObjectParams
            params = ObjectParams(grid_size=object_2d_or_3d.shape, pixel_size=pixel_size,
                                  wavelength=wavelength, background_ri=n_background)
            gen = ObjectGenerator(params)
            V = gen.ri_to_scattering_potential(object_2d_or_3d)
            det_field = born_3d_forward(inc_field, V, wavelength, pixel_size,
                                         slice_thickness, z_detector, n_background)
        else:
            # 2D: apply transmission and propagate
            k = wavenumber(wavelength)
            delta_n = object_2d_or_3d - n_background
            transmission = np.exp(1j * k * delta_n * pixel_size)
            exit_field = inc_field * transmission
            det_field = angular_spectrum(exit_field, wavelength, z_detector,
                                         pixel_size, n_background)
        
        intensities.append(np.abs(det_field)**2)
    
    return intensities


def multi_wavelength_measurement(object_ri: np.ndarray,
                                 wavelengths: List[float],
                                 pixel_size: float,
                                 z_detector: float,
                                 n_background: float = 1.0) -> List[np.ndarray]:
    """Acquire intensity measurements at multiple wavelengths.
    
    Note: The object's refractive index may be wavelength-dependent,
    but here we assume a fixed RI distribution for simplicity.
    
    Args:
        object_ri: 2D RI distribution.
        wavelengths: List of wavelengths [m].
        pixel_size: Pixel size [m].
        z_detector: Propagation distance [m].
        n_background: Background RI.
    
    Returns:
        List of intensity patterns at different wavelengths.
    """
    Ny, Nx = object_ri.shape[-2:]
    intensities = []
    for lam in wavelengths:
        inc = np.ones((Ny, Nx), dtype=np.complex128)
        k = wavenumber(lam)
        delta_n = object_ri - n_background
        if object_ri.ndim == 2:
            transmission = np.exp(1j * k * delta_n * pixel_size)
            exit_field = inc * transmission
        else:
            exit_field = multislice_propagation(
                inc, object_ri, lam, pixel_size, pixel_size, n_background
            )
        det_field = angular_spectrum(exit_field, lam, z_detector, pixel_size, n_background)
        intensities.append(np.abs(det_field)**2)
    return intensities
