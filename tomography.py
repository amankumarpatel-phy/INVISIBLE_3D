"""
3D Tomographic reconstruction for INVISIBLE³D.

Reconstructs 3D refractive-index distributions n(x,y,z) from multi-angle
diffraction measurements using:
  - Filtered backprojection (Fourier diffraction theorem)
  - SIRT iterative reconstruction
  - Regularized gradient-based reconstruction

Supports limited-angle, full-angle, and missing-angle scenarios.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from .propagation import angular_spectrum, tilted_plane_wave, multislice_propagation
from .utils import wavenumber


@dataclass
class TomographyParams:
    """Parameters for tomographic reconstruction."""
    # Grid
    Nx: int = 64
    Ny: int = 64
    Nz: int = 32
    pixel_size: float = 0.5e-6
    slice_thickness: float = 0.5e-6
    
    # Optics
    wavelength: float = 632.8e-9
    n_background: float = 1.0
    z_detector: float = 100e-6
    
    # Angles
    angles: Optional[List[float]] = None  # Illumination angles [rad]
    num_angles: int = 36
    angle_range: Tuple[float, float] = (-np.pi/3, np.pi/3)
    
    # Reconstruction
    max_iterations: int = 50
    method: str = 'sirt'               # 'fbp', 'sirt', 'gradient'
    
    # Regularization
    tv_weight: float = 0.0             # Total variation
    tikhonov_weight: float = 0.01      # Tikhonov regularization
    positivity: bool = True            # Enforce Δn ≥ 0
    ri_bounds: Tuple[float, float] = (1.0, 2.0)  # RI bounds
    
    seed: int = 42

    def __post_init__(self):
        if self.angles is None:
            self.angles = np.linspace(
                self.angle_range[0], self.angle_range[1], self.num_angles
            ).tolist()


class TomographyEngine:
    """3D optical diffraction tomography reconstruction.
    
    Forward model:
      For each illumination angle θ:
        1. Generate tilted plane wave
        2. Propagate through 3D object via multislice
        3. Measure intensity at detector
    
    Inverse:
      Reconstruct n(x,y,z) from the collection of measurements.
    """

    def __init__(self, params: TomographyParams):
        self.params = params
        self.errors = []

    def generate_angles(self, mode: str = 'uniform') -> List[float]:
        """Generate illumination angles.
        
        Args:
            mode: 'uniform', 'limited', 'random', 'missing_cone'.
        
        Returns:
            List of angles in radians.
        """
        p = self.params
        
        if mode == 'uniform':
            angles = np.linspace(p.angle_range[0], p.angle_range[1], p.num_angles)
        elif mode == 'limited':
            # Limited angle range (e.g., ±30°)
            max_angle = np.pi / 6
            angles = np.linspace(-max_angle, max_angle, p.num_angles)
        elif mode == 'random':
            rng = np.random.default_rng(p.seed)
            angles = rng.uniform(p.angle_range[0], p.angle_range[1], p.num_angles)
            angles.sort()
        elif mode == 'missing_cone':
            # Full except a missing wedge
            all_angles = np.linspace(-np.pi/2, np.pi/2, p.num_angles * 2)
            # Remove angles near ±90°
            mask = np.abs(all_angles) < np.pi / 3
            angles = all_angles[mask][:p.num_angles]
        else:
            angles = np.array(p.angles)
        
        p.angles = angles.tolist()
        return angles.tolist()

    def forward_multi_angle(self, object_3d: np.ndarray,
                            angles: Optional[List[float]] = None
                            ) -> List[np.ndarray]:
        """Simulate multi-angle diffraction measurements.
        
        Args:
            object_3d: 3D complex RI distribution [Nz, Ny, Nx].
            angles: List of illumination angles (defaults to params.angles).
        
        Returns:
            List of 2D measured intensity patterns.
        """
        p = self.params
        if angles is None:
            angles = p.angles
        
        Nz, Ny, Nx = object_3d.shape
        intensities = []
        
        for theta in angles:
            # Tilted plane wave
            inc = tilted_plane_wave(Ny, Nx, p.pixel_size, p.wavelength,
                                    theta_x=theta, theta_y=0, n_medium=p.n_background)
            
            # Multislice propagation
            exit_field = multislice_propagation(
                inc, object_3d, p.wavelength, p.pixel_size,
                p.slice_thickness, p.n_background
            )
            
            # Propagate to detector
            det_field = angular_spectrum(exit_field, p.wavelength, p.z_detector,
                                         p.pixel_size, p.n_background)
            
            intensities.append(np.abs(det_field)**2)
        
        return intensities

    def extract_phase(self, intensities: List[np.ndarray],
                      angles: Optional[List[float]] = None) -> List[np.ndarray]:
        """Extract phase from intensity measurements using simple heuristic.
        
        For a more rigorous approach, use phase retrieval first.
        Here we assume access to the complex field (or use sqrt as amplitude).
        
        In practice, this would be replaced by actual phase retrieval.
        """
        phases = []
        for I in intensities:
            # Placeholder: in real experiments, phase must be retrieved
            # Here we just store the amplitude for projection
            phases.append(np.sqrt(np.maximum(I, 0)))
        return phases

    def reconstruct_fbp(self, scattered_fields: List[np.ndarray],
                        angles: Optional[List[float]] = None) -> np.ndarray:
        """Filtered back-projection using Fourier diffraction theorem.
        
        For each angle θ, the measured scattered field fills an Ewald sphere
        arc in 3D Fourier space. We accumulate these and inverse-FFT.
        
        Simplified 2D slice version (projection-based).
        
        Args:
            scattered_fields: Complex scattered fields at each angle.
            angles: Illumination angles.
        
        Returns:
            Reconstructed 3D RI distribution.
        """
        p = self.params
        if angles is None:
            angles = p.angles
        
        Ny, Nx = scattered_fields[0].shape
        
        # 3D Fourier space accumulator
        F3D = np.zeros((p.Nz, Ny, Nx), dtype=np.complex128)
        weight = np.zeros((p.Nz, Ny, Nx), dtype=np.float64)
        
        k = wavenumber(p.wavelength) * p.n_background
        
        # Frequency grids
        fx = np.fft.fftfreq(Nx, d=p.pixel_size)
        fy = np.fft.fftfreq(Ny, d=p.pixel_size)
        fz = np.fft.fftfreq(p.Nz, d=p.slice_thickness)
        
        FX, FY = np.meshgrid(fx, fy)
        
        for angle_idx, theta in enumerate(angles):
            # 2D Fourier transform of scattered field
            F2D = np.fft.fft2(scattered_fields[angle_idx])
            
            # Map to 3D Fourier space (Ewald sphere mapping)
            # For small angles, approximate: fz ≈ fx·sin(θ)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # Rotated frequency coordinates
            fx_rot = FX * cos_t
            fz_rot = FX * sin_t
            
            # Map to nearest fz slice
            for iz in range(p.Nz):
                fz_val = fz[iz]
                # Weight based on proximity to Ewald sphere
                w = np.exp(-0.5 * ((fz_rot - fz_val) / (fz[1] - fz[0] + 1e-30))**2)
                F3D[iz] += F2D * w
                weight[iz] += w
        
        # Normalize
        weight_safe = np.maximum(weight, 1e-10)
        F3D /= weight_safe
        
        # Regularization (Tikhonov)
        if p.tikhonov_weight > 0:
            F3D *= weight_safe / (weight_safe + p.tikhonov_weight)
        
        # Inverse 3D FFT
        recon = np.fft.ifftn(F3D)
        
        # Convert scattering potential to RI
        # V = k² (n² - n_bg²) ≈ 2k²·n_bg·Δn for weak scattering
        delta_n = recon.real / (2 * k**2 * p.n_background + 1e-30)
        recon_ri = p.n_background + delta_n
        
        # Apply constraints
        if p.positivity:
            recon_ri = np.maximum(recon_ri.real, p.ri_bounds[0]) + 1j * np.maximum(recon_ri.imag, 0)
        
        return recon_ri

    def reconstruct_sirt(self, intensities: List[np.ndarray],
                         angles: Optional[List[float]] = None,
                         progress_callback=None) -> np.ndarray:
        """SIRT-like iterative reconstruction.
        
        Iteratively refines the 3D object to match measured data.
        
        At each iteration:
          1. Forward simulate all angles
          2. Compare with measured intensities
          3. Backproject the residual
          4. Update the object
        
        Args:
            intensities: Measured intensity patterns.
            angles: Illumination angles.
            progress_callback: Optional callback(iteration, error).
        
        Returns:
            Reconstructed 3D RI distribution.
        """
        p = self.params
        if angles is None:
            angles = p.angles
        
        # Initialize with uniform background
        recon = np.full((p.Nz, p.Ny, p.Nx), p.n_background, dtype=np.complex128)
        
        self.errors = []
        step_size = 0.01  # Conservative step size
        
        for iteration in range(p.max_iterations):
            total_error = 0.0
            gradient = np.zeros_like(recon)
            
            for angle_idx, theta in enumerate(angles):
                # Forward: simulate measurement with current estimate
                inc = tilted_plane_wave(p.Ny, p.Nx, p.pixel_size, p.wavelength,
                                        theta_x=theta, theta_y=0, n_medium=p.n_background)
                
                exit_field = multislice_propagation(
                    inc, recon, p.wavelength, p.pixel_size,
                    p.slice_thickness, p.n_background
                )
                
                det_field = angular_spectrum(exit_field, p.wavelength, p.z_detector,
                                             p.pixel_size, p.n_background)
                
                I_pred = np.abs(det_field)**2
                I_meas = intensities[angle_idx]
                
                # Residual
                residual = I_pred - I_meas
                total_error += np.sum(residual**2)
                
                # Simple backprojection: distribute residual back through slices
                # This is an approximate gradient
                residual_field = 2 * det_field * residual
                
                # Back-propagate to object plane
                back_field = angular_spectrum(residual_field, p.wavelength, -p.z_detector,
                                              p.pixel_size, p.n_background)
                
                k = wavenumber(p.wavelength)
                for iz in range(p.Nz):
                    # Approximate gradient contribution
                    gradient[iz] += (back_field * np.conj(inc)).real * p.slice_thickness
            
            # Normalize gradient
            gradient /= len(angles)
            
            # Update
            recon -= step_size * gradient
            
            # Regularization: Total Variation
            if p.tv_weight > 0:
                recon = self._tv_proximal(recon, p.tv_weight * step_size)
            
            # Constraints
            if p.positivity:
                recon.real = np.clip(recon.real, p.ri_bounds[0], p.ri_bounds[1])
                recon.imag = np.maximum(recon.imag, 0)
            
            avg_error = total_error / len(angles)
            self.errors.append(avg_error)
            
            if progress_callback:
                progress_callback(iteration, avg_error)
        
        return recon

    def _tv_proximal(self, volume: np.ndarray, weight: float) -> np.ndarray:
        """Proximal operator for total variation regularization.
        
        Simple implementation using gradient thresholding.
        """
        result = volume.copy()
        for axis in range(volume.ndim):
            grad = np.diff(volume, axis=axis)
            # Soft threshold
            grad_shrunk = np.sign(grad) * np.maximum(np.abs(grad) - weight, 0)
            # Reconstruct
            pad_shape = list(volume.shape)
            pad_shape[axis] = 1
            grad_shrunk = np.concatenate([grad_shrunk, np.zeros(pad_shape, dtype=volume.dtype)], axis=axis)
            result -= weight * grad_shrunk
        return result

    def analyze_angle_coverage(self, angles: List[float]) -> dict:
        """Analyze angular coverage and identify missing regions.
        
        Returns:
            Dict with coverage statistics.
        """
        angles_deg = np.degrees(angles)
        
        # Sort angles
        sorted_angles = np.sort(angles_deg)
        gaps = np.diff(sorted_angles)
        
        return {
            'num_angles': len(angles),
            'min_angle_deg': float(sorted_angles[0]),
            'max_angle_deg': float(sorted_angles[-1]),
            'range_deg': float(sorted_angles[-1] - sorted_angles[0]),
            'mean_spacing_deg': float(np.mean(gaps)) if len(gaps) > 0 else 0,
            'max_gap_deg': float(np.max(gaps)) if len(gaps) > 0 else 0,
            'is_limited_angle': float(sorted_angles[-1] - sorted_angles[0]) < 150,
        }
