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

    def extract_amplitude(self, intensities: List[np.ndarray]) -> List[np.ndarray]:
        """Convert intensity measurements to measured amplitudes.

        This is an amplitude operation only.  Intensity-only data do not
        contain the detector-plane phase, so taking sqrt(I) must never be
        labeled as phase retrieval.
        """
        return [np.sqrt(np.maximum(I, 0.0)) for I in intensities]

    def extract_phase(self, intensities: List[np.ndarray],
                      angles: Optional[List[float]] = None) -> List[np.ndarray]:
        """Retrieve detector phase.

        Phase cannot be recovered from a single intensity measurement by
        sqrt(I).  Use a dedicated phase-retrieval method (for example
        multi-distance phase retrieval) before calling a complex-field
        tomography algorithm.
        """
        raise NotImplementedError(
            "Intensity-only measurements do not provide phase. "
            "Run a phase-retrieval algorithm first and pass complex "
            "scattered fields to the tomography reconstruction."
        )

    def reconstruct_fbp(self, scattered_fields: List[np.ndarray],
                        angles: Optional[List[float]] = None,
                        numerical_aperture: Optional[float] = None,
                        return_scattering_potential: bool = False) -> np.ndarray:
        """Reconstruct using the first-Born Fourier diffraction theorem.

        Each detector spatial-frequency sample is mapped to the 3-D
        scattering-vector location

            q = k_s - k_i,

        where |k_i| = |k_s| = k0*n_background.  For illumination tilted
        about the y axis,

            k_i = (k sin(theta), 0, k cos(theta)).

        The measured detector spectrum supplies transverse components of
        k_s; its longitudinal component is obtained from the Ewald sphere:

            k_sz = sqrt(k^2 - k_sx^2 - k_sy^2).

        Under the first-Born convention used here,

            U_s_hat(k_sx,k_sy)
              = i exp(i k_sz z_d) / (2 k_sz) * V_hat(q),

        with V = k0^2 [n^2 - n_background^2].

        The method performs weighted trilinear splatting of the measured
        V_hat samples onto a Cartesian 3-D Fourier grid and then applies
        a 3-D inverse FFT.

        IMPORTANT: The input must be the complex *scattered* detector field,
        not intensity and not the total field. Intensity-only data require a
        phase-retrieval stage first.

        Args:
            scattered_fields: Complex scattered detector fields [Ny, Nx].
            angles: Illumination angles in radians.
            numerical_aperture: Optional detection NA. Defaults to the
                propagating-wave limit n_background.
            return_scattering_potential: If True, return V(r) instead of n(r).

        Returns:
            Complex 3-D refractive-index distribution or scattering potential.
        """
        p = self.params
        if angles is None:
            angles = p.angles

        if len(scattered_fields) != len(angles):
            raise ValueError("Number of scattered fields must equal number of angles.")
        if len(scattered_fields) == 0:
            raise ValueError("At least one scattered field is required.")

        fields = [np.asarray(f, dtype=np.complex128) for f in scattered_fields]
        Ny, Nx = fields[0].shape
        if any(f.shape != (Ny, Nx) for f in fields):
            raise ValueError("All scattered fields must have the same shape.")

        k0 = wavenumber(p.wavelength)
        k = k0 * p.n_background
        detector_na = p.n_background if numerical_aperture is None else float(numerical_aperture)
        if detector_na <= 0 or detector_na > p.n_background:
            raise ValueError("numerical_aperture must satisfy 0 < NA <= n_background.")

        # Detector spatial frequencies -> transverse outgoing wavevector.
        fx = np.fft.fftfreq(Nx, d=p.pixel_size)
        fy = np.fft.fftfreq(Ny, d=p.pixel_size)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        KSX = 2.0 * np.pi * FX
        KSY = 2.0 * np.pi * FY
        ks_rho = np.sqrt(KSX**2 + KSY**2)

        propagating = ks_rho < (k * detector_na / p.n_background)
        KSZ = np.zeros_like(KSX)
        KSZ[propagating] = np.sqrt(
            np.maximum(k**2 - KSX[propagating]**2 - KSY[propagating]**2, 0.0)
        )

        # Cartesian object Fourier grid in shifted ordering.
        qx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=p.pixel_size))
        qy = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=p.pixel_size))
        qz = 2.0 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(p.Nz, d=p.slice_thickness)
        )
        dqx = 2.0 * np.pi / (Nx * p.pixel_size)
        dqy = 2.0 * np.pi / (Ny * p.pixel_size)
        dqz = 2.0 * np.pi / (p.Nz * p.slice_thickness)

        F_shift = np.zeros((p.Nz, Ny, Nx), dtype=np.complex128)
        W_shift = np.zeros((p.Nz, Ny, Nx), dtype=np.float64)

        # Fourier transform convention: numpy FFT is sum(exp(-2π i f x)).
        # Multiplying by the pixel volume converts it to the continuous
        # Fourier integral convention used by the Born Green function.
        detector_pixel_area = p.pixel_size ** 2

        for field, theta in zip(fields, angles):
            F_det = np.fft.fft2(field)

            # Incident wavevector for the x-z illumination geometry.
            kix = k * np.sin(theta)
            kiy = 0.0
            kiz = k * np.cos(theta)

            valid = propagating & (KSZ > 0)
            if not np.any(valid):
                continue

            # Remove the detector propagation phase and invert the Green
            # function factor to estimate V_hat at the Ewald sample.
            phase = np.exp(1j * KSZ * p.z_detector)
            V_hat_sample = (
                F_det * detector_pixel_area
                * (2.0 * KSZ / (1j * phase))
            )

            QX = KSX - kix
            QY = KSY - kiy
            QZ = KSZ - kiz

            # Splat valid samples into the eight neighboring Cartesian
            # Fourier voxels using trilinear weights.
            ix_f = (QX - qx[0]) / dqx
            iy_f = (QY - qy[0]) / dqy
            iz_f = (QZ - qz[0]) / dqz

            valid &= (
                (ix_f >= 0) & (ix_f < Nx - 1) &
                (iy_f >= 0) & (iy_f < Ny - 1) &
                (iz_f >= 0) & (iz_f < p.Nz - 1)
            )
            if not np.any(valid):
                continue

            ix0 = np.floor(ix_f[valid]).astype(np.int64)
            iy0 = np.floor(iy_f[valid]).astype(np.int64)
            iz0 = np.floor(iz_f[valid]).astype(np.int64)
            dx = ix_f[valid] - ix0
            dy = iy_f[valid] - iy0
            dz = iz_f[valid] - iz0
            values = V_hat_sample[valid]

            for oz, wz in ((0, 1.0 - dz), (1, dz)):
                for oy, wy in ((0, 1.0 - dy), (1, dy)):
                    for ox, wx in ((0, 1.0 - dx), (1, dx)):
                        weights = wx * wy * wz
                        np.add.at(
                            F_shift,
                            (iz0 + oz, iy0 + oy, ix0 + ox),
                            values * weights,
                        )
                        np.add.at(
                            W_shift,
                            (iz0 + oz, iy0 + oy, ix0 + ox),
                            weights,
                        )

        # Normalize only where the Ewald mapping actually supplied data.
        covered = W_shift > 0
        F_shift[covered] /= W_shift[covered]

        # Tikhonov damping based on sampling coverage.
        if p.tikhonov_weight > 0:
            F_shift *= W_shift / (W_shift + p.tikhonov_weight)

        # Return to FFT ordering and convert the continuous Fourier integral
        # back to a spatial distribution.  For a sampled transform pair,
        # V(r) = ifftn(F_hat) / voxel_volume.
        F_fft = np.fft.ifftshift(F_shift)
        voxel_volume = p.pixel_size ** 2 * p.slice_thickness
        scattering_potential = np.fft.ifftn(F_fft) / voxel_volume

        if return_scattering_potential:
            return scattering_potential

        # Exact first-Born conversion:
        # V = k0^2 (n^2 - n_bg^2)  =>  n = sqrt(n_bg^2 + V/k0^2).
        recon_ri = np.sqrt(
            p.n_background**2 + scattering_potential / (k0**2)
        )

        if p.positivity:
            recon_ri = np.maximum(recon_ri.real, p.ri_bounds[0]) + 1j * np.maximum(
                recon_ri.imag, 0.0
            )

        return recon_ri

    def reconstruct_gradient(self, intensities: List[np.ndarray],
                            angles: Optional[List[float]] = None,
                            progress_callback=None) -> np.ndarray:
        """Reconstruct RI by gradient descent through the multislice model.

        Unlike the previous residual backprojection, this computes the
        derivative of the nonlinear intensity loss through every multislice
        transmission and propagation step.  It is therefore an adjoint-style
        optimization method rather than a projection-space SIRT algorithm.
        """
        p = self.params
        if angles is None:
            angles = p.angles
        if len(intensities) != len(angles):
            raise ValueError("Number of intensity measurements must equal number of angles.")

        recon = np.full((p.Nz, p.Ny, p.Nx), p.n_background, dtype=np.complex128)
        self.errors = []
        step_size = 1e-4
        k0 = wavenumber(p.wavelength)

        for iteration in range(p.max_iterations):
            gradient_real = np.zeros(recon.shape, dtype=np.float64)
            gradient_imag = np.zeros(recon.shape, dtype=np.float64)
            total_error = 0.0

            for angle_idx, theta in enumerate(angles):
                inc = tilted_plane_wave(
                    p.Ny, p.Nx, p.pixel_size, p.wavelength,
                    theta_x=theta, theta_y=0, n_medium=p.n_background
                )

                # Store the field entering each slice and the transmission
                # factor so the detector loss can be differentiated exactly.
                U = inc.astype(np.complex128)
                incoming = []
                transmissions = []
                for iz in range(p.Nz):
                    incoming.append(U.copy())
                    delta_n = recon[iz] - p.n_background
                    T = np.exp(1j * k0 * delta_n * p.slice_thickness)
                    transmissions.append(T)
                    U = U * T
                    if iz < p.Nz - 1:
                        U = angular_spectrum(
                            U, p.wavelength, p.slice_thickness,
                            p.pixel_size, p.n_background
                        )

                det_field = angular_spectrum(
                    U, p.wavelength, p.z_detector,
                    p.pixel_size, p.n_background
                )
                I_pred = np.abs(det_field) ** 2
                residual = I_pred - intensities[angle_idx]
                total_error += float(np.mean(residual ** 2))

                # d(sum residual^2)/dU* in the real-valued convention.
                adj = 2.0 * residual * det_field
                adj = angular_spectrum(
                    adj, p.wavelength, -p.z_detector,
                    p.pixel_size, p.n_background
                )

                # Reverse-mode differentiation through the multislice chain.
                for iz in range(p.Nz - 1, -1, -1):
                    U_in = incoming[iz]
                    T = transmissions[iz]
                    U_after = U_in * T

                    dU_dn = 1j * k0 * p.slice_thickness * U_after
                    dU_dkappa = -k0 * p.slice_thickness * U_after

                    gradient_real[iz] += 2.0 * np.real(np.conj(adj) * dU_dn)
                    gradient_imag[iz] += 2.0 * np.real(np.conj(adj) * dU_dkappa)

                    # Adjoint of multiplication by T.
                    adj = adj * np.conj(T)

                    if iz > 0:
                        adj = angular_spectrum(
                            adj, p.wavelength, -p.slice_thickness,
                            p.pixel_size, p.n_background
                        )

            norm = max(len(angles), 1)
            gradient_real /= norm
            gradient_imag /= norm

            recon.real -= step_size * gradient_real
            recon.imag -= step_size * gradient_imag

            if p.tv_weight > 0:
                recon = self._tv_proximal(recon, p.tv_weight * step_size)

            if p.positivity:
                recon.real = np.clip(
                    recon.real, p.ri_bounds[0], p.ri_bounds[1]
                )
                recon.imag = np.maximum(recon.imag, 0.0)

            avg_error = total_error / norm
            self.errors.append(avg_error)

            if progress_callback:
                progress_callback(iteration, avg_error)

        return recon

    def reconstruct_sirt(self, intensities: List[np.ndarray],
                         angles: Optional[List[float]] = None,
                         progress_callback=None) -> np.ndarray:
        """Legacy API alias for the physics-consistent gradient reconstruction.

        The old implementation was a residual backprojection that did not
        differentiate the multislice forward model.  Calling it SIRT was
        misleading for diffraction data, so the implementation now delegates
        to the adjoint-style gradient solver.
        """
        return self.reconstruct_gradient(
            intensities, angles=angles,
            progress_callback=progress_callback
        )

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
