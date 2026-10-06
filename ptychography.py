"""
Ptychographic reconstruction for INVISIBLE³D.

Implements ptychographic CDI using overlapping illumination positions.
Reconstructs both the object transmission function and (optionally) the probe.

Algorithm: extended Ptychographic Iterative Engine (ePIE).
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class PtychographyParams:
    """Parameters for ptychographic reconstruction."""
    # Grid
    object_shape: Tuple[int, int] = (256, 256)
    probe_shape: Tuple[int, int] = (64, 64)
    pixel_size: float = 0.5e-6       # [m]
    wavelength: float = 632.8e-9      # [m]
    z_detector: float = 50e-3         # Object-to-detector distance [m]
    
    # Scanning
    step_size: float = 10e-6          # Probe step size [m]
    overlap: float = 0.6              # Overlap fraction (overrides step_size if set)
    scan_pattern: str = 'raster'      # 'raster', 'spiral', 'random'
    
    # Reconstruction
    max_iterations: int = 50
    alpha: float = 1.0                # Object update step size
    beta_probe: float = 1.0           # Probe update step size
    recover_probe: bool = True        # Also recover the probe
    
    # Regularization
    object_regularization: float = 1e-3
    probe_regularization: float = 1e-3
    
    seed: int = 42


class PtychographyEngine:
    """Ptychographic CDI reconstruction engine.
    
    The ePIE algorithm:
      For each probe position j:
        1. ψ_j = O(r - r_j) · P(r)           [exit wave]
        2. Ψ_j = FFT(ψ_j)                     [diffraction]
        3. Ψ'_j = √I_j · Ψ_j / |Ψ_j|         [modulus constraint]
        4. ψ'_j = IFFT(Ψ'_j)                  [updated exit wave]
        5. ΔO = α · conj(P) · (ψ'_j - ψ_j) / max(|P|²)  [object update]
        6. ΔP = β · conj(O) · (ψ'_j - ψ_j) / max(|O|²)  [probe update]
    """

    def __init__(self, params: PtychographyParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.positions = []
        self.errors = []

    def generate_scan_positions(self) -> np.ndarray:
        """Generate probe scan positions based on scanning pattern.
        
        Returns:
            Array of (row, col) pixel positions for probe placement.
        """
        p = self.params
        probe_h, probe_w = p.probe_shape
        obj_h, obj_w = p.object_shape
        
        # Step size in pixels
        if p.overlap > 0:
            step_px = int(probe_w * (1 - p.overlap))
        else:
            step_px = max(1, int(p.step_size / p.pixel_size))
        
        step_px = max(1, step_px)
        
        if p.scan_pattern == 'raster':
            positions = []
            for row in range(0, obj_h - probe_h + 1, step_px):
                for col in range(0, obj_w - probe_w + 1, step_px):
                    positions.append((row, col))
        
        elif p.scan_pattern == 'spiral':
            cx, cy = obj_w // 2 - probe_w // 2, obj_h // 2 - probe_h // 2
            positions = [(cy, cx)]
            n_rings = min(obj_h, obj_w) // (2 * step_px)
            for ring in range(1, n_rings + 1):
                n_points = max(6, int(2 * np.pi * ring * step_px / step_px))
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    r = ring * step_px
                    col = int(cx + r * np.cos(angle))
                    row = int(cy + r * np.sin(angle))
                    if 0 <= row <= obj_h - probe_h and 0 <= col <= obj_w - probe_w:
                        positions.append((row, col))
        
        elif p.scan_pattern == 'random':
            n_positions = max(10, (obj_h // step_px) * (obj_w // step_px))
            positions = []
            for _ in range(n_positions):
                row = self.rng.integers(0, obj_h - probe_h + 1)
                col = self.rng.integers(0, obj_w - probe_w + 1)
                positions.append((row, col))
        
        else:
            raise ValueError(f"Unknown scan pattern: {p.scan_pattern}")
        
        self.positions = np.array(positions)
        return self.positions

    def generate_probe(self, probe_type: str = 'gaussian',
                       sigma: float = 0.3) -> np.ndarray:
        """Generate illumination probe function.
        
        Args:
            probe_type: 'gaussian', 'uniform', 'airy'.
            sigma: Width parameter (fraction of probe window).
        
        Returns:
            2D complex probe field.
        """
        h, w = self.params.probe_shape
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        XX, YY = np.meshgrid(x, y)
        R = np.sqrt(XX**2 + YY**2)
        
        if probe_type == 'gaussian':
            probe = np.exp(-R**2 / (2 * sigma**2))
        elif probe_type == 'uniform':
            probe = (R <= sigma).astype(np.float64)
        elif probe_type == 'airy':
            # Approximate Airy pattern
            r_arg = np.maximum(R * 10, 1e-10)
            from scipy.special import j1
            probe = np.abs(2 * j1(r_arg) / r_arg)**2
            probe /= probe.max()
        else:
            probe = np.ones((h, w), dtype=np.float64)
        
        return probe.astype(np.complex128)

    def simulate_measurements(self, obj_transmission: np.ndarray,
                              probe: np.ndarray) -> List[np.ndarray]:
        """Simulate ptychographic diffraction measurements.
        
        Args:
            obj_transmission: 2D complex object transmission function.
            probe: 2D complex probe field.
        
        Returns:
            List of 2D intensity diffraction patterns.
        """
        if len(self.positions) == 0:
            self.generate_scan_positions()
        
        patterns = []
        ph, pw = self.params.probe_shape
        
        for row, col in self.positions:
            # Extract object patch
            obj_patch = obj_transmission[row:row+ph, col:col+pw]
            
            # Exit wave
            exit_wave = obj_patch * probe
            
            # Diffraction
            psi_f = np.fft.fft2(exit_wave)
            intensity = np.abs(psi_f)**2
            patterns.append(intensity)
        
        return patterns

    def reconstruct(self, diffraction_patterns: List[np.ndarray],
                    probe_init: Optional[np.ndarray] = None,
                    progress_callback=None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Run ePIE ptychographic reconstruction.
        
        Args:
            diffraction_patterns: List of measured diffraction intensities.
            probe_init: Initial probe estimate (optional).
            progress_callback: Optional callback(iteration, error) for GUI.
        
        Returns:
            (reconstructed_object, reconstructed_probe, error_history)
        """
        p = self.params
        oh, ow = p.object_shape
        ph, pw = p.probe_shape
        
        if len(self.positions) == 0:
            self.generate_scan_positions()
        
        # Initialize object (uniform transmission)
        obj = np.ones((oh, ow), dtype=np.complex128)
        
        # Initialize probe
        if probe_init is not None:
            probe = probe_init.copy()
        else:
            probe = self.generate_probe()
        
        self.errors = []
        
        for iteration in range(p.max_iterations):
            total_error = 0.0
            
            # Shuffle positions for better convergence
            indices = self.rng.permutation(len(self.positions))
            
            for idx in indices:
                row, col = self.positions[idx]
                measured_amp = np.sqrt(np.maximum(diffraction_patterns[idx], 0))
                
                # Extract object patch
                obj_patch = obj[row:row+ph, col:col+pw]
                
                # Exit wave
                psi = obj_patch * probe
                
                # Forward: to detector
                Psi = np.fft.fft2(psi)
                
                # Error
                err = np.sum((np.abs(Psi) - measured_amp)**2)
                total_error += err
                
                # Modulus constraint
                Psi_amp = np.abs(Psi)
                Psi_safe = np.where(Psi_amp > 1e-30, Psi_amp, 1e-30)
                Psi_prime = measured_amp * Psi / Psi_safe
                
                # Backward
                psi_prime = np.fft.ifft2(Psi_prime)
                
                # Difference
                delta = psi_prime - psi
                
                # Object update (ePIE)
                probe_conj = np.conj(probe)
                probe_max_sq = np.max(np.abs(probe)**2) + p.object_regularization
                obj[row:row+ph, col:col+pw] += p.alpha * probe_conj * delta / probe_max_sq
                
                # Probe update (ePIE)
                if p.recover_probe:
                    obj_patch_updated = obj[row:row+ph, col:col+pw]
                    obj_conj = np.conj(obj_patch_updated)
                    obj_max_sq = np.max(np.abs(obj_patch_updated)**2) + p.probe_regularization
                    probe += p.beta_probe * obj_conj * delta / obj_max_sq
            
            avg_error = total_error / len(self.positions)
            self.errors.append(avg_error)
            
            if progress_callback is not None:
                progress_callback(iteration, avg_error)
        
        return obj, probe, self.errors

    def get_scan_info(self) -> dict:
        """Get scanning geometry information."""
        if len(self.positions) == 0:
            self.generate_scan_positions()
        
        p = self.params
        ph, pw = p.probe_shape
        
        # Calculate actual overlap
        if len(self.positions) > 1:
            dists = np.diff(self.positions, axis=0)
            step_sizes = np.sqrt(dists[:, 0]**2 + dists[:, 1]**2)
            mean_step = np.mean(step_sizes)
            actual_overlap = 1 - mean_step / pw
        else:
            actual_overlap = 0
        
        return {
            'num_positions': len(self.positions),
            'probe_shape': p.probe_shape,
            'object_shape': p.object_shape,
            'actual_overlap': actual_overlap,
            'scan_pattern': p.scan_pattern,
            'positions': self.positions.tolist(),
        }
