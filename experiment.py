"""
Experiment configuration and reproducibility for INVISIBLE³D.

Every experiment stores all parameters needed for exact reproduction:
  - Random seed
  - Object parameters
  - Wavelength, angles, distances
  - Detector settings
  - Noise parameters
  - Reconstruction algorithm + settings
  - Software version
  - Timestamp
"""

import json
import os
import datetime
import hashlib
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for reproducibility."""
    # Identification
    name: str = 'experiment'
    description: str = ''
    timestamp: str = ''
    software_version: str = '1.0.0'
    
    # Random seed
    seed: int = 42
    
    # Object
    object_type: str = 'disc'
    object_params: Dict = field(default_factory=dict)
    grid_size: List[int] = field(default_factory=lambda: [128, 128])
    pixel_size: float = 0.5e-6
    slice_thickness: float = 0.5e-6
    background_ri: float = 1.0
    
    # Optics
    wavelength: float = 632.8e-9
    wavelengths: List[float] = field(default_factory=list)
    propagation_model: str = 'angular_spectrum'
    z_detector: float = 100e-6
    propagation_distances: List[float] = field(default_factory=list)
    
    # Multi-angle
    illumination_angles: List[float] = field(default_factory=list)
    num_angles: int = 1
    angle_range: List[float] = field(default_factory=lambda: [-60, 60])
    
    # Detector
    photon_count: float = 1e5
    gaussian_noise_sigma: float = 0.0
    background_intensity: float = 0.0
    saturation_level: float = float('inf')
    numerical_aperture: float = float('inf')
    missing_pixel_fraction: float = 0.0
    bit_depth: int = 16
    
    # Reconstruction
    reconstruction_algorithm: str = 'hio'
    max_iterations: int = 300
    beta: float = 0.8
    use_support: bool = True
    use_positivity: bool = False
    
    # Regularization
    tv_weight: float = 0.0
    sparsity_weight: float = 0.0
    smoothness_weight: float = 0.0
    
    # Results (filled after experiment)
    final_error: float = 0.0
    metrics: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Handle inf/nan for JSON
        for key, val in d.items():
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                d[key] = str(val)
        return d

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            d = json.load(f)
        # Convert inf strings back
        for key, val in d.items():
            if val == 'inf':
                d[key] = float('inf')
        return cls(**d)

    def hash(self) -> str:
        """Generate a hash of the configuration (for uniqueness check)."""
        d = self.to_dict()
        d.pop('timestamp', None)
        d.pop('metrics', None)
        d.pop('final_error', None)
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:12]


class ExperimentRunner:
    """Manages experiment execution and result tracking."""

    def __init__(self, output_dir: str = 'experiments'):
        self.output_dir = output_dir
        self.results = {}

    def create_experiment_dir(self, config: ExperimentConfig) -> str:
        """Create a unique directory for this experiment."""
        dirname = f"{config.name}_{config.hash()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = os.path.join(self.output_dir, dirname)
        os.makedirs(exp_dir, exist_ok=True)
        config.save(os.path.join(exp_dir, 'config.json'))
        return exp_dir

    def run_parametric_study(self, base_config: ExperimentConfig,
                              param_name: str,
                              param_values: List[Any],
                              run_fn) -> Dict[str, List]:
        """Run a parametric study varying one parameter.
        
        Args:
            base_config: Base configuration.
            param_name: Name of parameter to vary.
            param_values: List of values.
            run_fn: Function(config) → metrics_dict.
        
        Returns:
            Dict with param_values and collected metrics.
        """
        results = {
            'param_name': param_name,
            'param_values': param_values,
            'metrics': [],
        }
        
        for val in param_values:
            config = ExperimentConfig(**base_config.to_dict())
            setattr(config, param_name, val)
            config.timestamp = datetime.datetime.now().isoformat()
            
            metrics = run_fn(config)
            results['metrics'].append(metrics)
        
        return results

    def save_results(self, results: Dict, filepath: str):
        """Save experiment results to JSON."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)


# ============================================================================
# Preset experiment configurations
# ============================================================================

def preset_2d_phase_retrieval() -> ExperimentConfig:
    """Preset config for 2D CDI phase retrieval experiment."""
    return ExperimentConfig(
        name='2d_phase_retrieval',
        description='2D phase retrieval from oversampled diffraction pattern',
        object_type='disc',
        grid_size=[128, 128],
        pixel_size=0.5e-6,
        wavelength=632.8e-9,
        z_detector=1e-3,
        reconstruction_algorithm='hio',
        max_iterations=300,
        beta=0.8,
        use_support=True,
        photon_count=1e5,
    )


def preset_multi_distance() -> ExperimentConfig:
    """Preset config for multi-distance phase retrieval."""
    return ExperimentConfig(
        name='multi_distance',
        description='Multi-distance Fresnel diffraction phase retrieval',
        object_type='disc',
        grid_size=[128, 128],
        pixel_size=0.5e-6,
        wavelength=632.8e-9,
        propagation_distances=[50e-6, 100e-6, 200e-6, 500e-6],
        reconstruction_algorithm='er',
        max_iterations=200,
        photon_count=1e5,
    )


def preset_tomography() -> ExperimentConfig:
    """Preset config for 3D tomographic reconstruction."""
    return ExperimentConfig(
        name='3d_tomography',
        description='3D ODT reconstruction from multi-angle measurements',
        object_type='sphere',
        grid_size=[32, 64, 64],
        pixel_size=0.5e-6,
        slice_thickness=0.5e-6,
        wavelength=532e-9,
        num_angles=36,
        angle_range=[-60, 60],
        z_detector=100e-6,
        max_iterations=50,
        tv_weight=0.01,
    )


def preset_ptychography() -> ExperimentConfig:
    """Preset config for ptychographic reconstruction."""
    return ExperimentConfig(
        name='ptychography',
        description='Ptychographic CDI with overlapping probes',
        object_type='siemens_star',
        grid_size=[256, 256],
        pixel_size=0.5e-6,
        wavelength=632.8e-9,
        max_iterations=50,
        photon_count=1e6,
    )
