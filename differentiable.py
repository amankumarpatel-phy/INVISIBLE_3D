"""
Differentiable physics-based reconstruction for INVISIBLE³D.

Uses PyTorch autograd to optimize a 3D refractive-index distribution
by minimizing data fidelity + regularization losses through the
physics forward model.

Loss = L_data + λ_TV·L_TV + λ_sparse·L_sparse + λ_smooth·L_smooth

The object is represented as trainable parameters optimized via
gradient descent through a differentiable forward model.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DifferentiableParams:
    """Parameters for differentiable reconstruction."""
    # Optimization
    learning_rate: float = 0.01
    optimizer: str = 'adam'               # 'adam', 'sgd', 'lbfgs'
    max_iterations: int = 200
    
    # Forward model
    wavelength: float = 632.8e-9
    pixel_size: float = 0.5e-6
    slice_thickness: float = 0.5e-6
    n_background: float = 1.0
    z_detector: float = 100e-6
    propagation_method: str = 'angular_spectrum'
    
    # Regularization weights
    tv_weight: float = 0.01
    sparsity_weight: float = 0.0
    smoothness_weight: float = 0.0
    support_weight: float = 0.0
    
    # Constraints
    ri_min: float = 1.0
    ri_max: float = 2.0
    kappa_min: float = 0.0
    kappa_max: float = 0.1
    support_mask: Optional[np.ndarray] = None
    
    # Device
    device: str = 'cpu'                   # 'cpu' or 'cuda'
    
    seed: int = 42


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for differentiable reconstruction. "
                         "Install with: pip install torch")


class DifferentiableForwardModel:
    """Differentiable forward model using PyTorch.
    
    Implements angular spectrum propagation and multislice
    in a fully differentiable manner for gradient-based optimization.
    """

    def __init__(self, params: DifferentiableParams):
        _check_torch()
        self.params = params
        self.device = torch.device(params.device)

    def angular_spectrum_torch(self, field: torch.Tensor, z: float) -> torch.Tensor:
        """Differentiable angular spectrum propagation.
        
        Args:
            field: Complex tensor [Ny, Nx].
            z: Propagation distance [m].
        
        Returns:
            Propagated complex field.
        """
        p = self.params
        Ny, Nx = field.shape
        
        fx = torch.fft.fftfreq(Nx, d=p.pixel_size, device=self.device)
        fy = torch.fft.fftfreq(Ny, d=p.pixel_size, device=self.device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        
        f_max = p.n_background / p.wavelength
        f_sq = FX**2 + FY**2
        
        kz_sq = torch.clamp(f_max**2 - f_sq, min=0)
        kz = 2 * np.pi * torch.sqrt(kz_sq)
        
        H = torch.exp(1j * z * kz)
        H = H * (f_sq < f_max**2).to(field.dtype)
        
        spectrum = torch.fft.fft2(field)
        return torch.fft.ifft2(spectrum * H)

    def multislice_torch(self, field: torch.Tensor,
                         object_ri: torch.Tensor) -> torch.Tensor:
        """Differentiable multislice propagation.
        
        Args:
            field: 2D input field [Ny, Nx].
            object_ri: 3D RI distribution [Nz, Ny, Nx] (complex).
        
        Returns:
            Exit field [Ny, Nx].
        """
        p = self.params
        k = 2 * np.pi / p.wavelength
        Nz = object_ri.shape[0]
        
        U = field.clone()
        for iz in range(Nz):
            delta_n = object_ri[iz] - p.n_background
            transmission = torch.exp(1j * k * delta_n * p.slice_thickness)
            U = U * transmission
            if iz < Nz - 1:
                U = self.angular_spectrum_torch(U, p.slice_thickness)
        
        return U

    def forward(self, object_ri: torch.Tensor,
                angles: Optional[List[float]] = None) -> List[torch.Tensor]:
        """Run full forward model to generate predicted intensities.
        
        Args:
            object_ri: 3D complex RI [Nz, Ny, Nx] as trainable tensor.
            angles: Illumination angles.
        
        Returns:
            List of predicted intensity patterns.
        """
        p = self.params
        if object_ri.ndim == 2:
            Ny, Nx = object_ri.shape
            is_3d = False
        else:
            Nz, Ny, Nx = object_ri.shape
            is_3d = True
        
        if angles is None:
            angles = [0.0]
        
        intensities = []
        for theta in angles:
            # Tilted plane wave
            k = 2 * np.pi * p.n_background / p.wavelength
            kx = k * np.sin(theta)
            x = (torch.arange(Nx, device=self.device, dtype=torch.float64) - Nx // 2) * p.pixel_size
            y = (torch.arange(Ny, device=self.device, dtype=torch.float64) - Ny // 2) * p.pixel_size
            XX, YY = torch.meshgrid(x, y, indexing='xy')
            inc = torch.exp(1j * kx * XX).to(object_ri.dtype)
            
            if is_3d:
                exit_field = self.multislice_torch(inc, object_ri)
            else:
                delta_n = object_ri - p.n_background
                transmission = torch.exp(1j * (2*np.pi/p.wavelength) * delta_n * p.pixel_size)
                exit_field = inc * transmission
            
            det_field = self.angular_spectrum_torch(exit_field, p.z_detector)
            I_pred = torch.abs(det_field)**2
            intensities.append(I_pred)
        
        return intensities


class DifferentiableReconstructor:
    """Gradient-based reconstruction via differentiable physics.
    
    Optimizes the object representation by backpropagating through
    the forward model.
    """

    def __init__(self, params: DifferentiableParams):
        _check_torch()
        self.params = params
        self.device = torch.device(params.device)
        self.forward_model = DifferentiableForwardModel(params)
        self.losses = []

    def _total_variation(self, volume: torch.Tensor) -> torch.Tensor:
        """Total variation regularization."""
        tv = 0.0
        for dim in range(volume.ndim):
            slices_a = [slice(None)] * volume.ndim
            slices_b = [slice(None)] * volume.ndim
            slices_a[dim] = slice(1, None)
            slices_b[dim] = slice(None, -1)
            diff = volume[tuple(slices_a)] - volume[tuple(slices_b)]
            tv = tv + torch.sum(torch.abs(diff))
        return tv

    def _sparsity(self, volume: torch.Tensor) -> torch.Tensor:
        """L1 sparsity on refractive-index contrast."""
        delta_n = volume - self.params.n_background
        return torch.sum(torch.abs(delta_n))

    def _smoothness(self, volume: torch.Tensor) -> torch.Tensor:
        """L2 smoothness (Tikhonov) regularization."""
        delta_n = volume - self.params.n_background
        return torch.sum(torch.abs(delta_n)**2)

    def _support_loss(self, volume: torch.Tensor,
                      support: torch.Tensor) -> torch.Tensor:
        """Penalize object values outside support."""
        outside = 1 - support
        delta_n = torch.abs(volume - self.params.n_background)
        return torch.sum(delta_n * outside)

    def reconstruct(self, measured_intensities: List[np.ndarray],
                    angles: List[float],
                    initial_guess: Optional[np.ndarray] = None,
                    shape: Optional[Tuple[int, ...]] = None,
                    progress_callback=None) -> Tuple[np.ndarray, List[float]]:
        """Run differentiable reconstruction.
        
        Args:
            measured_intensities: List of measured intensity patterns.
            angles: Illumination angles.
            initial_guess: Optional initial RI estimate.
            shape: Object shape (required if no initial_guess).
            progress_callback: Optional callback(iteration, loss).
        
        Returns:
            (reconstructed_ri, loss_history)
        """
        p = self.params
        torch.manual_seed(p.seed)
        
        # Convert measurements to tensors
        I_meas = [torch.tensor(I, dtype=torch.float64, device=self.device)
                  for I in measured_intensities]
        
        # Initialize object
        if initial_guess is not None:
            obj_real = torch.tensor(initial_guess.real, dtype=torch.float64,
                                     device=self.device, requires_grad=True)
            obj_imag = torch.tensor(initial_guess.imag, dtype=torch.float64,
                                     device=self.device, requires_grad=True)
        else:
            if shape is None:
                raise ValueError("Either initial_guess or shape must be provided")
            obj_real = torch.full(shape, p.n_background, dtype=torch.float64,
                                   device=self.device, requires_grad=True)
            obj_imag = torch.zeros(shape, dtype=torch.float64,
                                    device=self.device, requires_grad=True)
        
        # Support mask
        support_tensor = None
        if p.support_mask is not None:
            support_tensor = torch.tensor(p.support_mask, dtype=torch.float64,
                                           device=self.device)
        
        # Optimizer
        params_list = [obj_real, obj_imag]
        if p.optimizer == 'adam':
            optimizer = torch.optim.Adam(params_list, lr=p.learning_rate)
        elif p.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params_list, lr=p.learning_rate, momentum=0.9)
        elif p.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(params_list, lr=p.learning_rate,
                                           max_iter=5, line_search_fn='strong_wolfe')
        else:
            raise ValueError(f"Unknown optimizer: {p.optimizer}")
        
        self.losses = []
        
        for iteration in range(p.max_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Construct complex object
                obj = torch.complex(obj_real, obj_imag)
                
                # Forward model
                I_pred = self.forward_model.forward(obj, angles)
                
                # Data fidelity
                L_data = sum(torch.mean((Ip - Im)**2) for Ip, Im in zip(I_pred, I_meas))
                
                # Regularization
                L_reg = torch.tensor(0.0, dtype=torch.float64, device=self.device)
                
                if p.tv_weight > 0:
                    L_reg = L_reg + p.tv_weight * self._total_variation(obj_real)
                
                if p.sparsity_weight > 0:
                    L_reg = L_reg + p.sparsity_weight * self._sparsity(obj_real)
                
                if p.smoothness_weight > 0:
                    L_reg = L_reg + p.smoothness_weight * self._smoothness(obj_real)
                
                if p.support_weight > 0 and support_tensor is not None:
                    L_reg = L_reg + p.support_weight * self._support_loss(obj_real, support_tensor)
                
                loss = L_data + L_reg
                loss.backward()
                return loss
            
            if p.optimizer == 'lbfgs':
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            # Clamp to physical bounds
            with torch.no_grad():
                obj_real.clamp_(p.ri_min, p.ri_max)
                obj_imag.clamp_(p.kappa_min, p.kappa_max)
            
            loss_val = loss.item()
            self.losses.append(loss_val)
            
            if progress_callback:
                progress_callback(iteration, loss_val)
        
        # Final result
        with torch.no_grad():
            result = (obj_real.cpu().numpy() + 1j * obj_imag.cpu().numpy())
        
        return result, self.losses
