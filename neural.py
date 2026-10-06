"""
AI-assisted reconstruction for INVISIBLE³D.

Implements a neural network that predicts an initial reconstruction
from intensity measurements, followed by physics-based refinement.

Comparison modes:
  1. Pure iterative physics method
  2. Pure AI prediction
  3. AI initialization + Physics refinement
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class NeuralParams:
    """Parameters for neural reconstruction."""
    # Architecture
    num_channels: int = 32
    num_layers: int = 4
    
    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 8
    
    # Data
    num_training_samples: int = 200
    validation_fraction: float = 0.2
    
    # Device
    device: str = 'cpu'
    seed: int = 42
    background_ri: float = 1.0


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install: pip install torch")


class ReconstructionCNN(nn.Module):
    """Convolutional neural network for initial reconstruction estimate.
    
    Architecture: U-Net-like encoder-decoder with skip connections.
    Input: measured intensity (1 or N channels for multi-distance)
    Output: predicted Δn and extinction coefficient κ (2 channels)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 base_channels: int = 32, depth: int = 4):
        super().__init__()
        _check_torch()
        
        self.depth = depth
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(self._conv_block(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        
        # Bottleneck
        self.bottleneck = self._conv_block(ch, ch * 2)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2**i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(self._conv_block(out_ch * 2, out_ch))
            ch = out_ch
        
        # Output
        self.output_conv = nn.Conv2d(ch, out_channels, 1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        
        return self.output_conv(x)


class NeuralReconstructor:
    """AI-assisted reconstruction with physics refinement.
    
    Workflow:
      1. Train CNN on synthetic data
      2. Use CNN to predict initial estimate
      3. Refine with physics-based optimization
    """

    def __init__(self, params: NeuralParams):
        _check_torch()
        self.params = params
        self.device = torch.device(params.device)
        torch.manual_seed(params.seed)
        
        self.model = None
        self.train_losses = []
        self.val_losses = []

    def create_model(self, in_channels: int = 1, out_channels: int = 2):
        """Create the CNN model."""
        self.model = ReconstructionCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=self.params.num_channels,
            depth=self.params.num_layers
        ).to(self.device).double()

    def generate_training_data(self, object_generator_fn,
                               forward_fn,
                               num_samples: Optional[int] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data pairs (intensity → object).
        
        Args:
            object_generator_fn: Callable that returns a random object (2D complex).
            forward_fn: Callable that takes object and returns intensity.
            num_samples: Number of training pairs.
        
        Returns:
            (intensities_array, objects_array) with shape [N, 1, H, W] and [N, 2, H, W]
        """
        if num_samples is None:
            num_samples = self.params.num_training_samples
        
        intensities = []
        objects = []
        
        rng = np.random.default_rng(self.params.seed)
        
        for i in range(num_samples):
            obj = object_generator_fn(seed=self.params.seed + i)
            I = forward_fn(obj)
            
            # Normalize
            I_norm = I / (I.max() + 1e-30)
            
            # Learn physically meaningful RI parameters rather than the
            # magnitude/phase of the complex RI itself.
            # Channel 0 = refractive-index contrast Δn
            # Channel 1 = extinction coefficient κ
            delta_n = obj.real - self.params.background_ri
            kappa = obj.imag
            
            intensities.append(I_norm[np.newaxis, :, :])
            objects.append(np.stack([delta_n, kappa], axis=0))
        
        return np.array(intensities), np.array(objects)

    def train(self, train_intensities: np.ndarray,
              train_objects: np.ndarray,
              progress_callback=None) -> Dict[str, List[float]]:
        """Train the CNN model.
        
        Args:
            train_intensities: [N, C_in, H, W] intensity measurements.
            train_objects: [N, 2, H, W] object amplitude and phase.
            progress_callback: Optional callback(epoch, train_loss, val_loss).
        
        Returns:
            Training history with 'train_loss' and 'val_loss'.
        """
        p = self.params
        
        if self.model is None:
            self.create_model(in_channels=train_intensities.shape[1])
        
        # Split data
        N = len(train_intensities)
        n_val = max(1, int(N * p.validation_fraction))
        
        indices = np.random.default_rng(p.seed).permutation(N)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        X_train = torch.tensor(train_intensities[train_idx], dtype=torch.float64, device=self.device)
        Y_train = torch.tensor(train_objects[train_idx], dtype=torch.float64, device=self.device)
        X_val = torch.tensor(train_intensities[val_idx], dtype=torch.float64, device=self.device)
        Y_val = torch.tensor(train_objects[val_idx], dtype=torch.float64, device=self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=p.learning_rate)
        criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(p.num_epochs):
            self.model.train()
            
            # Mini-batch training
            perm = torch.randperm(len(X_train), device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, len(X_train), p.batch_size):
                end = min(start + p.batch_size, len(X_train))
                batch_idx = perm[start:end]
                
                x_batch = X_train[batch_idx]
                y_batch = Y_train[batch_idx]
                
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, Y_val).item()
            self.val_losses.append(val_loss)
            
            if progress_callback:
                progress_callback(epoch, avg_train_loss, val_loss)
        
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses}

    def predict(self, intensity: np.ndarray) -> np.ndarray:
        """Predict object from intensity measurement.
        
        Args:
            intensity: 2D intensity pattern.
        
        Returns:
            Complex object estimate.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Normalize
        I_norm = intensity / (intensity.max() + 1e-30)
        
        # To tensor [1, 1, H, W]
        x = torch.tensor(I_norm[np.newaxis, np.newaxis, :, :],
                         dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            pred = self.model(x)
        
        # Convert predicted physical parameters back to complex RI.
        pred_np = pred.cpu().numpy()[0]
        delta_n = pred_np[0]
        kappa = pred_np[1]
        
        return (self.params.background_ri + delta_n) + 1j * kappa

    def predict_and_refine(self, intensity: np.ndarray,
                           refine_fn,
                           measurements: Optional[List[np.ndarray]] = None,
                           **refine_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """AI prediction followed by physics refinement.
        
        Args:
            intensity: Measured intensity.
            refine_fn: Physics-based refinement function.
            **refine_kwargs: Arguments for refinement.
        
        Returns:
            (ai_prediction, refined_result, pure_physics_result)
        """
        # Multi-measurement physics solvers generally expect a list of
        # detector intensities.  Preserve an explicit escape hatch for callers
        # that have more than one measurement.
        if measurements is None:
            measurements = [intensity]
        
        ai_pred = self.predict(intensity)
        refined = refine_fn(measurements, initial_guess=ai_pred, **refine_kwargs)
        pure_physics = refine_fn(measurements, **refine_kwargs)
        
        return ai_pred, refined, pure_physics

    def compare_methods(self, intensity: np.ndarray,
                        ground_truth: np.ndarray,
                        refine_fn,
                        **refine_kwargs) -> Dict[str, dict]:
        """Compare AI, Physics, and AI+Physics reconstruction methods.
        
        Returns:
            Dict with results for each method.
        """
        from . import metrics as met
        
        ai_pred, refined, pure_physics = self.predict_and_refine(
            intensity, refine_fn, **refine_kwargs
        )
        
        results = {
            'AI_Only': {
                'reconstruction': ai_pred,
                'metrics': met.all_metrics(ground_truth, ai_pred),
            },
            'Physics_Only': {
                'reconstruction': pure_physics,
                'metrics': met.all_metrics(ground_truth, pure_physics),
            },
            'AI_plus_Physics': {
                'reconstruction': refined,
                'metrics': met.all_metrics(ground_truth, refined),
            },
        }
        
        return results
