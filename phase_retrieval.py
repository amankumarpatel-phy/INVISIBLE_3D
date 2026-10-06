"""
Phase retrieval algorithms for INVISIBLE³D.

Implements iterative phase retrieval methods that recover complex fields
from intensity-only measurements.

Algorithms:
  - Gerchberg-Saxton (GS)
  - Error Reduction (ER)
  - Hybrid Input-Output (HIO)
  - Relaxed Averaged Alternating Reflections (RAAR)
  - Multi-distance phase retrieval
  - Transport of Intensity Equation (TIE) — linearized

All algorithms track convergence error per iteration.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class PhaseRetrievalParams:
    """Parameters for phase retrieval."""
    max_iterations: int = 300
    beta: float = 0.8                    # HIO/RAAR feedback parameter
    
    # Constraints
    support: Optional[np.ndarray] = None # Binary support mask
    use_support: bool = True
    use_positivity: bool = False         # Non-negativity on real part
    use_amplitude_constraint: bool = False
    known_amplitude: Optional[np.ndarray] = None  # Object-plane amplitude
    phase_min: Optional[float] = None    # Phase bounds
    phase_max: Optional[float] = None
    
    # Initialization
    init_method: str = 'random'          # 'random', 'zeros', 'uniform', 'custom'
    init_field: Optional[np.ndarray] = None  # For 'custom' init
    seed: int = 42
    
    # Convergence
    convergence_threshold: float = 1e-8  # Stop if error drops below
    stagnation_window: int = 50          # Stop if no improvement in N iters


class PhaseRetriever:
    """Iterative phase retrieval from intensity measurements.
    
    The core loop alternates between:
      1. Modulus constraint in measurement (Fourier/detector) domain
      2. Object-domain constraints (support, positivity, etc.)
    """

    def __init__(self, params: PhaseRetrievalParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.errors = []

    def _initialize(self, measured_amplitude: np.ndarray) -> np.ndarray:
        """Create initial guess for the object-domain field."""
        p = self.params
        shape = measured_amplitude.shape
        
        if p.init_method == 'custom' and p.init_field is not None:
            return p.init_field.copy()
        elif p.init_method == 'zeros':
            return np.ones(shape, dtype=np.complex128)
        elif p.init_method == 'uniform':
            return measured_amplitude.mean() * np.ones(shape, dtype=np.complex128)
        else:  # random
            phase = self.rng.uniform(-np.pi, np.pi, size=shape)
            amp = measured_amplitude.mean() * np.ones(shape)
            return amp * np.exp(1j * phase)

    def _modulus_constraint(self, field_fourier: np.ndarray,
                            measured_amplitude: np.ndarray) -> np.ndarray:
        """Apply modulus (Fourier magnitude) constraint.
        
        Replace amplitude with measured √I while keeping phase:
        G' = √I_measured · G / |G|
        """
        amp = np.abs(field_fourier)
        # Avoid division by zero
        amp_safe = np.where(amp > 1e-30, amp, 1e-30)
        return measured_amplitude * field_fourier / amp_safe

    def _support_constraint(self, field: np.ndarray) -> np.ndarray:
        """Apply support constraint: zero outside support."""
        if self.params.support is not None and self.params.use_support:
            return field * self.params.support
        return field

    def _object_constraints(self, field: np.ndarray) -> np.ndarray:
        """Apply all object-domain constraints."""
        p = self.params
        
        # Support
        result = self._support_constraint(field)
        
        # Positivity (real part >= 0)
        if p.use_positivity:
            result = np.where(result.real < 0,
                              1j * result.imag, result)
        
        # Known amplitude
        if p.use_amplitude_constraint and p.known_amplitude is not None:
            phase = np.angle(result)
            result = p.known_amplitude * np.exp(1j * phase)
        
        # Phase bounds
        if p.phase_min is not None or p.phase_max is not None:
            phase = np.angle(result)
            amp = np.abs(result)
            if p.phase_min is not None:
                phase = np.maximum(phase, p.phase_min)
            if p.phase_max is not None:
                phase = np.minimum(phase, p.phase_max)
            result = amp * np.exp(1j * phase)
        
        return result

    def _compute_error(self, field_fourier: np.ndarray,
                       measured_amplitude: np.ndarray) -> float:
        """Compute Fourier-domain error (normalized).
        
        E = Σ(|G| - √I)² / Σ I
        """
        diff = np.abs(field_fourier) - measured_amplitude
        return float(np.sum(diff**2) / np.sum(measured_amplitude**2 + 1e-30))

    # ======================== Algorithms ========================

    def gerchberg_saxton(self, measured_intensity: np.ndarray,
                          object_amplitude: Optional[np.ndarray] = None,
                          propagator_forward: Optional[Callable] = None,
                          propagator_backward: Optional[Callable] = None,
                          ) -> Tuple[np.ndarray, List[float]]:
        """Gerchberg-Saxton algorithm for two-plane phase retrieval.
        
        Standard GS alternates between object and measurement planes,
        enforcing known amplitudes in both planes.
        
        Args:
            measured_intensity: Measured intensity |U_det|² at detector.
            object_amplitude: Known amplitude at object plane (optional).
            propagator_forward: Forward propagation function (default: FFT).
            propagator_backward: Backward propagation function (default: IFFT).
        
        Returns:
            (recovered_field, error_history)
        """
        measured_amp = np.sqrt(np.maximum(measured_intensity, 0))
        p = self.params
        
        if propagator_forward is None:
            propagator_forward = np.fft.fft2
        if propagator_backward is None:
            propagator_backward = np.fft.ifft2
        
        # Initialize
        field = self._initialize(measured_amp)
        self.errors = []
        
        for iteration in range(p.max_iterations):
            # Forward propagate
            G = propagator_forward(field)
            
            # Error
            error = self._compute_error(G, measured_amp)
            self.errors.append(error)
            
            # Modulus constraint in Fourier domain
            G_prime = self._modulus_constraint(G, measured_amp)
            
            # Backward propagate
            field_prime = propagator_backward(G_prime)
            
            # Object-domain constraints
            if object_amplitude is not None:
                # GS: replace amplitude, keep phase
                phase = np.angle(field_prime)
                field = object_amplitude * np.exp(1j * phase)
            else:
                field = self._object_constraints(field_prime)
            
            # Convergence check
            if error < p.convergence_threshold:
                break
        
        return field, self.errors

    def error_reduction(self, measured_intensity: np.ndarray,
                         propagator_forward: Optional[Callable] = None,
                         propagator_backward: Optional[Callable] = None,
                         ) -> Tuple[np.ndarray, List[float]]:
        """Error Reduction algorithm (ER).
        
        Same as GS but uses support constraint in object domain
        instead of known amplitude.
        
        f_{n+1} = P_S(P_M(f_n))
        """
        measured_amp = np.sqrt(np.maximum(measured_intensity, 0))
        p = self.params
        
        if propagator_forward is None:
            propagator_forward = np.fft.fft2
        if propagator_backward is None:
            propagator_backward = np.fft.ifft2
        
        field = self._initialize(measured_amp)
        self.errors = []
        
        for iteration in range(p.max_iterations):
            G = propagator_forward(field)
            error = self._compute_error(G, measured_amp)
            self.errors.append(error)
            
            G_prime = self._modulus_constraint(G, measured_amp)
            field_prime = propagator_backward(G_prime)
            
            # ER: apply support constraint
            field = self._object_constraints(field_prime)
            
            if error < p.convergence_threshold:
                break
        
        return field, self.errors

    def hybrid_input_output(self, measured_intensity: np.ndarray,
                             propagator_forward: Optional[Callable] = None,
                             propagator_backward: Optional[Callable] = None,
                             ) -> Tuple[np.ndarray, List[float]]:
        """Hybrid Input-Output algorithm (HIO).
        
        f_{n+1}(x) = f'(x)              if x ∈ support
                    = f_n(x) - β·f'(x)   if x ∉ support
        
        where f' = IFFT(P_M(FFT(f_n)))
        """
        measured_amp = np.sqrt(np.maximum(measured_intensity, 0))
        p = self.params
        
        if propagator_forward is None:
            propagator_forward = np.fft.fft2
        if propagator_backward is None:
            propagator_backward = np.fft.ifft2
        
        field = self._initialize(measured_amp)
        self.errors = []
        
        support = p.support if p.support is not None else np.ones(measured_amp.shape, dtype=bool)
        
        for iteration in range(p.max_iterations):
            G = propagator_forward(field)
            error = self._compute_error(G, measured_amp)
            self.errors.append(error)
            
            G_prime = self._modulus_constraint(G, measured_amp)
            field_prime = propagator_backward(G_prime)
            
            # HIO update rule
            new_field = np.where(
                support,
                field_prime,
                field - p.beta * field_prime
            )
            
            # Additional constraints on supported region
            if p.use_positivity:
                violates = support & (new_field.real < 0)
                new_field[violates] = field[violates] - p.beta * field_prime[violates]
            
            field = new_field
            
            if error < p.convergence_threshold:
                break
        
        return field, self.errors

    def raar(self, measured_intensity: np.ndarray,
             propagator_forward: Optional[Callable] = None,
             propagator_backward: Optional[Callable] = None,
             ) -> Tuple[np.ndarray, List[float]]:
        """Relaxed Averaged Alternating Reflections (RAAR).
        
        f_{n+1} = (1/2)β(R_S·R_M + I)f_n + (1-β)P_M·f_n
        
        where R_S = 2P_S - I, R_M = 2P_M - I.
        """
        measured_amp = np.sqrt(np.maximum(measured_intensity, 0))
        p = self.params
        
        if propagator_forward is None:
            propagator_forward = np.fft.fft2
        if propagator_backward is None:
            propagator_backward = np.fft.ifft2
        
        field = self._initialize(measured_amp)
        self.errors = []
        
        support = p.support if p.support is not None else np.ones(measured_amp.shape, dtype=bool)
        
        for iteration in range(p.max_iterations):
            # P_M: modulus projection
            G = propagator_forward(field)
            error = self._compute_error(G, measured_amp)
            self.errors.append(error)
            
            G_m = self._modulus_constraint(G, measured_amp)
            pm_field = propagator_backward(G_m)  # P_M(f)
            
            # R_M = 2*P_M - I
            rm_field = 2 * pm_field - field
            
            # P_S(R_M(f))
            ps_rm = np.where(support, rm_field, 0)
            
            # R_S(R_M(f)) = 2*P_S(R_M(f)) - R_M(f)
            rs_rm = 2 * ps_rm - rm_field
            
            # RAAR update
            field = 0.5 * p.beta * (rs_rm + field) + (1 - p.beta) * pm_field
            
            if error < p.convergence_threshold:
                break
        
        return field, self.errors

    # ======================== Multi-distance Phase Retrieval ========================

    def multi_distance(self, intensities: List[np.ndarray],
                       propagators_forward: List[Callable],
                       propagators_backward: List[Callable],
                       algorithm: str = 'er'
                       ) -> Tuple[np.ndarray, List[float]]:
        """Multi-distance iterative phase retrieval.
        
        Uses measurements at multiple propagation distances to improve
        convergence and uniqueness.
        
        At each iteration, cycle through all distances applying the
        modulus constraint at each.
        
        Args:
            intensities: List of measured intensities at different distances.
            propagators_forward: Forward propagation functions to each distance.
            propagators_backward: Corresponding backward propagators.
            algorithm: 'er', 'hio', or 'raar'.
        
        Returns:
            (recovered_field, error_history)
        """
        p = self.params
        N_dist = len(intensities)
        measured_amps = [np.sqrt(np.maximum(I, 0)) for I in intensities]
        
        support = p.support if p.support is not None else np.ones(intensities[0].shape, dtype=bool)
        
        # Initialize
        field = self._initialize(measured_amps[0])
        self.errors = []
        
        for iteration in range(p.max_iterations):
            total_error = 0.0
            
            for dist_idx in range(N_dist):
                # Forward propagate to this distance
                G = propagators_forward[dist_idx](field)
                
                # Error
                error = self._compute_error(G, measured_amps[dist_idx])
                total_error += error
                
                # Modulus constraint
                G_prime = self._modulus_constraint(G, measured_amps[dist_idx])
                
                # Back propagate
                field_prime = propagators_backward[dist_idx](G_prime)
                
                # Object constraints
                if algorithm == 'hio':
                    field = np.where(support, field_prime,
                                     field - p.beta * field_prime)
                elif algorithm == 'raar':
                    pm_field = field_prime
                    rm_field = 2 * pm_field - field
                    ps_rm = np.where(support, rm_field, 0)
                    rs_rm = 2 * ps_rm - rm_field
                    field = 0.5 * p.beta * (rs_rm + field) + (1 - p.beta) * pm_field
                else:  # ER
                    field = self._object_constraints(field_prime)
            
            self.errors.append(total_error / N_dist)
            
            if self.errors[-1] < p.convergence_threshold:
                break
        
        return field, self.errors

    # ======================== Transport of Intensity ========================

    @staticmethod
    def tie_retrieve(intensities: List[np.ndarray],
                     distances: List[float],
                     wavelength: float,
                     pixel_size: float) -> np.ndarray:
        """Transport of Intensity Equation (TIE) phase retrieval.
        
        Uses intensity measurements at two or three closely spaced planes.
        Solves: ∇²φ = -(2πk/I) · ∂I/∂z
        
        Args:
            intensities: At least 2 intensity measurements at different z.
            distances: Corresponding propagation distances.
            wavelength: Wavelength [m].
            pixel_size: Pixel size [m].
        
        Returns:
            Retrieved phase map.
        """
        k = 2 * np.pi / wavelength
        
        if len(intensities) == 2:
            dI_dz = (intensities[1] - intensities[0]) / (distances[1] - distances[0])
            I_avg = (intensities[0] + intensities[1]) / 2
        else:
            # Central difference
            dI_dz = (intensities[2] - intensities[0]) / (distances[2] - distances[0])
            I_avg = intensities[1]
        
        Ny, Nx = I_avg.shape
        
        # Solve Poisson equation: ∇²φ = rhs
        rhs = -(k / np.maximum(I_avg, 1e-30)) * dI_dz
        
        # Solve in Fourier domain
        fx = np.fft.fftfreq(Nx, d=pixel_size)
        fy = np.fft.fftfreq(Ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy)
        
        lap_kernel = -4 * np.pi**2 * (FX**2 + FY**2)
        lap_kernel[0, 0] = 1.0  # Avoid division by zero (DC component)
        
        rhs_ft = np.fft.fft2(rhs)
        phase_ft = rhs_ft / lap_kernel
        phase_ft[0, 0] = 0  # Zero mean phase
        
        phase = np.real(np.fft.ifft2(phase_ft))
        return phase


# ======================== Convenience Functions ========================

def run_phase_retrieval(measured_intensity: np.ndarray,
                        algorithm: str = 'hio',
                        support: Optional[np.ndarray] = None,
                        max_iterations: int = 300,
                        beta: float = 0.8,
                        seed: int = 42,
                        propagator_forward: Optional[Callable] = None,
                        propagator_backward: Optional[Callable] = None,
                        ) -> Tuple[np.ndarray, List[float]]:
    """Convenience function to run phase retrieval with a single call.
    
    Args:
        measured_intensity: Measured |U|² at detector.
        algorithm: 'gs', 'er', 'hio', or 'raar'.
        support: Binary support mask (optional).
        max_iterations: Number of iterations.
        beta: HIO/RAAR parameter.
        seed: Random seed.
        propagator_forward: Forward propagation function.
        propagator_backward: Backward propagation function.
    
    Returns:
        (recovered_field, error_history)
    """
    params = PhaseRetrievalParams(
        max_iterations=max_iterations,
        beta=beta,
        support=support,
        use_support=support is not None,
        seed=seed,
    )
    
    retriever = PhaseRetriever(params)
    
    algorithm_map = {
        'gs': retriever.gerchberg_saxton,
        'er': retriever.error_reduction,
        'hio': retriever.hybrid_input_output,
        'raar': retriever.raar,
    }
    
    if algorithm.lower() not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algorithm_map.keys())}")
    
    return algorithm_map[algorithm.lower()](
        measured_intensity,
        propagator_forward=propagator_forward,
        propagator_backward=propagator_backward,
    )


def compare_algorithms(measured_intensity: np.ndarray,
                       ground_truth: Optional[np.ndarray] = None,
                       support: Optional[np.ndarray] = None,
                       algorithms: List[str] = None,
                       max_iterations: int = 300,
                       seed: int = 42) -> Dict[str, dict]:
    """Run and compare multiple phase retrieval algorithms.
    
    Returns:
        Dict mapping algorithm name → {'field': array, 'errors': list, 'metrics': dict}
    """
    if algorithms is None:
        algorithms = ['er', 'hio', 'raar']
    
    from . import metrics as met
    
    results = {}
    for algo in algorithms:
        field, errors = run_phase_retrieval(
            measured_intensity, algorithm=algo,
            support=support, max_iterations=max_iterations, seed=seed
        )
        
        result = {'field': field, 'errors': errors}
        
        if ground_truth is not None:
            result['metrics'] = met.all_metrics(ground_truth, field)
        
        results[algo.upper()] = result
    
    return results
