"""
Object generation module for INVISIBLE³D.

Generates synthetic 2D and 3D objects with complex refractive-index distributions.
Object representation: n(x,y,z) + i·κ(x,y,z)
where n is the real refractive index and κ is the extinction coefficient.

Supports binary objects, continuous distributions, and composite structures.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
from .utils import coordinate_grid_2d, coordinate_grid_3d


@dataclass
class ObjectParams:
    """Parameters for synthetic object generation."""
    grid_size: Tuple[int, ...]           # (Ny, Nx) for 2D or (Nz, Ny, Nx) for 3D
    pixel_size: float = 0.5e-6           # Lateral pixel size [m]
    slice_thickness: Optional[float] = None  # Axial step [m], defaults to pixel_size
    wavelength: float = 632.8e-9         # Illumination wavelength [m]
    background_ri: complex = 1.0 + 0j    # Background refractive index

    def __post_init__(self):
        if self.slice_thickness is None:
            self.slice_thickness = self.pixel_size


class ObjectGenerator:
    """Generate synthetic 2D and 3D complex refractive-index objects.
    
    All objects are returned as complex arrays where:
      - real part = refractive index n
      - imaginary part = extinction coefficient κ
    
    The transmission function t = exp(i·k·Δn·thickness) can be derived from
    the refractive-index contrast Δn = n_object - n_background.
    """

    def __init__(self, params: ObjectParams):
        self.params = params
        self.ndim = len(params.grid_size)
        self._init_grids()

    def _init_grids(self):
        """Initialize coordinate grids based on dimensionality."""
        p = self.params
        if self.ndim == 2:
            Ny, Nx = p.grid_size
            self.X, self.Y = coordinate_grid_2d(Nx, p.pixel_size)
            self.R = np.sqrt(self.X**2 + self.Y**2)
        elif self.ndim == 3:
            Nz, Ny, Nx = p.grid_size
            self.X, self.Y, self.Z = coordinate_grid_3d(
                Nz, Ny, Nx, p.pixel_size, p.slice_thickness
            )
            self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.ndim}")

    def _empty(self) -> np.ndarray:
        """Create empty object filled with background refractive index."""
        return np.full(self.params.grid_size, self.params.background_ri, dtype=np.complex128)

    # ======================== 2D Objects ========================

    def disc(self, center: Tuple[float, float] = (0, 0),
             radius: float = 5e-6,
             ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a 2D disc (circular cross-section).
        
        Args:
            center: (x0, y0) center position in meters.
            radius: Disc radius in meters.
            ri: Complex refractive index of the disc.
        
        Returns:
            2D complex refractive-index array.
        """
        obj = self._empty()
        r = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
        mask = r <= radius
        obj[mask] = ri
        return obj

    def ring(self, center: Tuple[float, float] = (0, 0),
             inner_radius: float = 3e-6,
             outer_radius: float = 6e-6,
             ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a 2D ring (annular cross-section)."""
        obj = self._empty()
        r = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
        mask = (r >= inner_radius) & (r <= outer_radius)
        obj[mask] = ri
        return obj

    def rectangle(self, center: Tuple[float, float] = (0, 0),
                  width: float = 8e-6,
                  height: float = 5e-6,
                  ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a 2D rectangle."""
        obj = self._empty()
        mask = ((np.abs(self.X - center[0]) <= width / 2) &
                (np.abs(self.Y - center[1]) <= height / 2))
        obj[mask] = ri
        return obj

    def gaussian_phase_object(self, center: Tuple[float, float] = (0, 0),
                               sigma: float = 3e-6,
                               max_delta_n: float = 0.1) -> np.ndarray:
        """Generate a 2D Gaussian refractive-index distribution (pure phase object)."""
        obj = self._empty()
        r2 = (self.X - center[0])**2 + (self.Y - center[1])**2
        dn = max_delta_n * np.exp(-r2 / (2 * sigma**2))
        obj += dn  # Add to background
        return obj

    def multilayer_2d(self, layers: List[dict]) -> np.ndarray:
        """Generate concentric multilayer 2D object.
        
        Args:
            layers: List of dicts with keys 'radius' and 'ri'.
                    Layers are painted outer-to-inner.
        
        Returns:
            2D complex refractive-index array.
        """
        obj = self._empty()
        # Sort layers by radius descending to paint outer first
        sorted_layers = sorted(layers, key=lambda l: l['radius'], reverse=True)
        for layer in sorted_layers:
            r = self.R if self.ndim == 2 else np.sqrt(self.X**2 + self.Y**2)
            mask = r <= layer['radius']
            obj[mask] = layer['ri']
        return obj

    def siemens_star(self, num_spokes: int = 36,
                     radius: float = 8e-6,
                     ri: complex = 1.5 + 0j) -> np.ndarray:
        """Generate a Siemens star resolution target (2D)."""
        obj = self._empty()
        theta = np.arctan2(self.Y, self.X)
        r = np.sqrt(self.X**2 + self.Y**2)
        # Alternating spokes
        spoke_width = np.pi / num_spokes
        spoke_mask = (np.mod(theta, 2 * spoke_width) < spoke_width) & (r <= radius)
        obj[spoke_mask] = ri
        return obj

    # ======================== 3D Objects ========================

    def sphere(self, center: Tuple[float, float, float] = (0, 0, 0),
               radius: float = 5e-6,
               ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a 3D sphere."""
        assert self.ndim == 3, "sphere() requires 3D grid"
        obj = self._empty()
        r = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2 + (self.Z - center[2])**2)
        obj[r <= radius] = ri
        return obj

    def ellipsoid(self, center: Tuple[float, float, float] = (0, 0, 0),
                  semi_axes: Tuple[float, float, float] = (6e-6, 4e-6, 3e-6),
                  ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a 3D ellipsoid.
        
        Args:
            semi_axes: (ax, ay, az) semi-axis lengths in meters.
        """
        assert self.ndim == 3, "ellipsoid() requires 3D grid"
        obj = self._empty()
        ax, ay, az = semi_axes
        dist = ((self.X - center[0]) / ax)**2 + \
               ((self.Y - center[1]) / ay)**2 + \
               ((self.Z - center[2]) / az)**2
        obj[dist <= 1.0] = ri
        return obj

    def cube(self, center: Tuple[float, float, float] = (0, 0, 0),
             side: float = 8e-6,
             ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a 3D cube."""
        assert self.ndim == 3, "cube() requires 3D grid"
        obj = self._empty()
        half = side / 2
        mask = ((np.abs(self.X - center[0]) <= half) &
                (np.abs(self.Y - center[1]) <= half) &
                (np.abs(self.Z - center[2]) <= half))
        obj[mask] = ri
        return obj

    def hollow_sphere(self, center: Tuple[float, float, float] = (0, 0, 0),
                      inner_radius: float = 3e-6,
                      outer_radius: float = 6e-6,
                      ri: complex = 1.5 + 0.01j) -> np.ndarray:
        """Generate a hollow sphere (shell)."""
        assert self.ndim == 3, "hollow_sphere() requires 3D grid"
        obj = self._empty()
        r = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2 + (self.Z - center[2])**2)
        mask = (r >= inner_radius) & (r <= outer_radius)
        obj[mask] = ri
        return obj

    def multilayer_sphere(self, center: Tuple[float, float, float] = (0, 0, 0),
                          layers: List[dict] = None) -> np.ndarray:
        """Generate a multilayer (core-shell) sphere.
        
        Args:
            layers: List of dicts with 'radius' and 'ri'.
                    Painted from outermost to innermost.
        """
        assert self.ndim == 3, "multilayer_sphere() requires 3D grid"
        if layers is None:
            layers = [
                {'radius': 6e-6, 'ri': 1.4 + 0.005j},
                {'radius': 4e-6, 'ri': 1.5 + 0.01j},
                {'radius': 2e-6, 'ri': 1.6 + 0.02j},
            ]
        obj = self._empty()
        r = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2 + (self.Z - center[2])**2)
        sorted_layers = sorted(layers, key=lambda l: l['radius'], reverse=True)
        for layer in sorted_layers:
            obj[r <= layer['radius']] = layer['ri']
        return obj

    def nanoparticles(self, num_particles: int = 20,
                     radius_range: Tuple[float, float] = (0.5e-6, 2e-6),
                     ri: complex = 1.6 + 0.02j,
                     seed: int = 42) -> np.ndarray:
        """Generate random nanoparticle distribution.
        
        Args:
            num_particles: Number of particles.
            radius_range: (min_radius, max_radius) in meters.
            ri: Complex RI of particles.
            seed: Random seed for reproducibility.
        """
        assert self.ndim == 3, "nanoparticles() requires 3D grid"
        rng = np.random.default_rng(seed)
        obj = self._empty()
        p = self.params

        # Volume bounds
        extent_x = (p.grid_size[2] // 2 - 5) * p.pixel_size
        extent_y = (p.grid_size[1] // 2 - 5) * p.pixel_size
        extent_z = (p.grid_size[0] // 2 - 5) * p.slice_thickness

        for _ in range(num_particles):
            cx = rng.uniform(-extent_x, extent_x)
            cy = rng.uniform(-extent_y, extent_y)
            cz = rng.uniform(-extent_z, extent_z)
            rad = rng.uniform(*radius_range)
            r = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2)
            obj[r <= rad] = ri

        return obj

    def cylinder(self, center: Tuple[float, float, float] = (0, 0, 0),
                 radius: float = 4e-6,
                 height: float = 10e-6,
                 ri: complex = 1.5 + 0.01j,
                 axis: str = 'z') -> np.ndarray:
        """Generate a 3D cylinder aligned along a specified axis."""
        assert self.ndim == 3, "cylinder() requires 3D grid"
        obj = self._empty()
        half_h = height / 2
        if axis == 'z':
            r_lat = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
            mask = (r_lat <= radius) & (np.abs(self.Z - center[2]) <= half_h)
        elif axis == 'y':
            r_lat = np.sqrt((self.X - center[0])**2 + (self.Z - center[2])**2)
            mask = (r_lat <= radius) & (np.abs(self.Y - center[1]) <= half_h)
        else:
            r_lat = np.sqrt((self.Y - center[1])**2 + (self.Z - center[2])**2)
            mask = (r_lat <= radius) & (np.abs(self.X - center[0]) <= half_h)
        obj[mask] = ri
        return obj

    def user_defined(self, voxels: np.ndarray) -> np.ndarray:
        """Load a user-defined voxel array as a refractive-index distribution.
        
        Args:
            voxels: Array of same shape as grid_size. Can be:
                    - binary mask (will be converted using background and 1.5+0j)
                    - real-valued (interpreted as real RI)
                    - complex-valued (used directly)
        
        Returns:
            Complex RI distribution.
        """
        expected_shape = self.params.grid_size
        if voxels.shape != expected_shape:
            raise ValueError(f"Voxel array shape {voxels.shape} doesn't match grid {expected_shape}")
        obj = self._empty()
        if voxels.dtype == bool or set(np.unique(voxels)).issubset({0, 1}):
            mask = voxels.astype(bool)
            obj[mask] = 1.5 + 0j
        elif np.isrealobj(voxels):
            obj = voxels.astype(np.complex128)
        else:
            obj = voxels.copy()
        return obj

    def composite(self, objects: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> np.ndarray:
        """Combine multiple objects with optional masks.
        
        Args:
            objects: List of (ri_array, mask) tuples.
                     If mask is None, non-background voxels are used.
        
        Returns:
            Combined complex RI distribution.
        """
        result = self._empty()
        bg = self.params.background_ri
        for ri_arr, mask in objects:
            if mask is None:
                mask = ~np.isclose(ri_arr, bg)
            result[mask] = ri_arr[mask]
        return result

    # ======================== Transmission Function ========================

    def ri_to_transmission(self, ri_distribution: np.ndarray,
                           thickness: Optional[float] = None) -> np.ndarray:
        """Convert a 2D refractive-index slice to a transmission function.
        
        t(x,y) = exp(i·k·Δn·thickness)
        
        where Δn = n_object - n_background.
        
        Args:
            ri_distribution: 2D complex RI array.
            thickness: Object thickness [m]. Defaults to pixel_size.
        
        Returns:
            Complex transmission function.
        """
        if thickness is None:
            thickness = self.params.pixel_size
        k = 2.0 * np.pi / self.params.wavelength
        delta_n = ri_distribution - self.params.background_ri
        return np.exp(1j * k * delta_n * thickness)

    def ri_to_scattering_potential(self, ri_distribution: np.ndarray) -> np.ndarray:
        """Convert refractive-index distribution to scattering potential.
        
        V(r) = k² · (n²(r) - n²_bg)
        
        Used in Born and Rytov approximations.
        """
        k = 2.0 * np.pi / self.params.wavelength
        n_bg = self.params.background_ri
        return k**2 * (ri_distribution**2 - n_bg**2)

    def get_support_mask(self, ri_distribution: np.ndarray,
                         threshold: float = 1e-6) -> np.ndarray:
        """Extract binary support mask from RI distribution.
        
        Returns True where object differs from background.
        """
        delta = np.abs(ri_distribution - self.params.background_ri)
        return delta > threshold


# ======================== Preset Objects ========================

def create_shepp_logan_2d(N: int = 256, pixel_size: float = 0.5e-6) -> np.ndarray:
    """Create a simplified 2D Shepp-Logan phantom for testing.
    
    Returns complex RI distribution.
    """
    obj = np.full((N, N), 1.0 + 0j, dtype=np.complex128)
    y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
    y = y / (N // 2)
    x = x / (N // 2)

    # Outer ellipse
    mask = (x / 0.69)**2 + (y / 0.92)**2 <= 1
    obj[mask] = 1.04 + 0.0j

    # Inner ellipse
    mask = (x / 0.6624)**2 + ((y - 0.0184) / 0.874)**2 <= 1
    obj[mask] = 1.02 + 0.0j

    # Small features
    for cx, cy, rx, ry, val in [
        (0.22, 0.0, 0.11, 0.31, 1.05),
        (-0.22, 0.0, 0.16, 0.41, 1.05),
        (0.0, 0.35, 0.21, 0.25, 1.03),
        (0.0, 0.1, 0.046, 0.046, 1.06),
        (-0.08, -0.605, 0.046, 0.023, 1.06),
        (0.06, -0.605, 0.046, 0.023, 1.06),
    ]:
        mask = ((x - cx) / rx)**2 + ((y - cy) / ry)**2 <= 1
        obj[mask] = val + 0j

    return obj


def create_cell_phantom_3d(Nz: int = 32, Ny: int = 64, Nx: int = 64,
                           pixel_size: float = 0.5e-6) -> np.ndarray:
    """Create a simplified 3D cell-like phantom.
    
    Features: outer membrane, cytoplasm, nucleus, nucleolus.
    """
    params = ObjectParams(
        grid_size=(Nz, Ny, Nx),
        pixel_size=pixel_size,
        wavelength=532e-9,
        background_ri=1.337 + 0j  # Water
    )
    gen = ObjectGenerator(params)

    # Cell membrane (outer shell)
    cell = gen.hollow_sphere(
        center=(0, 0, 0),
        inner_radius=8e-6,
        outer_radius=9e-6,
        ri=1.37 + 0.001j
    )

    # Cytoplasm
    r = np.sqrt(gen.X**2 + gen.Y**2 + gen.Z**2)
    cell[r <= 8e-6] = 1.36 + 0.0005j

    # Nucleus
    r_nuc = np.sqrt((gen.X - 1e-6)**2 + gen.Y**2 + gen.Z**2)
    cell[r_nuc <= 3.5e-6] = 1.39 + 0.002j

    # Nucleolus
    r_nucl = np.sqrt((gen.X - 1.5e-6)**2 + (gen.Y - 0.5e-6)**2 + gen.Z**2)
    cell[r_nucl <= 1.2e-6] = 1.42 + 0.005j

    return cell
