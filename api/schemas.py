"""Pydantic request/response models for the INVISIBLE³D API."""

from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Configuration for synthetic diffraction-data generation."""

    object_type: Literal["sphere"] = "sphere"
    grid_size: Tuple[int, int, int] = Field(
        default=(16, 32, 32), description="(Nz, Ny, Nx)"
    )
    pixel_size: float = Field(default=1e-6, gt=0, description="Lateral sampling [m]")
    slice_thickness: Optional[float] = Field(
        default=None, gt=0, description="Axial sampling [m]"
    )
    wavelength: float = Field(default=532e-9, gt=0, description="Vacuum wavelength [m]")
    background_ri: float = Field(default=1.33, gt=0)
    object_ri_real: float = Field(default=1.40, gt=0)
    object_ri_imag: float = Field(default=0.0, ge=0)
    radius: float = Field(default=5e-6, gt=0, description="Sphere radius [m]")
    z_detector: float = Field(default=20e-6, gt=0, description="Detector distance [m]")
    num_angles: int = Field(default=9, ge=1, le=72)
    angle_range_deg: Tuple[float, float] = (-20.0, 20.0)
    forward_model: Literal["born", "multislice"] = "born"


class EwaldReconstructionRequest(BaseModel):
    """Run Ewald-sphere reconstruction from a simulation job."""

    job_id: str = Field(min_length=1)
    numerical_aperture: Optional[float] = Field(default=None, gt=0)
    return_scattering_potential: bool = False


class GradientReconstructionRequest(BaseModel):
    """Run nonlinear multislice gradient reconstruction from a simulation job."""

    job_id: str = Field(min_length=1)
    max_iterations: int = Field(default=10, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0)


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    kind: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None
