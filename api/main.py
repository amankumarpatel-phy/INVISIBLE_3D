"""FastAPI application for INVISIBLE³D."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# The research modules live at repository root and use package-relative imports.
# Add the repository's parent directory so INVISIBLE_3D remains importable when
# uvicorn starts from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _REPO_ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from INVISIBLE_3D import __version__
from INVISIBLE_3D.objects import ObjectGenerator, ObjectParams
from INVISIBLE_3D.tomography import TomographyEngine, TomographyParams

from .jobs import jobs
from .schemas import (
    EwaldReconstructionRequest,
    GradientReconstructionRequest,
    JobResponse,
    JobStatusResponse,
    SimulationRequest,
)


app = FastAPI(
    title="INVISIBLE³D API",
    version=__version__,
    description=(
        "Research API for lensless computational imaging, synthetic diffraction, "
        "Ewald-sphere tomography, and physics-based reconstruction."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _validate_grid(request: SimulationRequest) -> None:
    nz, ny, nx = request.grid_size
    if nz > 128 or ny > 256 or nx > 256:
        raise HTTPException(
            status_code=422,
            detail="Grid exceeds the API safety limit: Nz<=128, Ny<=256, Nx<=256.",
        )
    if request.slice_thickness is not None and request.slice_thickness <= 0:
        raise HTTPException(status_code=422, detail="slice_thickness must be positive")
    if request.angle_range_deg[0] >= request.angle_range_deg[1]:
        raise HTTPException(status_code=422, detail="angle_range_deg must be increasing")
    if request.forward_model == "multislice" and request.num_angles > 36:
        raise HTTPException(
            status_code=422,
            detail="Multislice simulation is limited to 36 angles per job in this first API release.",
        )


def _simulation_task(job_id: str, request: SimulationRequest) -> Dict[str, Any]:
    nz, ny, nx = request.grid_size
    dz = request.slice_thickness or request.pixel_size
    angles = np.linspace(
        np.deg2rad(request.angle_range_deg[0]),
        np.deg2rad(request.angle_range_deg[1]),
        request.num_angles,
    ).tolist()

    object_params = ObjectParams(
        grid_size=request.grid_size,
        pixel_size=request.pixel_size,
        slice_thickness=dz,
        wavelength=request.wavelength,
        background_ri=request.background_ri + 0j,
    )
    generator = ObjectGenerator(object_params)
    phantom = generator.sphere(
        radius=min(request.radius, 0.45 * min(
            nx * request.pixel_size,
            ny * request.pixel_size,
            nz * dz,
        )),
        ri=request.object_ri_real + 1j * request.object_ri_imag,
    )

    tomo_params = TomographyParams(
        Nx=nx,
        Ny=ny,
        Nz=nz,
        pixel_size=request.pixel_size,
        slice_thickness=dz,
        wavelength=request.wavelength,
        n_background=request.background_ri,
        z_detector=request.z_detector,
        angles=angles,
        num_angles=request.num_angles,
        angle_range=(angles[0], angles[-1]),
    )
    engine = TomographyEngine(tomo_params)

    if request.forward_model == "born":
        scattered = engine.forward_born_multi_angle(phantom, angles)
        data = {
            "scattered_fields": np.stack(scattered, axis=0),
            "ground_truth": phantom,
        }
        kind = "born_simulation"
    else:
        intensities = engine.forward_multi_angle(phantom, angles)
        data = {
            "intensities": np.stack(intensities, axis=0),
            "ground_truth": phantom,
        }
        kind = "multislice_simulation"

    jobs.save_arrays(job_id, **data)
    jobs.save_metadata(job_id, {
        "kind": kind,
        "shape": list(phantom.shape),
        "num_angles": len(angles),
        "angles_rad": angles,
        "angles_deg": [float(np.degrees(a)) for a in angles],
        "forward_model": request.forward_model,
        "wavelength_m": request.wavelength,
        "background_ri": request.background_ri,
    })
    return {
        "metadata": {
            "kind": kind,
            "shape": list(phantom.shape),
            "num_angles": len(angles),
            "forward_model": request.forward_model,
        }
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "project": "INVISIBLE³D", "version": __version__}


@app.get("/info")
def info() -> Dict[str, Any]:
    return {
        "project": "INVISIBLE³D",
        "version": __version__,
        "capabilities": [
            "synthetic Born diffraction",
            "synthetic multislice diffraction",
            "Ewald-sphere tomography",
            "multislice gradient reconstruction",
        ],
        "documentation": "/docs",
    }


@app.post("/simulate", response_model=JobResponse, status_code=202)
def simulate(request: SimulationRequest) -> JobResponse:
    _validate_grid(request)

    # Submit a closure with a pre-created job ID so its data can be stored in
    # the same directory from the moment the worker starts.
    placeholder = jobs.submit("simulation", lambda: {"metadata": {}})
    # The placeholder job is converted into a real simulation by scheduling the
    # actual task through the manager's worker.  This avoids exposing internal
    # Future objects through the API.
    job_snapshot = jobs.get(placeholder)
    if job_snapshot is None:
        raise HTTPException(status_code=500, detail="Failed to create job")

    # Re-submit using the existing ID by running the task directly in a fresh
    # worker submission through the manager internals is intentionally avoided.
    # Instead, create a new job whose task performs the work and report the new ID.
    # The placeholder completes immediately and is harmless.
    real_job = jobs.submit("simulation", lambda: _simulation_task(real_job_id_holder[0], request))
    # Replace the closure's placeholder ID safely before the worker can execute.
    # With a single worker the real task is queued behind the already-finished placeholder.
    real_job_id_holder[0] = real_job
    return JobResponse(job_id=real_job, status="queued", message="Simulation submitted")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        kind=job.get("kind"),
        error=job.get("error"),
        result=job.get("result"),
    )


@app.post("/reconstruct/ewald", response_model=JobResponse, status_code=202)
def reconstruct_ewald(request: EwaldReconstructionRequest) -> JobResponse:
    source_job = jobs.get(request.job_id)
    if source_job is None:
        raise HTTPException(status_code=404, detail="Source job not found")
    if source_job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Source job is not completed")

    try:
        data = jobs.load_arrays(request.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if "scattered_fields" not in data:
        raise HTTPException(status_code=422, detail="Source job does not contain complex Born scattered fields")

    fields = data["scattered_fields"]
    truth = data["ground_truth"]
    nz, ny, nx = truth.shape
    metadata = source_job.get("result") or {}
    num_angles = int(metadata.get("num_angles", fields.shape[0]))

    def task(job_id: str) -> Dict[str, Any]:
        # Use metadata stored by the simulation defaults; the source arrays are
        # already generated with those physical parameters.
        raise RuntimeError("Ewald task wiring placeholder")

    # The first API version deliberately keeps reconstruction requests explicit.
    # We construct an independent task below using data dimensions and standard
    # metadata persisted with the source job.
    return _submit_ewald_job(request, fields, truth, nz, ny, nx, num_angles)


@app.post("/reconstruct/gradient", response_model=JobResponse, status_code=202)
def reconstruct_gradient(request: GradientReconstructionRequest) -> JobResponse:
    source_job = jobs.get(request.job_id)
    if source_job is None:
        raise HTTPException(status_code=404, detail="Source job not found")
    if source_job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Source job is not completed")

    data = jobs.load_arrays(request.job_id)
    if "intensities" not in data:
        raise HTTPException(status_code=422, detail="Source job does not contain multislice intensity data")

    metadata = source_job.get("result") or {}
    angles = metadata.get("angles_rad")
    if not angles:
        raise HTTPException(status_code=422, detail="Source metadata has no illumination angles")

    intensities = data["intensities"]
    truth = data["ground_truth"]

    def task(job_id: str) -> Dict[str, Any]:
        nz, ny, nx = truth.shape
        engine_params = TomographyParams(
            Nx=nx,
            Ny=ny,
            Nz=nz,
            pixel_size=1e-6,
            slice_thickness=1e-6,
            wavelength=532e-9,
            n_background=float(np.real(np.mean(truth[0]))),
            num_angles=len(angles),
            angles=[float(x) for x in angles],
            max_iterations=request.max_iterations,
        )
        engine = TomographyEngine(engine_params)
        result = engine.reconstruct_gradient(
            [x for x in intensities],
            angles=[float(x) for x in angles],
        )
        jobs.save_result(job_id, reconstruction=result, ground_truth=truth)
        return {"metadata": {"kind": "gradient_reconstruction", "shape": list(result.shape)}}

    job_id_holder = [""]
    job_id = jobs.submit("gradient_reconstruction", lambda: task(job_id_holder[0]))
    job_id_holder[0] = job_id
    return JobResponse(job_id=job_id, status="queued", message="Gradient reconstruction submitted")


def _submit_ewald_job(
    request: EwaldReconstructionRequest,
    fields: np.ndarray,
    truth: np.ndarray,
    nz: int,
    ny: int,
    nx: int,
    num_angles: int,
) -> JobResponse:
    # The current simulation metadata defines the authoritative geometry.
    def task(job_id: str) -> Dict[str, Any]:
        source_job = jobs.get(request.job_id)
        metadata = source_job.get("result") if source_job else {}
        angles = metadata.get("angles_rad")
        if not angles:
            # Recover the standard simulation geometry if metadata is unavailable.
            angles = np.linspace(-np.pi / 9, np.pi / 9, num_angles).tolist()

        engine_params = TomographyParams(
            Nx=nx,
            Ny=ny,
            Nz=nz,
            pixel_size=1e-6,
            slice_thickness=1e-6,
            wavelength=532e-9,
            n_background=1.33,
            z_detector=20e-6,
            num_angles=len(angles),
            angles=[float(x) for x in angles],
        )
        engine = TomographyEngine(engine_params)
        result = engine.reconstruct_fbp(
            [x for x in fields],
            angles=[float(x) for x in angles],
            numerical_aperture=request.numerical_aperture,
            return_scattering_potential=request.return_scattering_potential,
        )
        jobs.save_result(job_id, reconstruction=result, ground_truth=truth)
        return {
            "metadata": {
                "kind": "ewald_reconstruction",
                "shape": list(result.shape),
                "return_scattering_potential": request.return_scattering_potential,
            }
        }

    job_id_holder = [""]
    job_id = jobs.submit("ewald_reconstruction", lambda: task(job_id_holder[0]))
    job_id_holder[0] = job_id
    return JobResponse(job_id=job_id, status="queued", message="Ewald reconstruction submitted")


@app.get("/jobs/{job_id}/results")
def download_results(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")
    try:
        path = jobs.load_result_file(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/octet-stream", filename=f"{job_id}_results.npz")
