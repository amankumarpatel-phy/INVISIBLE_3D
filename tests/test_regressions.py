import sys
from pathlib import Path

import numpy as np

# The repository root is itself the Python package (INVISIBLE_3D).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

from INVISIBLE_3D.objects import ObjectGenerator, ObjectParams
from INVISIBLE_3D.experiment import ExperimentRunner, preset_tomography
from INVISIBLE_3D.metrics import refractive_index_error
from INVISIBLE_3D.phase_retrieval import PhaseRetriever, PhaseRetrievalParams
from INVISIBLE_3D.tomography import TomographyEngine, TomographyParams


def test_rectangular_object_grid():
    gen = ObjectGenerator(ObjectParams(grid_size=(12, 20), pixel_size=1e-6))
    obj = gen.disc(radius=2e-6)
    assert obj.shape == (12, 20)


def test_small_nanoparticle_volume_does_not_fail():
    params = ObjectParams(
        grid_size=(16, 16, 16),
        pixel_size=1e-6,
        slice_thickness=1e-6,
    )
    obj = ObjectGenerator(params).nanoparticles(
        num_particles=2, radius_range=(0.2e-6, 0.5e-6), seed=1
    )
    assert obj.shape == (16, 16, 16)
    assert np.iscomplexobj(obj)


def test_zero_initialization_is_zero():
    retriever = PhaseRetriever(
        PhaseRetrievalParams(init_method="zeros", seed=1)
    )
    field = retriever._initialize(np.ones((8, 8)))
    assert np.all(field == 0)


def test_phase_metric_wraps_branch_cut():
    gt = np.exp(1j * (np.pi - 0.01)) * np.ones((4, 4))
    recon = np.exp(1j * (-np.pi + 0.01)) * np.ones((4, 4))
    metrics = refractive_index_error(gt, recon)
    assert metrics["phase_mse"] < 1e-3


def test_parametric_study_preserves_numeric_inf():
    base = preset_tomography()
    result = ExperimentRunner().run_parametric_study(
        base, "max_iterations", [1], lambda cfg: {
            "saturation_is_inf": bool(np.isinf(cfg.saturation_level)),
            "na_is_inf": bool(np.isinf(cfg.numerical_aperture)),
        }
    )
    assert result["metrics"][0]["saturation_is_inf"]
    assert result["metrics"][0]["na_is_inf"]


def test_tomography_preset_angles_are_radians():
    p = preset_tomography()
    assert np.isclose(p.angle_range[0], -np.pi / 3)
    assert np.isclose(p.angle_range[1], np.pi / 3)


def test_ewald_tomography_operator_matches_born_data_shape_and_finiteness():
    p = TomographyParams(
        Nx=16, Ny=16, Nz=8,
        pixel_size=1e-6, slice_thickness=1e-6,
        wavelength=532e-9, n_background=1.33,
        z_detector=20e-6, num_angles=5,
        angle_range=(-0.2, 0.2), tikhonov_weight=1e-3,
    )
    engine = TomographyEngine(p)

    phantom = np.full((p.Nz, p.Ny, p.Nx), p.n_background, dtype=np.complex128)
    phantom[p.Nz // 2, p.Ny // 2, p.Nx // 2] += 1e-3

    scattered = engine.forward_born_multi_angle(phantom)
    assert len(scattered) == p.num_angles
    assert all(field.shape == (p.Ny, p.Nx) for field in scattered)

    recon = engine.reconstruct_fbp(scattered)
    assert recon.shape == phantom.shape
    assert np.all(np.isfinite(recon.real))
    assert np.all(np.isfinite(recon.imag))
