# INVISIBLE³D — Seeing 3D Matter Without a Lens

INVISIBLE³D is a research-oriented computational imaging framework for simulating coherent diffraction measurements and reconstructing complex refractive-index distributions.

## Current capabilities

- Angular-spectrum and Fresnel propagation
- Multislice beam propagation
- First-Born and Rytov forward models
- Iterative phase retrieval: GS, ER, HIO and RAAR
- Multi-distance phase retrieval and TIE
- Ptychographic simulation/reconstruction
- Multi-angle 3D forward modelling
- Adjoint-style gradient reconstruction through the multislice model
- Detector effects and noise simulation
- Quantitative reconstruction metrics
- Experiment configuration and parameter sweeps
- Scientific export and visualization utilities
- Optional neural-network initialization

## Important scientific scope

This repository is intended for computational research and algorithm development. Reconstruction quality must be validated against analytical phantoms, numerical forward-model consistency tests, and—where quantitative claims are made—an independent reference implementation.

The tomography module distinguishes detector amplitude from phase: intensity alone does not provide detector phase. A dedicated phase-retrieval stage is therefore required before a complex-field diffraction-tomography inversion.

The legacy reconstruct_sirt() API is retained for compatibility, but now delegates to an adjoint-style multislice gradient solver because simple residual backprojection is not a valid SIRT formulation for the nonlinear intensity diffraction model.

The Fourier-space reconstruct_fbp() implementation remains an approximate research prototype rather than a complete Ewald-sphere ODT solver.

## Installation

    git clone https://github.com/amankumarpatel-phy/INVISIBLE_3D.git
    cd INVISIBLE_3D
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/macOS:
    # source .venv/bin/activate
    pip install -r requirements.txt

## Running tests

    pytest -q

The regression suite checks rectangular grids, small-volume nanoparticle generation, phase-retrieval initialization, circular phase error, experiment serialization, and tomography angle units.

## Package layout

The current repository keeps the Python modules at the repository root so that the root __init__.py can expose the project as the INVISIBLE_3D package.

## Roadmap

1. Validate forward models against analytical solutions.
2. Add adjoint/gradient finite-difference checks.
3. Implemented an Ewald-sphere diffraction-tomography operator using q = k_s - k_i mapping.
4. Add quantitative Born/Rytov validation against known weak-scattering phantoms.
5. Add reproducible benchmark datasets and reconstruction reports.
6. Add continuous integration for regression tests.
7. Separate experimental prototypes from validated reconstruction algorithms.
