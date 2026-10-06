"""
Export module for INVISIBLE³D.

Supports exporting reconstructed data in multiple formats:
  - TIFF slices
  - NumPy arrays (.npy / .npz)
  - CSV voxel data
  - HDF5 datasets
  - PNG/SVG publication figures
  - PDF report
"""

import numpy as np
import os
import json
import datetime
from typing import Optional, Dict, List, Union
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def export_numpy(data: np.ndarray, filepath: str):
    """Export data as NumPy .npy file."""
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, data)


def export_numpy_compressed(data_dict: Dict[str, np.ndarray], filepath: str):
    """Export multiple arrays as compressed .npz file."""
    ensure_dir(os.path.dirname(filepath))
    np.savez_compressed(filepath, **data_dict)


def export_tiff_slices(volume: np.ndarray, output_dir: str,
                        prefix: str = 'slice', component: str = 'real'):
    """Export 3D volume as a stack of TIFF slices.
    
    Args:
        volume: 3D array [Nz, Ny, Nx].
        output_dir: Output directory.
        prefix: Filename prefix.
        component: 'real', 'imag', 'magnitude', or 'phase'.
    """
    ensure_dir(output_dir)
    
    try:
        import tifffile
    except ImportError:
        # Fallback to PIL
        from PIL import Image
        tifffile = None
    
    if component == 'real':
        data = volume.real
    elif component == 'imag':
        data = volume.imag
    elif component == 'magnitude':
        data = np.abs(volume)
    elif component == 'phase':
        data = np.angle(volume)
    else:
        data = volume.real
    
    # Normalize to 16-bit
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        normalized = ((data - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
    else:
        normalized = np.zeros_like(data, dtype=np.uint16)
    
    for iz in range(normalized.shape[0]):
        filename = os.path.join(output_dir, f'{prefix}_{iz:04d}.tiff')
        if tifffile is not None:
            tifffile.imwrite(filename, normalized[iz])
        else:
            from PIL import Image
            Image.fromarray(normalized[iz]).save(filename)
    
    # Save metadata
    meta = {
        'component': component,
        'data_min': float(d_min),
        'data_max': float(d_max),
        'num_slices': int(normalized.shape[0]),
        'shape': list(volume.shape),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def export_tiff_stack(volume: np.ndarray, filepath: str, component: str = 'real'):
    """Export 3D volume as a single multi-page TIFF file."""
    ensure_dir(os.path.dirname(filepath))
    
    if component == 'real':
        data = volume.real
    elif component == 'imag':
        data = volume.imag
    elif component == 'magnitude':
        data = np.abs(volume)
    else:
        data = np.angle(volume)
    
    # Normalize to float32
    data32 = data.astype(np.float32)
    
    try:
        import tifffile
        tifffile.imwrite(filepath, data32)
    except ImportError:
        # Save individual slices
        export_tiff_slices(volume, os.path.dirname(filepath), component=component)


def export_csv(data: np.ndarray, filepath: str, include_coords: bool = True,
               pixel_size: float = 1e-6):
    """Export 2D or 3D data as CSV file.
    
    For 3D data, exports as (z, y, x, real, imag) columns.
    """
    ensure_dir(os.path.dirname(filepath))
    
    if data.ndim == 2:
        if include_coords:
            Ny, Nx = data.shape
            rows = []
            for iy in range(Ny):
                for ix in range(Nx):
                    y = (iy - Ny // 2) * pixel_size
                    x = (ix - Nx // 2) * pixel_size
                    val = data[iy, ix]
                    if np.iscomplexobj(data):
                        rows.append(f'{x:.6e},{y:.6e},{val.real:.6e},{val.imag:.6e}')
                    else:
                        rows.append(f'{x:.6e},{y:.6e},{val:.6e}')
            
            header = 'x_m,y_m,real,imag' if np.iscomplexobj(data) else 'x_m,y_m,value'
        else:
            np.savetxt(filepath, data.real, delimiter=',')
            return
    elif data.ndim == 3:
        Nz, Ny, Nx = data.shape
        rows = []
        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx):
                    z = (iz - Nz // 2) * pixel_size
                    y = (iy - Ny // 2) * pixel_size
                    x = (ix - Nx // 2) * pixel_size
                    val = data[iz, iy, ix]
                    if np.iscomplexobj(data):
                        rows.append(f'{x:.6e},{y:.6e},{z:.6e},{val.real:.6e},{val.imag:.6e}')
                    else:
                        rows.append(f'{x:.6e},{y:.6e},{z:.6e},{val:.6e}')
        
        header = 'x_m,y_m,z_m,real,imag' if np.iscomplexobj(data) else 'x_m,y_m,z_m,value'
    else:
        raise ValueError(f"Unsupported array dimensions: {data.ndim}")
    
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))


def export_hdf5(data_dict: Dict[str, np.ndarray], filepath: str,
                metadata: Optional[Dict] = None):
    """Export data as HDF5 file.
    
    Args:
        data_dict: Dictionary mapping dataset names to arrays.
        filepath: Output .h5 file path.
        metadata: Optional metadata dictionary (stored as attributes).
    """
    ensure_dir(os.path.dirname(filepath))
    
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 export. Install: pip install h5py")
    
    with h5py.File(filepath, 'w') as f:
        for name, data in data_dict.items():
            if np.iscomplexobj(data):
                grp = f.create_group(name)
                grp.create_dataset('real', data=data.real, compression='gzip')
                grp.create_dataset('imag', data=data.imag, compression='gzip')
            else:
                f.create_dataset(name, data=data, compression='gzip')
        
        if metadata:
            for key, val in metadata.items():
                try:
                    f.attrs[key] = val
                except TypeError:
                    f.attrs[key] = str(val)


def export_figure(fig, filepath: str, formats: List[str] = None):
    """Export matplotlib figure to PNG, SVG, or PDF.
    
    Args:
        fig: Matplotlib figure.
        filepath: Base filepath (without extension).
        formats: List of formats ('png', 'svg', 'pdf').
    """
    if formats is None:
        formats = ['png']
    
    ensure_dir(os.path.dirname(filepath))
    
    for fmt in formats:
        fig.savefig(f'{filepath}.{fmt}', format=fmt, dpi=300, bbox_inches='tight')


def export_pdf_report(figures: List, titles: List[str],
                       metadata: Dict, filepath: str):
    """Generate a PDF report with figures and metadata.
    
    Args:
        figures: List of matplotlib figures.
        titles: Corresponding titles.
        metadata: Experiment metadata dict.
        filepath: Output PDF path.
    """
    ensure_dir(os.path.dirname(filepath))
    
    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: save figures as individual PNGs
        base = os.path.splitext(filepath)[0]
        for i, (fig, title) in enumerate(zip(figures, titles)):
            fig.savefig(f'{base}_fig{i}_{title.replace(" ", "_")}.png', dpi=300)
        return
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 30, 'INVISIBLE 3D', ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, 'Reconstruction Report', ln=True, align='C')
    pdf.cell(0, 10, f'Generated: {datetime.datetime.now().isoformat()}', ln=True, align='C')
    
    # Metadata
    pdf.ln(10)
    pdf.set_font('Courier', '', 9)
    for key, val in metadata.items():
        pdf.cell(0, 5, f'{key}: {val}', ln=True)
    
    # Figures
    import tempfile
    for fig, title in zip(figures, titles):
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, title, ln=True, align='C')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name, dpi=200, bbox_inches='tight')
            pdf.image(tmp.name, x=10, y=30, w=190)
            os.unlink(tmp.name)
    
    pdf.output(filepath)


def export_experiment(output_dir: str,
                       ground_truth: Optional[np.ndarray] = None,
                       reconstruction: Optional[np.ndarray] = None,
                       diffraction: Optional[np.ndarray] = None,
                       metrics: Optional[Dict] = None,
                       config: Optional[Dict] = None,
                       figures: Optional[List] = None,
                       figure_titles: Optional[List[str]] = None):
    """Export a complete experiment's results.
    
    Creates:
      output_dir/
        data/
          ground_truth.npy
          reconstruction.npy
          diffraction.npy
          data.h5
        figures/
          *.png
        metrics.json
        config.json
        report.pdf
    """
    ensure_dir(output_dir)
    data_dir = os.path.join(output_dir, 'data')
    fig_dir = os.path.join(output_dir, 'figures')
    ensure_dir(data_dir)
    ensure_dir(fig_dir)
    
    # Data
    data_dict = {}
    if ground_truth is not None:
        export_numpy(ground_truth, os.path.join(data_dir, 'ground_truth.npy'))
        data_dict['ground_truth'] = ground_truth
    if reconstruction is not None:
        export_numpy(reconstruction, os.path.join(data_dir, 'reconstruction.npy'))
        data_dict['reconstruction'] = reconstruction
    if diffraction is not None:
        export_numpy(diffraction, os.path.join(data_dir, 'diffraction.npy'))
        data_dict['diffraction'] = diffraction
    
    if data_dict:
        export_hdf5(data_dict, os.path.join(data_dir, 'data.h5'),
                     metadata=config)
    
    # Metrics
    if metrics:
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    # Config
    if config:
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    # Figures
    if figures:
        for i, fig in enumerate(figures):
            title = figure_titles[i] if figure_titles and i < len(figure_titles) else f'figure_{i}'
            fig.savefig(os.path.join(fig_dir, f'{title.replace(" ", "_")}.png'),
                       dpi=300, bbox_inches='tight')
        
        # PDF report
        if figure_titles is None:
            figure_titles = [f'Figure {i}' for i in range(len(figures))]
        export_pdf_report(figures, figure_titles, config or {},
                          os.path.join(output_dir, 'report.pdf'))
