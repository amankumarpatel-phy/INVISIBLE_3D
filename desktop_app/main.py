"""INVISIBLE³D standalone desktop application.

This GUI is intentionally a thin presentation layer over the research engine.
It can run synthetic Born diffraction experiments and Ewald-sphere
reconstruction without requiring Python to be installed on the target PC.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

# Make the repository package importable when launched from source.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from INVISIBLE_3D.objects import ObjectGenerator, ObjectParams
from INVISIBLE_3D.tomography import TomographyEngine, TomographyParams


APP_NAME = "INVISIBLE³D"
APP_VERSION = "1.0.0"


class SimulationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        grid: int,
        angles_count: int,
        angle_limit: float,
        wavelength_nm: float,
        background_ri: float,
        object_ri: float,
        radius_um: float,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.angles_count = angles_count
        self.angle_limit = angle_limit
        self.wavelength = wavelength_nm * 1e-9
        self.background_ri = background_ri
        self.object_ri = object_ri
        self.radius = radius_um * 1e-6

    @Slot()
    def run(self) -> None:
        try:
            n = self.grid
            pixel = 1.0e-6
            dz = 1.0e-6
            angles = np.linspace(
                np.deg2rad(-self.angle_limit),
                np.deg2rad(self.angle_limit),
                self.angles_count,
            ).tolist()

            self.progress.emit("Generating 3D phantom…")
            object_params = ObjectParams(
                grid_size=(n, n, n),
                pixel_size=pixel,
                slice_thickness=dz,
                wavelength=self.wavelength,
                background_ri=self.background_ri + 0j,
            )
            phantom = ObjectGenerator(object_params).sphere(
                radius=min(self.radius, 0.4 * n * pixel),
                ri=self.object_ri + 0j,
            )

            self.progress.emit("Simulating first-Born diffraction…")
            tomo_params = TomographyParams(
                Nx=n,
                Ny=n,
                Nz=n,
                pixel_size=pixel,
                slice_thickness=dz,
                wavelength=self.wavelength,
                n_background=self.background_ri,
                z_detector=20e-6,
                angles=angles,
                num_angles=len(angles),
                angle_range=(angles[0], angles[-1]),
            )
            engine = TomographyEngine(tomo_params)
            scattered = engine.forward_born_multi_angle(phantom, angles)

            self.progress.emit("Running Ewald-sphere reconstruction…")
            reconstruction = engine.reconstruct_fbp(
                scattered,
                angles=angles,
            )

            self.finished.emit(
                {
                    "phantom": phantom,
                    "reconstruction": reconstruction,
                    "scattered_fields": np.stack(scattered, axis=0),
                    "angles": np.asarray(angles),
                    "wavelength": self.wavelength,
                    "pixel_size": pixel,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class SliceView(QWidget):
    """Simple Qt-native image view using a QLabel-rendered numpy slice.

    Keeping this dependency-free avoids requiring a second plotting runtime in
    the installed application. The numeric result can still be exported as NPZ.
    """

    def __init__(self) -> None:
        super().__init__()
        self.label = QLabel("Run a reconstruction to display the result.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(560, 420)
        self.label.setStyleSheet(
            "QLabel { background: #111827; color: #cbd5e1; border: 1px solid #334155; }"
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def show_array(self, array: np.ndarray) -> None:
        from PySide6.QtGui import QImage, QPixmap

        data = np.asarray(np.real(array), dtype=np.float64)
        if data.ndim == 3:
            data = data[data.shape[0] // 2]
        lo, hi = np.percentile(data, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        image = np.clip((data - lo) / (hi - lo), 0, 1)
        image = np.ascontiguousarray((255 * image).astype(np.uint8))
        h, w = image.shape
        qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(
            pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Lensless 3D Computational Imaging")
        self.resize(1100, 760)
        self.setMinimumSize(900, 650)

        self.result = None
        self.thread = None
        self.worker = None

        self._build_ui()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_simulation_tab(), "Simulation & Reconstruction")
        tabs.addTab(self._build_about_tab(), "About")
        self.setCentralWidget(tabs)

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(f"{APP_NAME} {APP_VERSION} — Ready")

    def _spin_int(self, value: int, low: int, high: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        return box

    def _spin_float(
        self, value: float, low: float, high: float, decimals: int = 4
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSingleStep((high - low) / 100)
        return box

    def _build_simulation_tab(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)

        controls = QGroupBox("Experiment")
        form = QFormLayout(controls)

        self.grid = self._spin_int(24, 12, 64)
        self.angles = self._spin_int(7, 3, 36)
        self.angle_limit = self._spin_float(20.0, 1.0, 60.0, 1)
        self.wavelength = self._spin_float(532.0, 350.0, 1000.0, 1)
        self.background = self._spin_float(1.33, 1.0, 2.0, 4)
        self.object_ri = self._spin_float(1.40, 1.0, 2.0, 4)
        self.radius = self._spin_float(5.0, 0.5, 15.0, 2)

        form.addRow("Grid (N³)", self.grid)
        form.addRow("Illumination angles", self.angles)
        form.addRow("Angle range ± (°)", self.angle_limit)
        form.addRow("Wavelength (nm)", self.wavelength)
        form.addRow("Background RI", self.background)
        form.addRow("Sphere RI", self.object_ri)
        form.addRow("Sphere radius (µm)", self.radius)

        self.run_button = QPushButton("Run Simulation + Ewald Reconstruction")
        self.run_button.setMinimumHeight(46)
        self.run_button.clicked.connect(self.start_run)
        form.addRow(self.run_button)

        self.export_button = QPushButton("Export Last Result (.npz)")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        form.addRow(self.export_button)

        root.addWidget(controls, 0)

        self.view = SliceView()
        root.addWidget(self.view, 1)

        return page

    def _build_about_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel(
            "<h1>INVISIBLE³D</h1>"
            "<h3>Seeing 3D Matter Without a Lens</h3>"
        )
        description = QLabel(
            "A research-oriented computational imaging platform for "
            "physics-based diffraction simulation and 3D refractive-index "
            "reconstruction. This desktop edition uses the same core engine "
            "as the INVISIBLE³D Python package."
        )
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()
        return page

    def start_run(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            return

        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.statusBar().showMessage("Starting…")

        self.thread = QThread()
        self.worker = SimulationWorker(
            self.grid.value(),
            self.angles.value(),
            self.angle_limit.value(),
            self.wavelength.value(),
            self.background.value(),
            self.object_ri.value(),
            self.radius.value(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.run_finished)
        self.worker.failed.connect(self.run_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread_finished)
        self.thread.start()

    @Slot(object)
    def run_finished(self, result: dict) -> None:
        self.result = result
        self.view.show_array(result["reconstruction"])
        self.export_button.setEnabled(True)
        self.statusBar().showMessage(
            "Completed — central reconstructed RI slice displayed."
        )

    @Slot(str)
    def run_failed(self, error: str) -> None:
        self.statusBar().showMessage("Reconstruction failed")
        QMessageBox.critical(
            self,
            "INVISIBLE³D — Reconstruction Error",
            error[-5000:],
        )

    @Slot()
    def thread_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.thread = None
        self.worker = None

    def export_result(self) -> None:
        if self.result is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export INVISIBLE³D result",
            "invisible3d_result.npz",
            "NumPy archive (*.npz)",
        )
        if not path:
            return

        np.savez_compressed(
            path,
            phantom=self.result["phantom"],
            reconstruction=self.result["reconstruction"],
            scattered_fields=self.result["scattered_fields"],
            angles=self.result["angles"],
            wavelength=self.result["wavelength"],
            pixel_size=self.result["pixel_size"],
        )
        self.statusBar().showMessage(f"Saved: {path}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("INVISIBLE³D Research")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
