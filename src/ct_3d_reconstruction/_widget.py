# widget.py
import concurrent.futures

import napari
import numpy as np
import scipy.ndimage as ndi
from napari.layers import Image
from napari.utils.notifications import show_info
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.transform import iradon

from ._reader import napari_get_reader
from ._writer import LSFMVolumeWriter


def _estimate_center(sino_2d, max_fraction=0.25):
    """
    Estimate horizontal center shift by minimizing left-right asymmetry.
    """
    sino_1d = np.mean(sino_2d, axis=0)
    w = sino_1d.size
    mid = w // 2
    max_shift = int(w * max_fraction)
    best_shift = 0
    best_err = np.inf
    for shift in range(-max_shift, max_shift + 1):
        rolled = np.roll(sino_1d, shift)
        left = rolled[:mid]
        right = rolled[-mid:][::-1]
        err = np.sum(np.abs(left - right))
        if err < best_err:
            best_err = err
            best_shift = shift
    return best_shift


class ReconstructionWorker(QWidget):
    progress_updated = Signal(int)
    reconstruction_complete = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.writer = LSFMVolumeWriter()

    def reconstruct_volume(
        self,
        data,
        angle_step,
        downsample,
        center_offset,
        auto_center,
        filter_name,
        circle,
        log_normalize,
        pre_filter,
        post_filter,
        window_level,
        ring_remove,
        angle_averaging=1,  # New parameter for angle averaging
        stronger_post_filter=False,  # New parameter for stronger post-filtering
    ):
        # data: (n_proj, H, W)
        n_proj, h_orig, w_orig = data.shape

        # 1) optional log‐normalize
        if log_normalize:
            p = data.astype(np.float32) + 1e-6
            data = -np.log(p / np.max(p))

        # 2) downsample Y/X
        data_f = data.astype(np.float32)
        if downsample != 1.0:
            data_ds = ndi.zoom(data_f, (1, downsample, downsample), order=1)
        else:
            data_ds = data_f
        _, h_ds, w_ds = data_ds.shape

        # 3) angles array
        theta = np.arange(n_proj) * angle_step

        # 4) auto-center sinograms
        if auto_center:
            sino_avg = np.mean(data_ds, axis=1)
            best = _estimate_center(sino_avg)
            center_offset = best
            show_info(f"Auto-center shift = {best} px")

        # 5) allocate reconstruction buffer [Y, X, X]
        recon = np.zeros((h_ds, w_ds, w_ds), dtype=np.float32)

        def _reconstruct_row(y):
            sino = data_ds[:, y, :]

            # ring artifact removal (subtract column median)
            if ring_remove:
                sino = sino - np.median(sino, axis=0)[None, :]

            # Apply angle averaging if enabled (reduces noise by averaging neighboring projections)
            if angle_averaging > 1:
                sino_avg = np.zeros_like(sino)
                for i in range(sino.shape[0]):
                    # Calculate indices for averaging window, handling wrap-around
                    avg_indices = [
                        (i + j) % sino.shape[0]
                        for j in range(
                            -angle_averaging // 2 + 1, angle_averaging // 2 + 1
                        )
                    ]
                    # Average the projections
                    sino_avg[i] = np.mean(sino[avg_indices], axis=0)
                sino = sino_avg

            # pre-filter sinogram
            if pre_filter:
                # Use larger kernel for stronger noise reduction
                sino = ndi.median_filter(sino, size=(1, 5))

            # apply center offset
            if center_offset:
                sino = np.roll(sino, int(center_offset), axis=1)

            # transpose for iradon: (detector, angles)
            sino = sino.T

            # FBP reconstruction
            rec = iradon(
                sino,
                theta=theta,
                filter_name=filter_name,
                circle=circle,
                output_size=w_ds,
            ).astype(np.float32)
            return y, rec

        # 6) parallel back-projection
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for y, rec in executor.map(_reconstruct_row, range(h_ds)):
                recon[y] = rec
                pct = int((y + 1) / h_ds * 100)
                self.progress_updated.emit(pct)

        # 7) reorder to (Z, Y, X)
        vol = recon.transpose(2, 0, 1)

        # 8) post‐filter volume
        if post_filter:
            if stronger_post_filter:
                # Apply stronger 3D filtering
                vol = ndi.median_filter(vol, size=(5, 5, 5))
                # Additional Gaussian smoothing to reduce noise
                vol = ndi.gaussian_filter(vol, sigma=1.0)
            else:
                vol = ndi.median_filter(vol, size=(3, 3, 3))

        # 9) window‐level normalization
        if window_level:
            p1, p99 = np.percentile(vol, (1, 99))
            vol = np.clip(vol, p1, p99)
            vol = (vol - p1) / (p99 - p1)

        self.progress_updated.emit(100)
        self.reconstruction_complete.emit(vol)
        return vol

    def save_volume(self, volume, path, format_name, meta=None):
        try:
            if format_name.lower() in ("tif", "tiff"):
                self.writer.write_tiff(volume, path, metadata=meta)
            else:
                self.writer.write_hdf5(volume, path, meta=meta)
            show_info(f"Volume saved to {path}")
        except Exception as e:
            show_info(f"Error saving: {e}")


class LSFMReconstructionWidget(QWidget):
    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer or napari.current_viewer()
        self.worker = ReconstructionWorker()
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.reconstruction_complete.connect(self._on_complete)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Load Data ---
        load_g = QGroupBox("Load Data")
        load_l = QVBoxLayout(load_g)
        btn = QPushButton("Load CT File")
        btn.clicked.connect(self._load)
        load_l.addWidget(btn)
        layout.addWidget(load_g)

        # --- Input Data ---
        in_g = QGroupBox("Input Data")
        in_f = QFormLayout(in_g)
        self.layer_cb = QComboBox()
        self._refresh_layers()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        in_f.addRow("Rotation Stack:", self.layer_cb)
        layout.addWidget(in_g)

        # --- Reconstruction Parameters ---
        p_g = QGroupBox("Reconstruction Parameters")
        p_f = QFormLayout(p_g)
        self.angle_sb = QDoubleSpinBox()
        self.angle_sb.setRange(0.01, 360)
        self.angle_sb.setValue(3.6)
        p_f.addRow("Angle Step (°):", self.angle_sb)
        self.down_sb = QDoubleSpinBox()
        self.down_sb.setRange(0.1, 1.0)
        self.down_sb.setValue(0.5)
        p_f.addRow("Downsample:", self.down_sb)
        self.center_sb = QDoubleSpinBox()
        self.center_sb.setRange(-1e6, 1e6)
        self.center_sb.setValue(0)
        p_f.addRow("Center Offset:", self.center_sb)
        self.auto_center = QCheckBox("Auto-center")
        p_f.addRow("", self.auto_center)

        # New angle averaging parameter
        self.avg_sb = QDoubleSpinBox()
        self.avg_sb.setRange(1, 9)
        self.avg_sb.setValue(1)
        self.avg_sb.setSingleStep(2)
        p_f.addRow("Angle Averaging:", self.avg_sb)

        self.filter_cb = QComboBox()
        for f in ("ramp", "shepp-logan", "cosine", "hamming", "hann"):
            self.filter_cb.addItem(f)
        p_f.addRow("Filter:", self.filter_cb)
        self.circle = QCheckBox("Crop circle")
        p_f.addRow("", self.circle)
        self.log_norm = QCheckBox("Log-normalize")
        p_f.addRow("", self.log_norm)
        self.ring = QCheckBox("Ring artifact removal")
        p_f.addRow("", self.ring)
        self.pre = QCheckBox("Pre-filter sinogram")
        p_f.addRow("", self.pre)
        self.post = QCheckBox("Post-filter volume")
        p_f.addRow("", self.post)

        # New stronger post-filter option
        self.strong_post = QCheckBox("Strong post-filter")
        p_f.addRow("", self.strong_post)

        self.window = QCheckBox("Window-level 1–99%")
        p_f.addRow("", self.window)
        self.voxel_sb = QDoubleSpinBox()
        self.voxel_sb.setRange(0.01, 1000)
        self.voxel_sb.setValue(1.0)
        self.voxel_sb.setSuffix(" µm")
        p_f.addRow("Voxel Size:", self.voxel_sb)
        layout.addWidget(p_g)

        # --- Progress & Actions ---
        prog_g = QGroupBox("Progress")
        prog_l = QVBoxLayout(prog_g)
        self.pb = QProgressBar()
        prog_l.addWidget(self.pb)
        layout.addWidget(prog_g)

        h = QHBoxLayout()
        self.run_btn = QPushButton("Reconstruct")
        self.run_btn.clicked.connect(self._start)
        h.addWidget(self.run_btn)
        self.save_btn = QPushButton("Save Volume")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        h.addWidget(self.save_btn)
        layout.addLayout(h)

    def _refresh_layers(self, *a):
        self.layer_cb.clear()
        for lyr in self.viewer.layers:
            if isinstance(lyr, Image) and lyr.data.ndim == 3:
                self.layer_cb.addItem(lyr.name)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CT Stack", "", "*.tif *.tiff *.h5 *.hdf5"
        )
        if not path:
            return
        reader = napari_get_reader(path)
        if reader is None:
            show_info("Unsupported file type")
            return
        for data, kw, _ in reader(path):
            self.viewer.add_image(data, **kw)

    def _start(self):
        nm = self.layer_cb.currentText()
        lyr = next((layer for layer in self.viewer.layers if layer.name == nm), None)
        if lyr is None:
            show_info("Select a rotation stack")
            return
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.pb.setValue(0)
        self.worker.reconstruct_volume(
            lyr.data,
            angle_step=self.angle_sb.value(),
            downsample=self.down_sb.value(),
            center_offset=self.center_sb.value(),
            auto_center=self.auto_center.isChecked(),
            filter_name=self.filter_cb.currentText(),
            circle=self.circle.isChecked(),
            log_normalize=self.log_norm.isChecked(),
            pre_filter=self.pre.isChecked(),
            post_filter=self.post.isChecked(),
            window_level=self.window.isChecked(),
            ring_remove=self.ring.isChecked(),
            angle_averaging=int(self.avg_sb.value()),  # New parameter
            stronger_post_filter=self.strong_post.isChecked(),  # New parameter
        )

    @Slot(int)
    def _on_progress(self, v):
        self.pb.setValue(v)

    @Slot(object)
    def _on_complete(self, vol):
        # add reconstructed volume
        self.viewer.dims.ndisplay = 3
        layer = self.viewer.add_image(
            vol,
            name="Reconstructed Volume",
            scale=[self.voxel_sb.value()] * 3,
            colormap="gist_earth",
        )
        layer.rendering = "mip"
        layer.interpolation = "linear"

        # Reset view to properly center on the volume
        self.viewer.reset_view()

        # Make sure the camera is centered on the volume
        # Get the midpoint of the volume
        center_z = vol.shape[0] // 2
        center_y = vol.shape[1] // 2
        center_x = vol.shape[2] // 2

        # Set the camera center in data coordinates
        self.viewer.camera.center = (
            center_z * self.voxel_sb.value(),
            center_y * self.voxel_sb.value(),
            center_x * self.voxel_sb.value(),
        )

        # Adjust the camera zoom to fit the volume nicely
        max_dim = max(vol.shape)
        self.viewer.camera.zoom = (
            0.8 * 512 / max_dim
        )  # Scale factor for reasonable view

        # Make sure we're in 3D view mode
        self.viewer.dims.ndisplay = 3

        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        show_info("Reconstruction complete!")

    def _save(self):
        path, filt = QFileDialog.getSaveFileName(
            self, "Save Volume", "", "TIFF (*.tif *.tiff);;HDF5 (*.h5 *.hdf5)"
        )
        if not path:
            return
        fmt = "tiff" if filt.startswith("TIF") else "h5"
        vol = next(layer for layer in self.viewer.layers if layer.name == "Reconstructed Volume").data
        self.worker.save_volume(
            vol,
            path,
            fmt,
            meta={"spacing": self.voxel_sb.value(), "unit": "um", "axes": "ZYX"},
        )


@napari_hook_implementation
def napari_experimental_provide_dock_widget(napari_viewer=None):
    return LSFMReconstructionWidget(napari_viewer)
