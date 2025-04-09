import concurrent.futures
import os

import napari
import numpy as np
import scipy.ndimage
from napari.layers import Image
from napari.utils.notifications import show_info

# Import the hook decorator from napari_plugin_engine (npe2)
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._reader import napari_get_reader
from ._writer import LSFMVolumeWriter


# --------------------------
# Helper function for parallel processing.
# Processes one projection:
#  - Extract a slice from the input data (assumed shape: (num_slices, height, width))
#  - Downscale the slice using the provided downsample factor.
#  - Crop the downscaled slice to exactly (new_height, new_width)
#  - Insert the slice into a temporary volume at the computed center slice.
#  - Rotate that temporary volume by the computed angle.
#  - Build a corresponding normalization mask.
# --------------------------
def process_slice(args):
    (i, angle, data, angle_step, center_slice, out_shape, ds) = args
    # Determine the slice index.
    idx = min(i * angle_step, data.shape[0] - 1)
    slice_img = data[idx, :, :]
    # Downscale using scipy.ndimage.zoom (bilinear interpolation, order=1).
    slice_img_ds = scipy.ndimage.zoom(slice_img, zoom=(ds, ds), order=1)
    # Force the downscaled slice to have exactly the expected dimensions.
    new_height, new_width, _ = out_shape
    slice_img_ds = slice_img_ds[:new_height, :new_width]

    # Allocate a temporary volume and insert the slice into the computed center.
    vol_tmp = np.zeros(out_shape, dtype=np.float32)
    vol_tmp[:, :, center_slice] = slice_img_ds

    # Rotate the temporary volume by the computed angle.
    vol_tmp = scipy.ndimage.rotate(
        vol_tmp, angle, mode="nearest", axes=(0, 2), reshape=False
    )

    # Create the corresponding normalization volume.
    norm_tmp = np.zeros(out_shape, dtype=np.float32)
    norm_tmp[:, :, center_slice] = np.ones(
        (new_height, new_width), dtype=np.float32
    )
    norm_tmp = scipy.ndimage.rotate(
        norm_tmp, angle, mode="nearest", axes=(0, 2), reshape=False
    )

    return vol_tmp, norm_tmp


# --------------------------
# Worker class that performs CT reconstruction using parallel processing.
# --------------------------
class ReconstructionWorker(QWidget):
    progress_updated = Signal(int)
    reconstruction_complete = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.writer = LSFMVolumeWriter()

    def reconstruct_volume(
        self, data, angle_step=1, center_offset=0, downsample=1.0
    ):
        """
        Reconstruct a 3D volume from a rotation stack using parallel processing and downsampling.

        Parameters
        ----------
        data : numpy.ndarray
            3D rotation stack in ZYX order (shape: (num_slices, height, width)).
        angle_step : int or float
            Process every 'angle_step'-th slice; also used as the per-slice angle increment.
        center_offset : int
            Offset (in pixels) for the center slice (on the downscaled dimensions).
        downsample : float, optional
            Factor to downscale the images (from 0.1 to 1.0; default 1.0 means full resolution).

        Returns
        -------
        numpy.ndarray
            The reconstructed 3D volume.
        """
        num_slices, height, width = data.shape

        ds = downsample
        new_height = int(height * ds)
        new_width = int(width * ds)
        # The current algorithm produces a volume with shape (new_height, new_width, new_height).
        out_shape = (new_height, new_width, new_height)

        # Compute the center slice of the output volume.
        center_slice = new_height // 2 + int(center_offset * ds)

        # Initialize accumulator arrays.
        myvolume = np.zeros(out_shape, dtype=np.float32)
        myvolume_norm = np.zeros(out_shape, dtype=np.float32)

        # Select every 'angle_step'-th slice.
        indices = list(range(0, num_slices, int(angle_step)))
        total_tasks = len(indices)
        tasks = [
            (
                i,
                indices[i] * angle_step,
                data,
                angle_step,
                center_slice,
                out_shape,
                ds,
            )
            for i in range(total_tasks)
        ]

        # Process tasks in parallel.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_slice, tasks))

        # Accumulate the results.
        for i, (vol_tmp, norm_tmp) in enumerate(results):
            myvolume += vol_tmp
            myvolume_norm += norm_tmp
            progress = int(((i + 1) / total_tasks) * 100)
            self.progress_updated.emit(progress)

        # Avoid division by zero.
        myvolume_norm[myvolume_norm == 0] = 1

        # Normalize the volume to a [0, 1] range.
        recon_vol = myvolume / myvolume_norm
        recon_vol -= np.min(recon_vol)
        if np.max(recon_vol) > 0:
            recon_vol /= np.max(recon_vol)

        self.progress_updated.emit(100)
        self.reconstruction_complete.emit(recon_vol)
        return recon_vol

    def save_volume(self, volume, path, format_name, meta=None):
        try:
            if format_name.lower() in ["tif", "tiff"]:
                self.writer.write_tiff(volume, path, metadata=meta)
            elif format_name.lower() in ["h5", "hdf5"]:
                self.writer.write_hdf5(volume, path, meta=meta)
            else:
                show_info(
                    f"Unsupported format: {format_name}. Saving as TIFF."
                )
                self.writer.write_tiff(volume, path, metadata=meta)
            show_info(f"Volume successfully saved to {path}")
        except Exception as e:  # noqa: BLE001
            show_info(f"Error saving volume: {str(e)}")


# --------------------------
# Main widget class for the CT 3D Reconstruction plugin.
# --------------------------
class LSFMReconstructionWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.reconstructed_volume = None
        self.worker = ReconstructionWorker()

        self.worker.progress_updated.connect(self.update_progress)
        self.worker.reconstruction_complete.connect(
            self.on_reconstruction_complete
        )

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Load file section.
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout(load_group)
        load_button = QPushButton("Load CT File")
        load_button.clicked.connect(self.load_file)
        load_layout.addWidget(load_button)
        layout.addWidget(load_group)

        # Input selection section.
        input_group = QGroupBox("Input Data")
        input_layout = QFormLayout(input_group)
        self.layer_combo = QComboBox()
        self.update_layer_list()
        self.viewer.layers.events.inserted.connect(self.update_layer_list)
        self.viewer.layers.events.removed.connect(self.update_layer_list)
        input_layout.addRow("Rotation Stack:", self.layer_combo)
        layout.addWidget(input_group)

        # Reconstruction parameters section.
        param_group = QGroupBox("Reconstruction Parameters")
        param_layout = QFormLayout(param_group)
        self.angle_step = QSpinBox()
        self.angle_step.setRange(1, 180)
        self.angle_step.setValue(1)
        param_layout.addRow("Angle Step (°):", self.angle_step)
        self.center_offset = QSpinBox()
        self.center_offset.setRange(-100, 100)
        self.center_offset.setValue(0)
        param_layout.addRow("Center Offset (pixels):", self.center_offset)
        self.voxel_size = QDoubleSpinBox()
        self.voxel_size.setRange(0.01, 100.0)
        self.voxel_size.setValue(1.0)
        self.voxel_size.setSuffix(" µm")
        self.voxel_size.setDecimals(2)
        param_layout.addRow("Voxel Size:", self.voxel_size)
        self.downsample_factor = QDoubleSpinBox()
        self.downsample_factor.setRange(0.1, 1.0)
        self.downsample_factor.setValue(1.0)
        self.downsample_factor.setSingleStep(0.1)
        self.downsample_factor.setDecimals(2)
        param_layout.addRow("Downsample Factor:", self.downsample_factor)
        layout.addWidget(param_group)

        # Progress section.
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)

        # Action buttons.
        button_layout = QHBoxLayout()
        self.reconstruct_button = QPushButton("Reconstruct Volume")
        self.reconstruct_button.clicked.connect(self.start_reconstruction)
        self.save_button = QPushButton("Save Volume")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_reconstructed_volume)
        button_layout.addWidget(self.reconstruct_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CT File",
            "",
            "CT Files (*.tif *.tiff *.h5 *.hdf5);;All Files (*)",
        )
        if not file_path:
            show_info("No file selected")
            return

        reader = napari_get_reader(file_path)
        if reader is None:
            show_info("File type not supported by the CT reader")
            return

        try:
            layer_data = reader(file_path)
            for data, add_kwargs, _ in layer_data:
                self.viewer.add_image(data, **add_kwargs)
            show_info(f"Loaded file: {os.path.basename(file_path)}")
        except Exception as e:  # noqa: BLE001
            show_info(f"Error loading file: {str(e)}")

    def update_layer_list(self, event=None):
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.data.ndim >= 3:
                self.layer_combo.addItem(layer.name)
        if hasattr(self, "reconstruct_button"):
            self.reconstruct_button.setEnabled(self.layer_combo.count() > 0)

    def get_selected_layer_data(self):
        layer_name = self.layer_combo.currentText()
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer.data, layer
        return None, None

    def start_reconstruction(self):
        data, _ = self.get_selected_layer_data()
        if data is None:
            show_info("Please select a valid rotation stack layer")
            return
        self.reconstruct_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.reconstructed_volume = self.worker.reconstruct_volume(
            data,
            angle_step=self.angle_step.value(),
            center_offset=self.center_offset.value(),
            downsample=self.downsample_factor.value(),
        )

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    @Slot(object)
    def on_reconstruction_complete(self, volume):
        self.reconstructed_volume = volume
        voxel_size = self.voxel_size.value()
        scale = [voxel_size, voxel_size, voxel_size]
        self.viewer.add_image(
            volume,
            name="Reconstructed Volume",
            scale=scale,
            colormap="viridis",
        )
        self.reconstruct_button.setEnabled(True)
        self.save_button.setEnabled(True)
        show_info("Reconstruction complete!")

    def save_reconstructed_volume(self):
        if self.reconstructed_volume is None:
            show_info("No reconstructed volume available")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Reconstructed Volume",
            "",
            "TIFF Files (*.tif *.tiff);;HDF5 Files (*.hdf5 *.h5);;All Files (*)",
        )

        if not path:
            show_info("Save cancelled")
            return

        if not path.lower().endswith((".tif", ".tiff", ".h5", ".hdf5")):
            if "TIFF" in selected_filter:
                path += ".tif"
                format_name = "tiff"
            elif "HDF5" in selected_filter:
                path += ".h5"
                format_name = "h5"
            else:
                path += ".tif"
                format_name = "tiff"
        else:
            ext = os.path.splitext(path)[1].lower()
            format_name = "tiff" if ext in (".tif", ".tiff") else "h5"

        meta = {
            "spacing": self.voxel_size.value(),
            "unit": "um",
            "axes": "ZYX",
            "description": "Reconstructed from CT rotation stack",
        }
        self.worker.save_volume(
            self.reconstructed_volume, path, format_name, meta
        )


@napari_hook_implementation
def napari_experimental_provide_dock_widget(napari_viewer=None):
    if napari_viewer is None:
        napari_viewer = napari.current_viewer()
    return LSFMReconstructionWidget(napari_viewer)
