import os
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
from napari.utils.notifications import show_info
from tifffile import TiffFile

# Preferred HDF5 dataset names, in order of priority
_PREFERRED_DATASETS = [
    "recon_vol_lsfm",
    "volume",
    "frames",
    "data",
    "stack",
]


def napari_get_reader(path: Union[str, List[str]]):
    """A basic implementation of a Reader contribution."""
    if isinstance(path, list):
        path = path[0]
    if path.lower().endswith((".tiff", ".tif", ".h5", ".hdf5")):
        return reader_function
    return None


def _check_3d(data: np.ndarray, fmt: str) -> None:
    if data.ndim < 3:
        show_info(f"The {fmt} file is 2D. A 3D rotation stack is required.")
        raise ValueError("Data must be 3D for CT reconstruction")


def _transpose_if_needed(data: np.ndarray) -> np.ndarray:
    # If slices are in last dim but fewer than height, assume (H, W, Z) → (Z, H, W)
    return data.transpose(2, 0, 1) if data.shape[-1] < data.shape[0] else data


def _read_tiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with TiffFile(path) as tif:
        pages = tif.pages
        data = (
            np.stack([p.asarray() for p in pages], axis=-1)
            if len(pages) > 1
            else tif.asarray()
        )
    _check_3d(data, "TIFF")
    data = _transpose_if_needed(data)
    return data, {"axes": "ZYX"}


def _read_hdf5(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with h5py.File(path, "r") as f:
        names = list(f.keys())
        if not names:
            raise ValueError("HDF5 file does not contain any datasets.")
        for name in _PREFERRED_DATASETS:
            if name in names:
                dataset_name = name
                break
        else:
            dataset_name = names[0]
            show_info(f"Loaded dataset: {dataset_name}")
        ds = f[dataset_name]
        data = np.array(ds)
        metadata = dict(ds.attrs)
    _check_3d(data, "HDF5")
    data = _transpose_if_needed(data)
    metadata.setdefault("axes", "ZYX")
    return data, metadata


def reader_function(path: Union[str, List[str]]):
    """Read a TIFF or HDF5 file into a Napari layer."""
    if isinstance(path, list):
        path = path[0]

    ext = path.lower()
    if ext.endswith((".tiff", ".tif")):
        data, metadata = _read_tiff(path)
    elif ext.endswith((".h5", ".hdf5")):
        data, metadata = _read_hdf5(path)
    else:
        raise ValueError("Unsupported file type. Use .tiff, .tif, .h5, or .hdf5.")

    add_kwargs: Dict[str, Any] = {"name": os.path.basename(path)}
    if "spacing" in metadata:
        add_kwargs["scale"] = metadata["spacing"]
    if "axes" in metadata:
        add_kwargs["metadata"] = {"axes": metadata["axes"]}

    return [(data, add_kwargs, "image")]
