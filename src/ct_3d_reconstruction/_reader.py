import os

import h5py
import numpy as np
from napari.utils.notifications import show_info
from tifffile import TiffFile


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    This reader supports TIFF and HDF5 files, checks that the data is 3D,
    and transposes the data if the stack dimension is last.
    """
    # If a list of paths is provided, use the first one.
    if isinstance(path, list):
        path = path[0]

    if path.lower().endswith((".tiff", ".tif", ".h5", ".hdf5")):
        return reader_function

    return None


def reader_function(path):
    """Take a path (or list of paths) and return a list of LayerData tuples.

    This function checks if the file contains 3D data.
    For TIFF files, it uses TiffFile to stack multiple pages.
    If the data is in shape (height, width, slices) with a small number
    of slices, it transposes to (slices, height, width) (i.e. "ZYX" order).

    Returns
    -------
    list
        A list containing a single tuple: (data, add_kwargs, layer_type)
    """
    data = None
    metadata = {}

    if path.lower().endswith((".tiff", ".tif")):
        # Use TiffFile to read all pages.
        with TiffFile(path) as tif:
            n_pages = len(tif.pages)
            if n_pages > 1:
                data = np.stack(
                    [page.asarray() for page in tif.pages], axis=-1
                )
            else:
                data = tif.asarray()

        # Check if the data is 3D.
        if data.ndim < 3:
            show_info(
                "The TIFF file appears to be 2D. A 3D rotation stack is required for CT reconstruction."
            )
            raise ValueError("Data must be 3D for CT reconstruction")

        # If the number of slices (last dimension) is smaller than the height,
        # assume the stack is stored in (height, width, slices), and transpose to (slices, height, width).
        if data.shape[-1] < data.shape[0]:
            data = np.transpose(data, (2, 0, 1))

        layer_type = "image"
        metadata["axes"] = "ZYX"

    elif path.lower().endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as f:
            dataset_names = list(f.keys())
            if len(dataset_names) == 0:
                raise ValueError("HDF5 file does not contain any datasets.")

            # Look for common dataset names.
            for name in [
                "recon_vol_lsfm",
                "volume",
                "frames",
                "data",
                "stack",
            ]:
                if name in dataset_names:
                    data = np.array(f[name])
                    metadata = dict(f[name].attrs)
                    break

            if data is None:
                # Fallback to the first dataset.
                data = np.array(f[dataset_names[0]])
                metadata = dict(f[dataset_names[0]].attrs)
                show_info(f"Loaded dataset: {dataset_names[0]}")

        if data.ndim < 3:
            show_info(
                "The HDF5 dataset is 2D. A 3D rotation stack is required for CT reconstruction."
            )
            raise ValueError("Loaded HDF5 dataset must be 3D")

        # If the dataset shape seems to be (height, width, slices), transpose it.
        if data.shape[-1] < data.shape[0]:
            data = np.transpose(data, (2, 0, 1))

        layer_type = "image"
        # Set a default for axes if not present.
        if "axes" not in metadata:
            metadata["axes"] = "ZYX"
    else:
        raise ValueError(
            "Unsupported file type. Please use .tiff, .tif, .h5, or .hdf5 files."
        )

    add_kwargs = {"name": os.path.basename(path)}
    if "spacing" in metadata:
        add_kwargs["scale"] = metadata["spacing"]
    if "axes" in metadata:
        add_kwargs["metadata"] = {"axes": metadata["axes"]}

    return [(data, add_kwargs, layer_type)]
