import os
import numpy as np
import tifffile
import h5py
from napari.utils.notifications import show_info


class LSFMVolumeWriter:
    def __init__(self):
        pass

    def write_tiff(self, volume, path, metadata=None):
        """Save volume as a TIFF file."""
        try:
            # Ensure the path has the proper extension
            if not path.lower().endswith((".tif", ".tiff")):
                path = f"{path}.tif"
            tifffile.imwrite(
                path,
                np.float32(volume),
                imagej=True,
                metadata=metadata or {"axes": "ZYX"},
            )
            show_info(f"Volume successfully saved as TIFF at: {path}")
            return path
        except OSError as e:  # noqa: BLE001
            show_info(f"Error saving TIFF: {e}")
            raise e

    def write_hdf5(self, volume, path, meta=None):
        """Save volume as an HDF5 file."""
        try:
            # Ensure the path has the proper extension
            if not path.lower().endswith((".h5", ".hdf5")):
                path = f"{path}.h5"
            with h5py.File(path, "w") as f:
                dset = f.create_dataset(
                    "reconstructed_volume",
                    data=np.float32(volume),
                    compression="gzip",
                )
                if meta:
                    for key, value in meta.items():
                        dset.attrs[key] = value
            show_info(f"Volume successfully saved as HDF5 at: {path}")
            return path
        except OSError as e:  # noqa: BLE001
            show_info(f"Error saving HDF5: {e}")
            raise e


def napari_write_image(path, data, meta):
    """Helper function to write image data using LSFMVolumeWriter."""
    writer = LSFMVolumeWriter()
    try:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext in ("tif", "tiff"):
            return writer.write_tiff(data, path, metadata=meta)
        elif ext in ("h5", "hdf5"):
            return writer.write_hdf5(data, path, meta=meta)
        else:
            show_info(f"Unrecognized extension: {ext}. Saving as TIFF.")
            return writer.write_tiff(data, path + ".tif", metadata=meta)
    except Exception as e:  # noqa: BLE001
        show_info(f"Error in napari_write_image: {str(e)}")
        return None


def napari_get_writer(path, layer_types):
    """Return a function capable of writing napari layer data to a path.

    Parameters
    ----------
    path : str
        The file path to write to.
    layer_types : list of str
        The list of layer types (all should be "image").

    Returns
    -------
    callable or None
        A function that accepts a path and a list of layer data tuples.
    """
    if not all(lt == "image" for lt in layer_types):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in ("tif", "tiff", "h5", "hdf5"):
        return None

    def writer_function(path, layer_data):
        """Write multiple layers to the specified path."""
        paths = []
        for data, meta, _ in layer_data:
            try:
                # If writing multiple layers, append a unique name if available.
                if len(layer_data) > 1 and "name" in meta:
                    basename, ext = os.path.splitext(path)
                    new_path = f"{basename}_{meta['name']}{ext}"
                else:
                    new_path = path
                written_path = napari_write_image(new_path, data, meta)
                if written_path:
                    paths.append(written_path)
            except Exception as e:  # noqa: BLE001
                show_info(f"Error writing layer: {str(e)}")
        return paths

    return writer_function
