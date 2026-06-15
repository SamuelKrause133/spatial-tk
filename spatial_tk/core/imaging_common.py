"""
Shared helpers for microscopy image commands (SpatialData I/O without requiring a gene table).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import spatialdata as sd

from spatial_tk.core.data_io import save_spatial_data


def read_sdata(path: Path) -> sd.SpatialData:
    """Load SpatialData from zarr; does not require an AnnData table."""
    if not path.exists():
        raise FileNotFoundError(path)
    return sd.read_zarr(path)


def first_image_key(sdata: sd.SpatialData) -> str:
    if not getattr(sdata, "images", None):
        raise ValueError("SpatialData has no images")
    keys = list(sdata.images.keys())
    if not keys:
        raise ValueError("SpatialData.images is empty")
    return keys[0]


def image_to_numpy_cyx(image_el: Any) -> np.ndarray:
    """Return a (C, Y, X) float32 array from a SpatialData spatial image element."""
    data = image_el.data if hasattr(image_el, "data") else image_el
    if hasattr(data, "compute"):
        data = data.compute()
    arr = np.asarray(data)
    # xarray-like (c, y, x) or (y, x, c)
    if arr.ndim == 2:
        return arr[np.newaxis, ...].astype(np.float32)
    if arr.ndim == 3:
        # Heuristic: small first dim = channels
        if arr.shape[0] <= 32 and arr.shape[0] < arr.shape[-1]:
            return arr.astype(np.float32)
        if arr.shape[-1] <= 32:
            return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    raise ValueError(f"Unsupported image array shape: {arr.shape}")


def labels_to_numpy(instance_el: Any) -> np.ndarray:
    data = instance_el.data if hasattr(instance_el, "data") else instance_el
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data, dtype=np.int32)


def ensure_obs_from_labels(
    sdata: sd.SpatialData,
    labels_key: str,
    label_array: np.ndarray,
    table_name: str = "table",
) -> None:
    """
    Build a minimal AnnData observation table (one row per non-zero label) with
    centroid coordinates for downstream quantify/extract commands.
    """
    import anndata as ad
    from scipy import ndimage as ndi

    from spatial_tk.utils.helpers import get_table

    ids = np.unique(label_array)
    ids = ids[ids > 0]
    xs = np.empty(len(ids), dtype=np.float64)
    ys = np.empty(len(ids), dtype=np.float64)
    for i, lid in enumerate(ids):
        m = label_array == lid
        ys[i], xs[i] = ndi.center_of_mass(m)

    adata = ad.AnnData(X=np.zeros((len(ids), 1), dtype=np.float32))
    adata.obs["label_id"] = ids.astype(np.int64)
    adata.obs["centroid_y"] = ys
    adata.obs["centroid_x"] = xs
    adata.obs["region"] = labels_key
    adata.obs["instance_id"] = np.arange(len(ids), dtype=np.int64).astype("category")

    if get_table(sdata) is not None and table_name in getattr(sdata, "tables", {}):
        logging.warning("Replacing existing table %s", table_name)
    if hasattr(sdata, "tables"):
        sdata.tables[table_name] = adata
    else:
        sdata.table = adata


def save_sdata(sdata: sd.SpatialData, path: Path, overwrite: bool = False) -> None:
    save_spatial_data(sdata, path, overwrite=overwrite)


def write_montage_png(chips: np.ndarray, path: Path, n: int = 12) -> None:
    """
    Write a single PNG with up to `n` chips tiled (chips shaped (N, H, W) or (N, H, W, C)).
    """
    import matplotlib.pyplot as plt

    n = min(n, chips.shape[0])
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol)
        ax = axes[r, c]
        if i < n:
            chip = chips[i]
            if chip.ndim == 3 and chip.shape[-1] in (1, 3, 4):
                ax.imshow(np.clip(chip, 0, 1))
            else:
                ax.imshow(chip, cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def z_project(stack_zyx: np.ndarray, mode: str = "max") -> np.ndarray:
    """Project a (Z, Y, X) stack to 2D."""
    if stack_zyx.ndim != 3:
        raise ValueError(f"Expected ZYX stack, got shape {stack_zyx.shape}")
    if mode == "max":
        return np.max(stack_zyx, axis=0)
    if mode == "middle":
        z = stack_zyx.shape[0] // 2
        return stack_zyx[z]
    raise ValueError(f"Unknown projection mode: {mode}")
