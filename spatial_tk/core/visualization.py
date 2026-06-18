#!/usr/bin/env python3
"""
Visualization utilities for full-slide and ROI rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import tomllib
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

# Only force the headless Agg backend when no interactive backend has been
# configured (e.g. by Jupyter's ``%matplotlib inline``). This keeps CLI and
# test usage headless while letting notebooks render figures inline.
if not os.environ.get("MPLBACKEND") and matplotlib.get_backend().lower() == "agg":
    matplotlib.use("Agg")
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ROI:
    """Container for a rectangular region of interest."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "source": self.source,
        }


@dataclass
class ImageOverlay:
    """Numeric image data plus coordinate extent for plotting."""

    data: np.ndarray
    extent: tuple[float, float, float, float]


@dataclass
class RoiPlotResult:
    """A rendered ROI plus its live Matplotlib figure and axes."""

    roi: ROI
    fig: Any
    ax: Any


def load_visualization_spec(spec_path: Optional[str]) -> Dict[str, Any]:
    """Load supplemental visualization TOML spec."""
    if not spec_path:
        return {}
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Visualization spec file not found: {path}")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def parse_roi_string(roi: str, name: str, source: str = "manual") -> ROI:
    """Parse ROI bbox string in xmin,ymin,xmax,ymax format."""
    parts = [p.strip() for p in roi.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid ROI '{roi}'. Expected xmin,ymin,xmax,ymax.")
    xmin, ymin, xmax, ymax = [float(v) for v in parts]
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid ROI '{roi}'. xmax/ymax must exceed xmin/ymin.")
    return ROI(name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, source=source)


def load_rois_from_csv(roi_file: str) -> List[ROI]:
    """Load ROI definitions from CSV with xmin,ymin,xmax,ymax columns."""
    path = Path(roi_file)
    if not path.exists():
        raise FileNotFoundError(f"ROI CSV file not found: {path}")
    frame = pd.read_csv(path)
    required = {"xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(frame.columns):
        raise ValueError(f"ROI CSV must contain columns: {sorted(required)}")

    rois: List[ROI] = []
    for idx, row in frame.iterrows():
        roi_name = str(row["name"]) if "name" in frame.columns and pd.notna(row["name"]) else f"roi_{idx + 1:03d}"
        rois.append(
            ROI(
                name=roi_name,
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
                source="roi_file",
            )
        )
    return rois


def generate_random_rois(
    coords: np.ndarray,
    n_rois: int,
    width: float,
    height: float,
    random_state: int = 0,
) -> List[ROI]:
    """Generate random fixed-size ROI windows within observed coordinate bounds."""
    if n_rois <= 0:
        return []
    if width <= 0 or height <= 0:
        raise ValueError("--roi-width and --roi-height must be > 0 for random ROI generation")
    if coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have at least two dimensions")

    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
    if (x_max - x_min) < width or (y_max - y_min) < height:
        raise ValueError("ROI width/height exceed available coordinate extent")

    rng = np.random.default_rng(random_state)
    rois: List[ROI] = []
    for i in range(n_rois):
        xmin = float(rng.uniform(x_min, x_max - width))
        ymin = float(rng.uniform(y_min, y_max - height))
        rois.append(
            ROI(
                name=f"roi_{i + 1:03d}",
                xmin=xmin,
                ymin=ymin,
                xmax=xmin + width,
                ymax=ymin + height,
                source="random",
            )
        )
    return rois


def resolve_rois(
    coords: np.ndarray,
    view: str,
    roi_strings: Optional[Iterable[str]] = None,
    roi_file: Optional[str] = None,
    random_rois: int = 0,
    roi_width: Optional[float] = None,
    roi_height: Optional[float] = None,
    random_state: int = 0,
) -> List[ROI]:
    """Resolve ROI list from CLI inputs."""
    if view == "full":
        return [ROI("full_slide", float(coords[:, 0].min()), float(coords[:, 1].min()), float(coords[:, 0].max()), float(coords[:, 1].max()), "full")]

    rois: List[ROI] = []
    for i, roi_str in enumerate(roi_strings or []):
        rois.append(parse_roi_string(roi_str, name=f"roi_{i + 1:03d}", source="manual"))

    if roi_file:
        rois.extend(load_rois_from_csv(roi_file))

    if random_rois:
        if roi_width is None or roi_height is None:
            raise ValueError("--roi-width and --roi-height are required when --random-rois > 0")
        rois.extend(
            generate_random_rois(
                coords=coords,
                n_rois=random_rois,
                width=float(roi_width),
                height=float(roi_height),
                random_state=random_state,
            )
        )
    if not rois:
        raise ValueError("ROI view selected but no ROI source provided. Use --roi, --roi-file, or --random-rois.")
    return rois


def _evaluate_where(obs: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate boolean expression against obs dataframe."""
    try:
        out = obs.eval(expression)
    except Exception:
        try:
            out = pd.Series(False, index=obs.index)
            out.loc[obs.query(expression).index] = True
        except Exception as exc:
            raise ValueError(f"Invalid rule where expression '{expression}': {exc}") from exc
    if not pd.api.types.is_bool_dtype(out):
        raise ValueError(f"Rule where expression did not return booleans: {expression}")
    return out.reindex(obs.index).fillna(False).astype(bool)


def compile_style_arrays(
    obs: pd.DataFrame,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile rule-based style arrays for each observation."""
    n = len(obs)
    points = spec.get("points", {})
    styles: Dict[str, Any] = {
        "color": np.array([points.get("default_color", "#808080")] * n, dtype=object),
        "marker": np.array([points.get("default_marker", "o")] * n, dtype=object),
        "size": np.array([float(points.get("default_size", 6.0))] * n, dtype=float),
        "alpha": np.array([float(points.get("alpha", 0.85))] * n, dtype=float),
        "linewidth": np.array([float(points.get("default_linewidth", 0.0))] * n, dtype=float),
        "edgecolor": np.array([points.get("default_edgecolor", "none")] * n, dtype=object),
        "zorder": np.array([float(points.get("default_zorder", 2))] * n, dtype=float),
    }

    continuous_color: Optional[Dict[str, Any]] = None
    for rule in spec.get("rules", []):
        mask = np.ones(n, dtype=bool)
        where = rule.get("where")
        if where:
            mask = _evaluate_where(obs, where).to_numpy()

        kind = rule.get("kind")
        values_map = rule.get("values", {})

        if kind == "categorical":
            for attr, column_key in (("color", "color_by"), ("marker", "marker_by"), ("alpha", "alpha_by"), ("size", "size_by")):
                col = rule.get(column_key)
                if not col:
                    continue
                if col not in obs.columns:
                    raise ValueError(f"Rule references missing obs column '{col}'")
                mapped = obs.loc[mask, col].astype(str).map(values_map)
                keep = mapped.notna().to_numpy()
                target_idx = np.where(mask)[0][keep]
                styles[attr][target_idx] = mapped[keep].to_numpy()
        elif kind == "continuous":
            col = rule.get("color_by")
            if not col:
                raise ValueError("Continuous rules require color_by")
            if col not in obs.columns:
                raise ValueError(f"Rule references missing obs column '{col}'")
            numeric = pd.to_numeric(obs[col], errors="coerce")
            if numeric.isna().all():
                raise ValueError(f"Continuous color column '{col}' is non-numeric")
            continuous_color = {
                "column": col,
                "values": numeric.to_numpy(),
                "mask": mask,
                "cmap": rule.get("cmap", "viridis"),
                "vmin": rule.get("vmin"),
                "vmax": rule.get("vmax"),
                "show_colorbar": bool(rule.get("show_colorbar", True)),
            }
        else:
            for key in ("color", "marker", "size", "alpha", "linewidth", "edgecolor", "zorder"):
                if key in rule:
                    styles[key][mask] = rule[key]

    styles["continuous_color"] = continuous_color
    return styles


def _as_numpy(data: Any) -> np.ndarray:
    """Materialize xarray/dask/numpy-like data as a numpy array."""
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(data, "values"):
        data = data.values
    return np.asarray(data)


def _data_array_from_tree_node(node: Any) -> Any:
    """Find the image DataArray stored in one DataTree pyramid level."""
    if hasattr(node, "dims") and hasattr(node, "shape"):
        return node
    if hasattr(node, "ds"):
        dataset = node.ds
        if hasattr(dataset, "data_vars") and dataset.data_vars:
            return next(iter(dataset.data_vars.values()))
    if hasattr(node, "to_dataset"):
        dataset = node.to_dataset()
        if hasattr(dataset, "data_vars") and dataset.data_vars:
            return next(iter(dataset.data_vars.values()))
    if hasattr(node, "data_vars") and node.data_vars:
        return next(iter(node.data_vars.values()))
    return node


def _sorted_pyramid_keys(image_element: Any) -> List[str]:
    """Return sorted child keys for a multiscale DataTree."""
    children = getattr(image_element, "children", None)
    if children:
        keys = list(children.keys())
    elif hasattr(image_element, "keys"):
        keys = [str(k) for k in image_element.keys()]
    else:
        return []

    def _key_order(key: str) -> tuple[int, str]:
        if key.isdigit():
            return (int(key), key)
        digits = "".join(ch for ch in key if ch.isdigit())
        return (int(digits), key) if digits else (9999, key)

    return sorted(keys, key=_key_order)


def _select_pyramid_data_array(image_element: Any, image_scale: Optional[int] = None) -> tuple[Any, tuple[int, ...]]:
    """
    Select one DataArray from a SpatialData image.

    For multiscale DataTree images, default to the coarsest level so full-slide
    inspection does not materialize the largest OME-TIFF pyramid level.
    """
    keys = _sorted_pyramid_keys(image_element)
    if not keys:
        data_array = _data_array_from_tree_node(image_element)
        return data_array, tuple(getattr(data_array, "shape", ()))

    level_index = len(keys) - 1 if image_scale is None else int(image_scale)
    if level_index < 0 or level_index >= len(keys):
        raise ValueError(f"image_scale must be between 0 and {len(keys) - 1}")

    original_data_array = _data_array_from_tree_node(image_element[keys[0]])
    selected_data_array = _data_array_from_tree_node(image_element[keys[level_index]])
    return selected_data_array, tuple(getattr(original_data_array, "shape", ()))


def _channel_index_from_data_array(data_array: Any, image_channel: Optional[Any]) -> Optional[int]:
    """Resolve requested channel from DataArray c coordinates."""
    dims = list(getattr(data_array, "dims", ()))
    if "c" not in dims:
        return None
    c_axis = dims.index("c")
    n_channels = int(getattr(data_array, "shape", ())[c_axis])
    if image_channel is None:
        # 3-channel images are usually RGB; 4-channel Xenium morphology_focus is not RGBA.
        return None if n_channels == 3 else 0
    if isinstance(image_channel, int) or str(image_channel).isdigit():
        idx = int(image_channel)
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"image_channel index {idx} outside 0..{n_channels - 1}")
        return idx
    if hasattr(data_array, "coords") and "c" in data_array.coords:
        labels = [str(v) for v in data_array.coords["c"].values]
        if str(image_channel) in labels:
            return labels.index(str(image_channel))
    raise ValueError(f"Could not resolve image_channel '{image_channel}'")


def _prepare_image_array(arr: np.ndarray, channel_index: Optional[int] = None) -> np.ndarray:
    """Convert image data into a numeric array accepted by matplotlib.imshow."""
    if channel_index is not None and arr.ndim >= 3:
        arr = arr[channel_index]

    # Some raw Xenium image readers can surface object arrays wrapping ndarrays.
    if arr.dtype == object:
        if arr.size == 1:
            arr = np.asarray(arr.item())
        elif arr.ndim >= 1 and isinstance(arr.flat[0], np.ndarray):
            arr = np.stack([np.asarray(v) for v in arr.flat], axis=0)
        else:
            arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        if arr.shape[0] == 4:
            arr = arr[0]
        else:
            arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    return arr


def _channel_axis(data_array: Any) -> Optional[int]:
    """Return channel axis index if the DataArray has one."""
    dims = list(getattr(data_array, "dims", ()))
    return dims.index("c") if "c" in dims else None


def _normalize_image_channel(arr: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Robustly normalize one image channel to 0..1."""
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, percentiles)
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _parse_channel_colors(channel_colors: Optional[Any]) -> List[tuple[float, float, float]]:
    """Parse TOML/CLI image channel color definitions."""
    if not channel_colors:
        return []
    if isinstance(channel_colors, str):
        channel_colors = [c.strip() for c in channel_colors.split(",") if c.strip()]
    if isinstance(channel_colors, dict):
        ordered_items = sorted(channel_colors.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
        channel_colors = [value for _, value in ordered_items]
    return [to_rgb(str(color)) for color in channel_colors]


def _parse_channel_indices(image_channels: Optional[Any], n_channels: int) -> List[int]:
    """Parse channel indices for composite rendering."""
    if image_channels is None:
        return list(range(n_channels))
    if isinstance(image_channels, str):
        values = [v.strip() for v in image_channels.split(",") if v.strip()]
    elif isinstance(image_channels, Iterable):
        values = list(image_channels)
    else:
        values = [image_channels]
    indices = [int(v) for v in values]
    for idx in indices:
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"image channel index {idx} outside 0..{n_channels - 1}")
    return indices


def _composite_image_array(
    arr: np.ndarray,
    data_array: Any,
    image_channels: Optional[Any] = None,
    channel_colors: Optional[Any] = None,
    contrast_percentiles: Optional[Any] = None,
) -> np.ndarray:
    """Composite multiple image channels into an RGB image."""
    colors = _parse_channel_colors(channel_colors)
    percentiles = tuple(float(v) for v in (contrast_percentiles or (1.0, 99.8)))
    if len(percentiles) != 2:
        raise ValueError("image contrast_percentiles must have exactly two values")

    c_axis = _channel_axis(data_array)
    if c_axis is not None and arr.ndim >= 3:
        arr = np.moveaxis(arr, c_axis, 0)
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        pass
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Cannot infer channel axis for image shape {arr.shape}")

    channel_indices = _parse_channel_indices(image_channels, n_channels=arr.shape[0])
    if not colors:
        colors = [to_rgb(c) for c in ("#264bff", "#00ff33", "#ff260d", "#ff00ff")]

    rgb = np.zeros((*arr.shape[-2:], 3), dtype=np.float32)
    for color_idx, channel_idx in enumerate(channel_indices):
        color = np.asarray(colors[color_idx % len(colors)], dtype=np.float32)
        normalized = _normalize_image_channel(arr[channel_idx], percentiles)
        rgb += normalized[..., None] * color
    return np.clip(rgb, 0.0, 1.0)


def _infer_full_image_shape(original_shape: tuple[int, ...]) -> tuple[int, int]:
    """Infer full-resolution image height/width from level 0 shape."""
    if len(original_shape) >= 3 and original_shape[0] in (1, 3, 4):
        return int(original_shape[-2]), int(original_shape[-1])
    if len(original_shape) >= 2:
        return int(original_shape[-2]), int(original_shape[-1])
    raise ValueError(f"Cannot infer image shape from {original_shape}")


def _image_extent_for_coords(
    full_width: int,
    full_height: int,
    coords: Optional[np.ndarray] = None,
    image_transform: str = "scale_xy",
) -> tuple[float, float, float, float]:
    """Return image extent in the same coordinate system used for cell dots."""
    transform = (image_transform or "scale_xy").lower()
    if coords is None or transform in {"pixel", "pixels", "none", "direct"}:
        return (0.0, float(full_width), 0.0, float(full_height))

    spatial_xmax = float(np.nanmax(coords[:, 0]))
    spatial_ymax = float(np.nanmax(coords[:, 1]))
    if spatial_xmax <= 0 or spatial_ymax <= 0:
        return (0.0, float(full_width), 0.0, float(full_height))

    if transform == "scale_xy":
        return (0.0, spatial_xmax, 0.0, spatial_ymax)
    if transform == "scale_uniform":
        x_scale = full_width / spatial_xmax
        y_scale = full_height / spatial_ymax
        uniform_scale = (x_scale + y_scale) / 2.0
        return (0.0, full_width / uniform_scale, 0.0, full_height / uniform_scale)
    raise ValueError(f"Unsupported image_transform '{image_transform}'")


def extract_image_overlay(
    image_element: Any,
    image_scale: Optional[int] = None,
    image_channel: Optional[Any] = None,
    coords: Optional[np.ndarray] = None,
    image_transform: str = "scale_xy",
    image_channels: Optional[Any] = None,
    channel_colors: Optional[Any] = None,
    contrast_percentiles: Optional[Any] = None,
) -> ImageOverlay:
    """Extract a numeric image overlay from a SpatialData image element."""
    data_array, original_shape = _select_pyramid_data_array(image_element, image_scale=image_scale)
    raw_arr = _as_numpy(data_array)

    if channel_colors or image_channels is not None:
        arr = _composite_image_array(
            raw_arr,
            data_array=data_array,
            image_channels=image_channels,
            channel_colors=channel_colors,
            contrast_percentiles=contrast_percentiles,
        )
    elif len(original_shape) >= 2:
        channel_index = _channel_index_from_data_array(data_array, image_channel=image_channel)
        arr = _prepare_image_array(raw_arr, channel_index=channel_index)

    full_height, full_width = _infer_full_image_shape(original_shape)

    return ImageOverlay(
        data=arr,
        extent=_image_extent_for_coords(
            full_width=full_width,
            full_height=full_height,
            coords=coords,
            image_transform=image_transform,
        ),
    )


def _plot_background(ax: Any, image_overlay: ImageOverlay, image_alpha: float) -> None:
    """Plot background image with ROI extent alignment."""
    ax.imshow(
        image_overlay.data,
        extent=image_overlay.extent,
        origin="lower",
        alpha=image_alpha,
        zorder=0,
    )


def plot_roi(
    coords: np.ndarray,
    obs: pd.DataFrame,
    roi: ROI,
    style_arrays: Dict[str, Any],
    *,
    title: Optional[str] = None,
    figsize: Optional[List[float]] = None,
    dpi: int = 300,
    background_image: Optional[ImageOverlay] = None,
    image_alpha: float = 0.5,
):
    """
    Build one ROI/full-slide point plot and return the live Matplotlib objects.

    The caller owns the returned figure (it is *not* closed here), making this
    suitable for interactive notebook use and further customization.

    Args:
        coords: ``(n_points, 2)`` coordinate array.
        obs: Per-point observation DataFrame (kept for API symmetry).
        roi: Region of interest to render.
        style_arrays: Style arrays from :func:`compile_style_arrays`.
        title: Optional title (defaults to ``roi.name``).
        figsize: Optional ``[width, height]``.
        dpi: Figure DPI.
        background_image: Optional background image overlay.
        image_alpha: Alpha for the background image.

    Returns:
        ``(fig, ax)`` tuple, or ``None`` when no points fall inside the ROI.
    """
    in_roi = (
        (coords[:, 0] >= roi.xmin)
        & (coords[:, 0] <= roi.xmax)
        & (coords[:, 1] >= roi.ymin)
        & (coords[:, 1] <= roi.ymax)
    )
    if not np.any(in_roi):
        logging.warning("No points fall inside ROI %s", roi.name)
        return None

    x = coords[in_roi, 0]
    y = coords[in_roi, 1]
    fig, ax = plt.subplots(figsize=tuple(figsize or [8, 8]), dpi=dpi)
    if background_image is not None:
        _plot_background(ax, background_image, image_alpha=image_alpha)

    markers = style_arrays["marker"][in_roi]
    sizes = style_arrays["size"][in_roi]
    alphas = style_arrays["alpha"][in_roi]
    linewidths = style_arrays["linewidth"][in_roi]
    edgecolors = style_arrays["edgecolor"][in_roi]
    zorders = style_arrays["zorder"][in_roi]
    cont = style_arrays.get("continuous_color")

    if cont is not None:
        values = cont["values"][in_roi]
        vmin = np.nanmin(values) if cont["vmin"] is None else float(cont["vmin"])
        vmax = np.nanmax(values) if cont["vmax"] is None else float(cont["vmax"])
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cont["cmap"])
        rgba = cmap(norm(values))
        rgba[:, 3] = np.clip(alphas, 0.0, 1.0)
        point_colors = rgba
    else:
        point_colors = style_arrays["color"][in_roi]

    groups = pd.DataFrame(
        {
            "marker": markers,
            "zorder": zorders,
            "linewidth": linewidths,
            "edgecolor": edgecolors,
        }
    )
    for _, idx in groups.groupby(["marker", "zorder", "linewidth", "edgecolor"]).groups.items():
        idx_array = np.asarray(list(idx))
        ax.scatter(
            x[idx_array],
            y[idx_array],
            c=np.asarray(point_colors)[idx_array],
            s=np.asarray(sizes)[idx_array],
            marker=str(markers[idx_array][0]),
            linewidths=float(linewidths[idx_array][0]),
            edgecolors=str(edgecolors[idx_array][0]),
            zorder=float(zorders[idx_array][0]),
        )

    if cont is not None and cont.get("show_colorbar", True):
        sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cont["cmap"]))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=cont["column"])

    ax.set_xlim(roi.xmin, roi.xmax)
    ax.set_ylim(roi.ymin, roi.ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or roi.name)
    fig.tight_layout()
    return fig, ax


def render_roi_plot(
    coords: np.ndarray,
    obs: pd.DataFrame,
    roi: ROI,
    style_arrays: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = None,
    figsize: Optional[List[float]] = None,
    dpi: int = 300,
    background_image: Optional[ImageOverlay] = None,
    image_alpha: float = 0.5,
) -> None:
    """Render one ROI/full-slide point plot to disk."""
    result = plot_roi(
        coords,
        obs,
        roi,
        style_arrays,
        title=title,
        figsize=figsize,
        dpi=dpi,
        background_image=background_image,
        image_alpha=image_alpha,
    )
    if result is None:
        return

    fig, _ = result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def resolve_background_image(
    image_sdata: Any,
    coords: np.ndarray,
    *,
    image_layer: Optional[str] = None,
    image_scale: Optional[int] = None,
    image_channel: Optional[str] = None,
    image_transform: Optional[str] = None,
    image_channels: Optional[str] = None,
    channel_colors: Optional[str] = None,
    contrast_percentiles: Optional[Any] = None,
) -> Optional[ImageOverlay]:
    """
    Build a background :class:`ImageOverlay` from a SpatialData images mapping.

    Selects the image layer (first available when ``image_layer`` is None),
    extracts the overlay aligned to ``coords``, and returns ``None`` when no
    images are available or extraction fails.

    Args:
        image_sdata: Object exposing an ``images`` mapping (e.g. SpatialData).
        coords: ``(n_points, 2)`` coordinate array for alignment.
        image_layer: Optional image layer key; defaults to the first layer.
        image_scale: Optional multiscale pyramid level.
        image_channel: Optional single channel name/index.
        image_transform: Coordinate transform (defaults to ``scale_xy``).
        image_channels: Optional comma-separated channels for compositing.
        channel_colors: Optional comma-separated colors for compositing.
        contrast_percentiles: Optional contrast percentile spec.

    Returns:
        An :class:`ImageOverlay`, or ``None`` if unavailable.
    """
    if not (hasattr(image_sdata, "images") and image_sdata.images):
        logging.warning("Overlay image requested but no SpatialData images found")
        return None

    layer = image_layer or list(image_sdata.images.keys())[0]
    try:
        return extract_image_overlay(
            image_sdata.images[layer],
            image_scale=image_scale,
            image_channel=image_channel,
            coords=coords,
            image_transform=image_transform or "scale_xy",
            image_channels=image_channels,
            channel_colors=channel_colors,
            contrast_percentiles=contrast_percentiles,
        )
    except Exception as exc:
        logging.warning("Could not parse image layer '%s' for overlay: %s", layer, exc)
        return None


def run_roi_visualization(
    coords: np.ndarray,
    obs: pd.DataFrame,
    *,
    rois: Optional[List[ROI]] = None,
    view: str = "full",
    spec: Optional[Dict[str, Any]] = None,
    spec_path: Optional[str] = None,
    roi_strings: Optional[List[str]] = None,
    roi_file: Optional[str] = None,
    random_rois: int = 0,
    roi_width: Optional[float] = None,
    roi_height: Optional[float] = None,
    random_state: int = 0,
    max_points: Optional[int] = None,
    background_image: Optional[ImageOverlay] = None,
    figsize: Optional[List[float]] = None,
    dpi: int = 300,
    image_alpha: float = 0.5,
    title: Optional[str] = None,
) -> List[RoiPlotResult]:
    """
    Render one or more ROIs programmatically and return live figures.

    This is the notebook/script entry point that mirrors the ``visualize``
    command. ROIs are resolved (unless provided), optional subsampling is
    applied, styles are compiled, and each ROI is plotted via
    :func:`plot_roi`. Figures are returned open for further customization.

    Args:
        coords: ``(n_points, 2)`` coordinate array.
        obs: Per-point observation DataFrame.
        rois: Optional explicit ROI list (skips ``resolve_rois``).
        view: ``"full"`` or ``"roi"`` when resolving ROIs.
        spec: Optional visualization spec dict.
        spec_path: Optional path to a TOML spec (used when ``spec`` is None).
        roi_strings: Optional ROI bbox strings.
        roi_file: Optional CSV of ROIs.
        random_rois: Number of random ROIs to generate.
        roi_width: Random ROI width.
        roi_height: Random ROI height.
        random_state: Seed for random ROIs / subsampling.
        max_points: Optional max points to render (uniform random sample).
        background_image: Optional background overlay applied to every ROI.
        figsize: Optional ``[width, height]``.
        dpi: Figure DPI.
        image_alpha: Background image alpha.
        title: Optional title override.

    Returns:
        List of :class:`RoiPlotResult`. ROIs containing no points are skipped.
    """
    if spec is None:
        spec = load_visualization_spec(spec_path)

    coords = np.asarray(coords)
    if max_points and max_points > 0 and coords.shape[0] > max_points:
        sample_idx = np.random.default_rng(random_state).choice(
            coords.shape[0], size=max_points, replace=False
        )
        coords = coords[sample_idx]
        obs = obs.iloc[sample_idx].copy()

    if rois is None:
        rois = resolve_rois(
            coords=coords,
            view=view,
            roi_strings=roi_strings,
            roi_file=roi_file,
            random_rois=random_rois,
            roi_width=roi_width,
            roi_height=roi_height,
            random_state=random_state,
        )

    plot_spec = spec.get("plot", {})
    resolved_figsize = figsize or plot_spec.get("figsize", [8, 8])
    style_arrays = compile_style_arrays(obs=obs, spec=spec)

    results: List[RoiPlotResult] = []
    for roi in rois:
        result = plot_roi(
            coords,
            obs,
            roi,
            style_arrays,
            title=title or plot_spec.get("title") or roi.name,
            figsize=resolved_figsize,
            dpi=dpi,
            background_image=background_image,
            image_alpha=image_alpha,
        )
        if result is None:
            continue
        fig, ax = result
        results.append(RoiPlotResult(roi=roi, fig=fig, ax=ax))

    return results


def write_roi_metadata(rois: List[ROI], output_path: Path, random_state: Optional[int] = None) -> None:
    """Write ROI metadata CSV."""
    rows = []
    for roi in rois:
        row = roi.as_dict()
        if roi.source == "random":
            row["random_state"] = random_state
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_resolved_settings(settings: Dict[str, Any], output_path: Path) -> None:
    """Write resolved config/spec settings as JSON."""
    output_path.write_text(json.dumps(settings, indent=2, default=str))

