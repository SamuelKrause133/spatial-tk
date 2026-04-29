#!/usr/bin/env python3
"""
Visualization utilities for full-slide and ROI rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import tomllib
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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


def extract_image_overlay(
    image_element: Any,
    image_scale: Optional[int] = None,
    image_channel: Optional[Any] = None,
) -> ImageOverlay:
    """Extract a numeric image overlay from a SpatialData image element."""
    data_array, original_shape = _select_pyramid_data_array(image_element, image_scale=image_scale)
    channel_index = _channel_index_from_data_array(data_array, image_channel=image_channel)
    arr = _prepare_image_array(_as_numpy(data_array), channel_index=channel_index)

    if len(original_shape) >= 3 and original_shape[0] in (1, 3, 4):
        full_height, full_width = original_shape[-2], original_shape[-1]
    elif len(original_shape) >= 2:
        full_height, full_width = original_shape[-2], original_shape[-1]
    else:
        full_height, full_width = arr.shape[-2], arr.shape[-1]

    return ImageOverlay(
        data=arr,
        extent=(0.0, float(full_width), 0.0, float(full_height)),
    )


def _extract_image_array(image_element: Any) -> np.ndarray:
    """Backward-compatible image array extraction helper."""
    return extract_image_overlay(image_element).data


def _plot_background(ax: Any, image_overlay: ImageOverlay, image_alpha: float) -> None:
    """Plot background image with ROI extent alignment."""
    ax.imshow(
        image_overlay.data,
        extent=image_overlay.extent,
        origin="lower",
        alpha=image_alpha,
        zorder=0,
    )


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
    in_roi = (
        (coords[:, 0] >= roi.xmin)
        & (coords[:, 0] <= roi.xmax)
        & (coords[:, 1] >= roi.ymin)
        & (coords[:, 1] <= roi.ymax)
    )
    if not np.any(in_roi):
        logging.warning("No points fall inside ROI %s", roi.name)
        return

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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

