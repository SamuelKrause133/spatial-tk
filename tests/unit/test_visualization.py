"""
Unit tests for visualization helpers.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spatial_tk.core import visualization


def test_parse_roi_string_valid():
    roi = visualization.parse_roi_string("1,2,11,22", name="r1")
    assert roi.xmin == 1.0
    assert roi.ymin == 2.0
    assert roi.xmax == 11.0
    assert roi.ymax == 22.0


def test_generate_random_rois_deterministic():
    coords = np.array([[0, 0], [100, 100], [40, 20]], dtype=float)
    rois_a = visualization.generate_random_rois(coords, n_rois=2, width=10, height=10, random_state=42)
    rois_b = visualization.generate_random_rois(coords, n_rois=2, width=10, height=10, random_state=42)
    assert [(r.xmin, r.ymin, r.xmax, r.ymax) for r in rois_a] == [
        (r.xmin, r.ymin, r.xmax, r.ymax) for r in rois_b
    ]


def test_compile_style_arrays_direct_rules():
    obs = pd.DataFrame({"cell_type": ["Macrophage", "T cell"]})
    spec = {
        "points": {"default_color": "#999999"},
        "rules": [{"where": "cell_type == 'Macrophage'", "color": "#ff0000", "marker": "x"}],
    }
    styles = visualization.compile_style_arrays(obs, spec)
    assert styles["color"][0] == "#ff0000"
    assert styles["marker"][0] == "x"
    assert styles["color"][1] == "#999999"


def test_compile_style_arrays_categorical_mapping():
    obs = pd.DataFrame({"infection_status": ["infected", "uninfected"]})
    spec = {
        "rules": [
            {
                "kind": "categorical",
                "marker_by": "infection_status",
                "values": {"infected": "x", "uninfected": "o"},
            }
        ]
    }
    styles = visualization.compile_style_arrays(obs, spec)
    assert styles["marker"][0] == "x"
    assert styles["marker"][1] == "o"


def test_compile_style_arrays_continuous_color():
    obs = pd.DataFrame({"viral_load": [0.1, 0.5, 1.0]})
    spec = {
        "rules": [
            {
                "kind": "continuous",
                "color_by": "viral_load",
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": 1.0,
            }
        ]
    }
    styles = visualization.compile_style_arrays(obs, spec)
    assert styles["continuous_color"] is not None
    assert styles["continuous_color"]["column"] == "viral_load"


def test_compile_style_arrays_continuous_non_numeric_raises():
    obs = pd.DataFrame({"viral_load": ["a", "b"]})
    spec = {"rules": [{"kind": "continuous", "color_by": "viral_load"}]}
    with pytest.raises(ValueError, match="non-numeric"):
        visualization.compile_style_arrays(obs, spec)


def test_extract_image_overlay_single_scale_channel_first():
    data = xr.DataArray(
        np.stack([np.ones((4, 5)), np.zeros((4, 5))]),
        dims=("c", "y", "x"),
        coords={"c": ["DAPI", "marker"]},
    )

    overlay = visualization.extract_image_overlay(data, image_channel="DAPI")

    assert overlay.data.shape == (4, 5)
    assert overlay.extent == (0.0, 5.0, 0.0, 4.0)
    assert np.all(overlay.data == 1)


# ---------------------------------------------------------------------------
# plot_roi / run_roi_visualization
# ---------------------------------------------------------------------------
def _coords_obs_styles(n=20):
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(n, 2))
    obs = pd.DataFrame({"cell_type": ["A" if i % 2 else "B" for i in range(n)]})
    styles = visualization.compile_style_arrays(
        obs, {"points": {"default_color": "#999999"}}
    )
    return coords, obs, styles


def test_plot_roi_returns_figure_and_axes(tmp_path):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    coords, obs, styles = _coords_obs_styles()
    roi = visualization.ROI("full", 0.0, 0.0, 100.0, 100.0, "full")

    result = visualization.plot_roi(coords, obs, roi, styles)
    assert result is not None
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # Figure remains open; caller owns its lifecycle.
    assert plt.fignum_exists(fig.number)
    fig.savefig(tmp_path / "roi.png")
    plt.close(fig)


def test_plot_roi_empty_returns_none():
    coords, obs, styles = _coords_obs_styles()
    # ROI far outside the point cloud.
    roi = visualization.ROI("empty", 1000.0, 1000.0, 2000.0, 2000.0, "manual")
    assert visualization.plot_roi(coords, obs, roi, styles) is None


def test_run_roi_visualization_full():
    from matplotlib.figure import Figure

    coords, obs, _ = _coords_obs_styles()
    results = visualization.run_roi_visualization(
        coords, obs, view="full", spec={"points": {"default_color": "#999999"}}
    )
    assert len(results) == 1
    assert isinstance(results[0].fig, Figure)
    assert results[0].roi.source == "full"


def test_run_roi_visualization_random_rois():
    coords, obs, _ = _coords_obs_styles(n=200)
    results = visualization.run_roi_visualization(
        coords,
        obs,
        view="roi",
        random_rois=2,
        roi_width=40,
        roi_height=40,
        random_state=7,
        spec={"points": {"default_color": "#999999"}},
    )
    assert len(results) == 2
    import matplotlib.pyplot as plt

    for r in results:
        assert plt.fignum_exists(r.fig.number)
        plt.close(r.fig)


def test_run_roi_visualization_roi_string():
    coords, obs, _ = _coords_obs_styles()
    results = visualization.run_roi_visualization(
        coords,
        obs,
        view="roi",
        roi_strings=["0,0,100,100"],
        spec={"points": {"default_color": "#999999"}},
    )
    assert results[0].roi.xmin == 0.0
    assert results[0].roi.xmax == 100.0


def test_run_roi_visualization_axes_customizable():
    coords, obs, _ = _coords_obs_styles()
    results = visualization.run_roi_visualization(
        coords, obs, view="full", spec={"points": {"default_color": "#999999"}}
    )
    ax = results[0].ax
    ax.set_title("custom title")
    assert ax.get_title() == "custom title"
