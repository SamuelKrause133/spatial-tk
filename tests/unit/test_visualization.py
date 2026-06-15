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
