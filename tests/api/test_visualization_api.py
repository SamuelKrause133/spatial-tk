"""
API integration tests for the visualization step.

Mirrors tests/functional/test_visualize_cli.py but exercises the core
``visualization`` API directly, returning live matplotlib figures/axes.
"""

import matplotlib
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spatial_tk.core import visualization

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.api

_SPEC = {"points": {"default_color": "#999999"}}


def test_plot_roi_returns_figure_and_axes(viz_coords_obs, tmp_path):
    coords, obs = viz_coords_obs
    styles = visualization.compile_style_arrays(obs, _SPEC)
    roi = visualization.ROI(
        "full",
        float(coords[:, 0].min()),
        float(coords[:, 1].min()),
        float(coords[:, 0].max()),
        float(coords[:, 1].max()),
        "full",
    )
    result = visualization.plot_roi(coords, obs, roi, styles)
    assert result is not None
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # plot_roi must not close the figure; caller owns its lifecycle.
    assert plt.fignum_exists(fig.number)
    fig.savefig(tmp_path / "roi.png")
    plt.close(fig)


def test_plot_roi_empty_returns_none(viz_coords_obs):
    coords, obs = viz_coords_obs
    styles = visualization.compile_style_arrays(obs, _SPEC)
    x_max = float(coords[:, 0].max())
    y_max = float(coords[:, 1].max())
    roi = visualization.ROI(
        "empty", x_max + 1e6, y_max + 1e6, x_max + 2e6, y_max + 2e6, "manual"
    )
    assert visualization.plot_roi(coords, obs, roi, styles) is None


def test_run_roi_visualization_single(viz_coords_obs):
    coords, obs = viz_coords_obs
    results = visualization.run_roi_visualization(
        coords, obs, view="full", spec=_SPEC
    )
    assert len(results) == 1
    assert isinstance(results[0].fig, Figure)
    assert results[0].roi.source == "full"
    plt.close(results[0].fig)


def test_run_roi_visualization_random_rois(viz_coords_obs):
    coords, obs = viz_coords_obs
    # Same parameters as tests/functional/test_visualize_cli.py.
    results = visualization.run_roi_visualization(
        coords,
        obs,
        view="roi",
        random_rois=2,
        roi_width=250.0,
        roi_height=250.0,
        random_state=7,
        spec=_SPEC,
    )
    assert len(results) == 2
    for r in results:
        assert plt.fignum_exists(r.fig.number)
        plt.close(r.fig)


def test_run_roi_visualization_roi_string(viz_coords_obs):
    coords, obs = viz_coords_obs
    xmin = float(coords[:, 0].min())
    ymin = float(coords[:, 1].min())
    xmax = float(coords[:, 0].max())
    ymax = float(coords[:, 1].max())
    roi_str = f"{xmin},{ymin},{xmax},{ymax}"
    results = visualization.run_roi_visualization(
        coords, obs, view="roi", roi_strings=[roi_str], spec=_SPEC
    )
    assert results[0].roi.xmin == xmin
    plt.close(results[0].fig)


def test_figure_axes_customizable(viz_coords_obs):
    coords, obs = viz_coords_obs
    results = visualization.run_roi_visualization(
        coords, obs, view="full", spec=_SPEC
    )
    ax = results[0].ax
    ax.set_title("custom title")
    assert ax.get_title() == "custom title"
    plt.close(results[0].fig)
