"""
Unit tests for visualize command wiring.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from spatial_tk.commands import visualize


def _args(**overrides):
    defaults = dict(
        input="test.zarr",
        output="out.png",
        table_key=None,
        spatial_key="spatial",
        view="full",
        roi=None,
        roi_file=None,
        random_rois=0,
        roi_width=None,
        roi_height=None,
        random_state=0,
        max_points=None,
        spec=None,
        figsize=None,
        dpi=None,
        title=None,
        overlay_image=False,
        image_layer=None,
        image_source=None,
        image_scale=None,
        image_channel=None,
        image_transform=None,
        image_channels=None,
        image_channel_colors=None,
        image_alpha=None,
        overwrite=True,
        config=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_visualize_full_view_calls_renderer(tmp_path):
    args = _args()
    input_path = tmp_path / "test.zarr"
    input_path.mkdir()

    mock_sdata = MagicMock()
    mock_adata = MagicMock()
    mock_adata.n_obs = 3
    mock_adata.obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5]])}
    mock_adata.obs = pd.DataFrame({"cell_type": ["A", "B", "A"]})

    with patch("spatial_tk.commands.visualize.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.visualize.get_table") as mock_get_table, \
         patch("spatial_tk.commands.visualize.load_visualization_spec") as mock_load_spec, \
         patch("spatial_tk.commands.visualize.compile_style_arrays") as mock_compile, \
         patch("spatial_tk.commands.visualize.resolve_rois") as mock_rois, \
         patch("spatial_tk.commands.visualize._resolve_output") as mock_out, \
         patch("spatial_tk.commands.visualize.render_roi_plot") as mock_render:
        args.input = str(input_path)
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_spec.return_value = {}
        mock_rois.return_value = [MagicMock(name="full_slide")]
        mock_out.return_value = {"mode": "single_file", "path": MagicMock()}
        mock_compile.return_value = {"color": np.array(["#000000"] * 3), "marker": np.array(["o"] * 3), "size": np.ones(3), "alpha": np.ones(3), "linewidth": np.zeros(3), "edgecolor": np.array(["none"] * 3), "zorder": np.ones(3), "continuous_color": None}

        visualize.main(args)

        assert mock_render.called


def test_visualize_roi_mode_requires_source(tmp_path):
    args = _args(view="roi")
    input_path = tmp_path / "test.zarr"
    input_path.mkdir()
    args.input = str(input_path)
    with patch("spatial_tk.commands.visualize.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.visualize.get_table") as mock_get_table, \
         patch("spatial_tk.commands.visualize.load_visualization_spec", return_value={}):
        mock_adata = MagicMock()
        mock_adata.n_obs = 2
        mock_adata.obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 1.0]])}
        mock_adata.obs = pd.DataFrame({"cell_type": ["A", "B"]})
        mock_load.return_value = MagicMock()
        mock_get_table.return_value = mock_adata
        try:
            visualize.main(args)
        except SystemExit as exc:
            assert exc.code == 1


def test_visualize_uses_explicit_image_source(tmp_path):
    args = _args(overlay_image=True, image_source="/tmp/raw_xenium_dir")
    input_path = tmp_path / "test.zarr"
    input_path.mkdir()
    args.input = str(input_path)

    mock_sdata = MagicMock()
    mock_adata = MagicMock()
    mock_adata.n_obs = 3
    mock_adata.obsm = {"spatial": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5]])}
    mock_adata.obs = pd.DataFrame({"cell_type": ["A", "B", "A"]})
    mock_adata.uns = {}

    mock_image_sdata = MagicMock()
    mock_image_sdata.images = {"morphology_focus": MagicMock()}

    with patch("spatial_tk.commands.visualize.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.visualize.load_image_source") as mock_load_image_source, \
         patch("spatial_tk.commands.visualize.get_table") as mock_get_table, \
         patch("spatial_tk.commands.visualize.load_visualization_spec") as mock_load_spec, \
         patch("spatial_tk.commands.visualize.compile_style_arrays") as mock_compile, \
         patch("spatial_tk.commands.visualize.resolve_rois") as mock_rois, \
         patch("spatial_tk.commands.visualize._resolve_output") as mock_out, \
         patch("spatial_tk.core.visualization.extract_image_overlay") as mock_extract, \
         patch("spatial_tk.commands.visualize.render_roi_plot"):
        mock_load.return_value = mock_sdata
        mock_load_image_source.return_value = mock_image_sdata
        mock_get_table.return_value = mock_adata
        mock_load_spec.return_value = {}
        mock_rois.return_value = [MagicMock(name="full_slide")]
        mock_out.return_value = {"mode": "single_file", "path": MagicMock()}
        mock_compile.return_value = {"color": np.array(["#000000"] * 3), "marker": np.array(["o"] * 3), "size": np.ones(3), "alpha": np.ones(3), "linewidth": np.zeros(3), "edgecolor": np.array(["none"] * 3), "zorder": np.ones(3), "continuous_color": None}
        mock_extract.return_value = MagicMock()

        visualize.main(args)

        assert mock_load_image_source.called
        assert mock_extract.call_args[0][0] is mock_image_sdata.images["morphology_focus"]
