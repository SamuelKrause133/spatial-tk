"""
Unit tests for the quantitate command.

Verifies that CLI arguments are correctly wired through to the core
annotation functions. All external I/O is mocked.
"""

import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock, call
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Return a Namespace mimicking fully-parsed quantitate arguments."""
    defaults = dict(
        input="test.zarr",
        output="output.zarr",
        inplace=False,
        markers=None,
        score_key="custom",
        method="mlm",
        tmin=2,
        preset_resources=None,
        panglao_min_sensitivity=0.5,
        panglao_canonical_only=True,
        filter_obs=None,
        save_plots=False,
        config=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


_MOCK_NET_DF = pd.DataFrame(
    {"source": ["T cells", "T cells"], "target": ["CD3D", "CD3E"], "weight": [1, 1]}
)

_COMMON_PATCHES = [
    "spatial_tk.commands.quantitate.load_existing_spatial_data",
    "spatial_tk.commands.quantitate.save_command_output",
    "spatial_tk.commands.quantitate.get_output_path",
]


def _run_main_with_patches(args, extra_patches=None):
    """
    Run quantitate.main(args) with common I/O mocked.

    Returns a dict mapping patch target → Mock for assertions.
    """
    from spatial_tk.commands import quantitate

    all_patches = _COMMON_PATCHES + (extra_patches or [])

    mocks = {}
    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls:
        # Make Path(anything).exists() return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.__truediv__ = lambda self, other: self  # plots_dir / "..."
        mock_path_cls.return_value = mock_path_instance

        with patch.multiple("spatial_tk.commands.quantitate", **{p.split(".")[-1]: MagicMock() for p in _COMMON_PATCHES}):
            mock_sdata = MagicMock()
            mock_adata = MagicMock()
            mock_adata.n_obs = 100
            mock_adata.n_vars = 300
            mock_adata.var_names = [f"gene_{i}" for i in range(300)]
            mock_adata.obsm = {}

            with patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
                 patch("spatial_tk.commands.quantitate.save_command_output"), \
                 patch("spatial_tk.commands.quantitate.get_output_path") as mock_out:
                mock_load.return_value = mock_sdata
                mock_out.return_value = mock_path_instance

                from spatial_tk.commands.quantitate import get_table as _gt
                with patch("spatial_tk.commands.quantitate.get_table") as mock_get_table:
                    mock_get_table.return_value = mock_adata

                    try:
                        quantitate.main(args)
                    except SystemExit:
                        pass

                    return mock_adata, mock_get_table


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_quantitate_custom_markers_calls_run_scoring():
    """Given --markers, run_enrichment_scoring is called with the marker DataFrame."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv")

    mock_markers = {"T cells": ["CD3D", "CD3E"]}

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.n_obs = 100
        mock_adata.n_vars = 300
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_markers.return_value = mock_markers
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called, "run_enrichment_scoring should have been called"
        call_kwargs = mock_score.call_args[1]
        assert call_kwargs["score_key"] == "custom"


def test_quantitate_custom_score_key_passed_through():
    """--score-key fibroblasts is forwarded to run_enrichment_scoring."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv", score_key="fibroblasts")

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_markers.return_value = {"T cells": ["CD3D"]}
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["score_key"] == "fibroblasts"


def test_quantitate_method_mlm_default():
    """Without --method, run_enrichment_scoring is called with method='mlm'."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv")  # default method="mlm"

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_markers.return_value = {"T cells": ["CD3D"]}
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["method"] == "mlm"


def test_quantitate_method_ulm():
    """--method ulm is forwarded to run_enrichment_scoring."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv", method="ulm")

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_markers.return_value = {"T cells": ["CD3D"]}
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["method"] == "ulm"


def test_quantitate_tmin_default():
    """Default tmin=2 is passed to run_enrichment_scoring."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv")

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_markers.return_value = {"T cells": ["CD3D"]}
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["tmin"] == 2


def test_quantitate_tmin_custom():
    """--tmin 1 is forwarded to run_enrichment_scoring."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv", tmin=1)

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_markers.return_value = {"T cells": ["CD3D"]}
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["tmin"] == 1


def test_quantitate_filter_obs_parsed_and_passed():
    """--filter-obs causes filter_cells_by_obs to be called before scoring."""
    from spatial_tk.commands import quantitate

    args = _make_args(markers="markers.csv", filter_obs="cell_type==Fibroblast")

    mock_mask = MagicMock()

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.filter_cells_by_obs") as mock_filter, \
         patch("spatial_tk.commands.quantitate.annotation.load_marker_genes") as mock_load_markers, \
         patch("spatial_tk.commands.quantitate.annotation.markers_dict_to_dataframe") as mock_to_df, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_filter.return_value = (mock_mask, MagicMock())
        mock_load_markers.return_value = {"T cells": ["CD3D"]}
        mock_to_df.return_value = _MOCK_NET_DF
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        # filter_cells_by_obs must be called with the expression
        mock_filter.assert_called_once_with(mock_adata, "cell_type==Fibroblast")

        # The mask must be forwarded to run_enrichment_scoring
        assert mock_score.called
        assert mock_score.call_args[1]["mask"] is mock_mask


def test_quantitate_preset_resources_calls_load_preset():
    """--preset-resources panglao,hallmark: load_preset_resource called twice; run_enrichment_scoring called for each."""
    from spatial_tk.commands import quantitate

    args = _make_args(preset_resources="panglao,hallmark")  # no custom markers

    mock_net = MagicMock()

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_preset_resource") as mock_load_preset, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_preset.return_value = mock_net
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        # load_preset_resource called once per resource
        assert mock_load_preset.call_count == 2
        preset_names_called = [c[0][0] for c in mock_load_preset.call_args_list]
        assert "panglao" in preset_names_called
        assert "hallmark" in preset_names_called

        # run_enrichment_scoring called once per resource
        assert mock_score.call_count == 2


def test_quantitate_panglao_sensitivity_passed():
    """--panglao-min-sensitivity 0.7 is forwarded to load_preset_resource."""
    from spatial_tk.commands import quantitate

    args = _make_args(preset_resources="panglao", panglao_min_sensitivity=0.7)

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table, \
         patch("spatial_tk.commands.quantitate.annotation.load_preset_resource") as mock_load_preset, \
         patch("spatial_tk.commands.quantitate.annotation.run_enrichment_scoring") as mock_score:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.var_names = []
        mock_adata.obsm = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_load_preset.return_value = MagicMock()
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_load_preset.called
        call_kwargs = mock_load_preset.call_args[1]
        assert call_kwargs["panglao_min_sensitivity"] == 0.7


def test_quantitate_no_markers_no_preset_exits():
    """Neither --markers nor --preset-resources → sys.exit(1)."""
    from spatial_tk.commands import quantitate

    args = _make_args()  # both None by default

    with patch("spatial_tk.commands.quantitate.Path") as mock_path_cls, \
         patch("spatial_tk.commands.quantitate.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.quantitate.save_command_output"), \
         patch("spatial_tk.commands.quantitate.get_output_path") as mock_out, \
         patch("spatial_tk.commands.quantitate.get_table") as mock_get_table:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata

        with pytest.raises(SystemExit) as exc_info:
            quantitate.main(args)

        assert exc_info.value.code == 1
