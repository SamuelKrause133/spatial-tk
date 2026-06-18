"""
Unit tests for CLI command modules.

These tests verify that command-line arguments are properly passed to core
functions. The annotate command has been replaced by quantitate + assign;
see test_quantitate_command.py and test_assign_command.py for those tests.
"""

import pytest
from unittest.mock import patch, MagicMock
from argparse import Namespace
from pathlib import Path


# ---------------------------------------------------------------------------
# quantitate – quick smoke tests for tmin and score_key wiring
# ---------------------------------------------------------------------------

def test_quantitate_passes_tmin_default():
    """quantitate uses default tmin=2."""
    from spatial_tk.commands import quantitate

    args = Namespace(
        input="test.zarr",
        output="output.zarr",
        inplace=False,
        markers="markers.csv",
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
        mock_to_df.return_value = MagicMock()
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["tmin"] == 2


def test_quantitate_passes_custom_tmin():
    """quantitate respects a custom tmin value."""
    from spatial_tk.commands import quantitate

    args = Namespace(
        input="test.zarr",
        output="output.zarr",
        inplace=False,
        markers="markers.csv",
        score_key="custom",
        method="mlm",
        tmin=1,
        preset_resources=None,
        panglao_min_sensitivity=0.5,
        panglao_canonical_only=True,
        filter_obs=None,
        save_plots=False,
        config=None,
    )

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
        mock_to_df.return_value = MagicMock()
        mock_score.return_value = mock_adata

        try:
            quantitate.main(args)
        except SystemExit:
            pass

        assert mock_score.called
        assert mock_score.call_args[1]["tmin"] == 1


# ---------------------------------------------------------------------------
# assign – smoke test for default DE behaviour
# ---------------------------------------------------------------------------

def test_assign_runs_de_by_default():
    """assign runs differential expression by default."""
    from spatial_tk.commands import assign

    args = Namespace(
        input="test.zarr",
        output="output.zarr",
        inplace=False,
        score_key="score_mlm_custom",
        cluster_key="leiden_res0p5",
        annotation_key=None,
        strategy="top_positive",
        run_de=True,
        save_plots=False,
        config=None,
    )

    mock_sdata = MagicMock()
    mock_adata = MagicMock()
    mock_adata.obsm = {"score_mlm_custom": MagicMock()}
    mock_adata.obs.columns = ["leiden_res0p5"]

    with patch("spatial_tk.commands.assign.Path") as mock_path_cls, \
         patch("spatial_tk.commands.assign.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.assign.save_command_output"), \
         patch("spatial_tk.commands.assign.get_output_path") as mock_out, \
         patch("spatial_tk.commands.assign.get_table") as mock_get_table, \
         patch("spatial_tk.commands.assign.annotation.assign_clusters") as mock_assign_clusters, \
         patch("spatial_tk.commands.assign.differential.run_gene_expression_de") as mock_de:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_assign_clusters.return_value = mock_adata
        mock_de.return_value = (mock_adata, None)

        try:
            assign.main(args)
        except SystemExit:
            pass

        assert mock_de.called, "run_gene_expression_de should have been called"
