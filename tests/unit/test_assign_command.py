"""
Unit tests for the assign command.

Verifies that CLI arguments are correctly wired through to the core
annotation functions. All external I/O is mocked.
"""

import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock, call
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Return a Namespace mimicking fully-parsed assign arguments."""
    defaults = dict(
        input="test.zarr",
        output="output.zarr",
        inplace=False,
        score_key="score_mlm_custom",
        cluster_key=None,
        annotation_key=None,
        strategy="top_positive",
        run_de=True,
        save_plots=False,
        config=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _run_assign(args):
    """Run assign.main(args) with all common I/O mocked; returns mock_adata."""
    from spatial_tk.commands import assign

    mock_sdata = MagicMock()
    mock_adata = MagicMock()
    mock_adata.n_obs = 100
    mock_adata.n_vars = 300
    # score_key is present in obsm by default
    mock_adata.obsm = {args.score_key: MagicMock()}
    mock_adata.obs.columns = ["leiden_res0p5", "leiden_res1p0"]

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

        return mock_adata, mock_assign_clusters, mock_de


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_assign_calls_assign_clusters():
    """assign_clusters is called with score_key and default strategy='top_positive'."""
    args = _make_args(cluster_key="leiden_res0p5")
    _, mock_assign_clusters, _ = _run_assign(args)

    assert mock_assign_clusters.called
    call_kwargs = mock_assign_clusters.call_args[1]
    assert call_kwargs["score_key"] == "score_mlm_custom"
    assert call_kwargs["strategy"] == "top_positive"


def test_assign_custom_strategy_passed():
    """--strategy threshold is forwarded to assign_clusters."""
    args = _make_args(cluster_key="leiden_res0p5", strategy="threshold")
    _, mock_assign_clusters, _ = _run_assign(args)

    assert mock_assign_clusters.called
    assert mock_assign_clusters.call_args[1]["strategy"] == "threshold"


def test_assign_auto_discovers_leiden_keys():
    """Without --cluster-key, assign_clusters is called once per leiden_res* column."""
    args = _make_args()  # cluster_key=None → auto-discover
    _, mock_assign_clusters, _ = _run_assign(args)

    # The mock_adata has two leiden_res* columns
    assert mock_assign_clusters.call_count == 2
    called_keys = {c[1]["cluster_key"] for c in mock_assign_clusters.call_args_list}
    assert "leiden_res0p5" in called_keys
    assert "leiden_res1p0" in called_keys


def test_assign_single_cluster_key():
    """--cluster-key leiden_res0p5 → assign_clusters called exactly once."""
    args = _make_args(cluster_key="leiden_res0p5")
    _, mock_assign_clusters, _ = _run_assign(args)

    assert mock_assign_clusters.call_count == 1
    assert mock_assign_clusters.call_args[1]["cluster_key"] == "leiden_res0p5"


def test_assign_annotation_key_derived_from_cluster_key():
    """leiden_res0p5 → annotation_key='cell_type_res0p5' (no --annotation-key set)."""
    args = _make_args(cluster_key="leiden_res0p5")
    _, mock_assign_clusters, _ = _run_assign(args)

    assert mock_assign_clusters.called
    assert mock_assign_clusters.call_args[1]["annotation_key"] == "cell_type_res0p5"


def test_assign_custom_annotation_key():
    """--annotation-key my_labels is forwarded to assign_clusters."""
    args = _make_args(cluster_key="leiden_res0p5", annotation_key="my_labels")
    _, mock_assign_clusters, _ = _run_assign(args)

    assert mock_assign_clusters.called
    assert mock_assign_clusters.call_args[1]["annotation_key"] == "my_labels"


def test_assign_de_runs_by_default():
    """run_gene_expression_de is called for each cluster key by default."""
    args = _make_args()  # run_de=True, auto-discover two cluster keys
    _, _, mock_de = _run_assign(args)

    assert mock_de.call_count == 2


def test_assign_de_skipped_when_disabled():
    """--run-de false → run_gene_expression_de is never called."""
    args = _make_args(run_de=False)
    _, _, mock_de = _run_assign(args)

    assert not mock_de.called


def test_assign_missing_score_key_exits():
    """When obsm does not contain the score key, main() exits with code 1."""
    from spatial_tk.commands import assign

    args = _make_args(score_key="score_mlm_nonexistent")

    mock_sdata = MagicMock()
    mock_adata = MagicMock()
    mock_adata.obsm = {}  # score key absent
    mock_adata.obs.columns = ["leiden_res0p5"]

    with patch("spatial_tk.commands.assign.Path") as mock_path_cls, \
         patch("spatial_tk.commands.assign.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.assign.save_command_output"), \
         patch("spatial_tk.commands.assign.get_output_path") as mock_out, \
         patch("spatial_tk.commands.assign.get_table") as mock_get_table:

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata

        with pytest.raises(SystemExit) as exc_info:
            assign.main(args)

        assert exc_info.value.code == 1
