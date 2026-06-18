"""
API integration tests for the normalize step.

Mirrors tests/functional/test_normalize_command.py but exercises the core
``preprocessing`` API directly instead of the CLI.
"""

from unittest.mock import patch

import pytest

from spatial_tk.core import preprocessing

pytestmark = pytest.mark.api


def test_normalize_adds_highly_variable_column(raw_adata):
    adata = raw_adata.copy()
    adata = preprocessing.calculate_qc_metrics(adata)
    adata = preprocessing.filter_cells_and_genes(adata, min_genes=10, min_cells=3)
    adata = preprocessing.normalize_and_log(adata)
    adata = preprocessing.select_variable_genes(adata, n_top_genes=500)

    assert "highly_variable" in adata.var.columns
    assert "counts" in adata.layers
    assert adata.var["highly_variable"].sum() <= 500


def test_normalize_resume_skips_recompute(normalized_adata):
    adata = normalized_adata.copy()  # already has the "counts" layer
    with patch("scanpy.pp.normalize_total") as mock_norm:
        preprocessing.normalize_and_log(adata, resume=True)
    assert not mock_norm.called


def test_qc_metrics_calculated(raw_adata):
    adata = raw_adata.copy()
    adata = preprocessing.calculate_qc_metrics(adata)
    assert "pct_counts_mt" in adata.obs.columns
    assert "n_genes_by_counts" in adata.obs.columns
    assert "total_counts" in adata.obs.columns


def test_filter_reduces_or_preserves_cell_count(raw_adata):
    adata = raw_adata.copy()
    n_before = adata.n_obs
    adata = preprocessing.calculate_qc_metrics(adata)
    adata = preprocessing.filter_cells_and_genes(adata, min_genes=10, min_cells=1)
    assert adata.n_obs <= n_before
