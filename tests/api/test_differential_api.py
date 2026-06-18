"""
API integration tests for the differential step.

Mirrors the differential steps in tests/functional/test_full_pipeline.py but
exercises the core ``differential`` API directly instead of the CLI.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from spatial_tk.core import differential

pytestmark = pytest.mark.api


def test_run_gene_expression_de_mode_b(assigned_adata):
    adata = assigned_adata.copy()
    adata, gene_df = differential.run_gene_expression_de(adata, groupby="leiden_res0p5")
    assert "rank_genes_leiden_res0p5" in adata.uns
    assert isinstance(gene_df, pd.DataFrame) and len(gene_df) > 0


def test_run_gene_expression_de_mode_a(assigned_adata):
    adata = assigned_adata.copy()
    subset, gene_df = differential.run_gene_expression_de(
        adata, groupby="status", compare_groups=["HIV", "NEG"]
    )
    assert set(gene_df["group1"].unique()) == {"HIV"}
    assert set(gene_df["group2"].unique()) == {"NEG"}


def test_run_gene_expression_de_within_mode_b(assigned_adata):
    adata = assigned_adata.copy()
    returned, gene_df = differential.run_gene_expression_de(
        adata, groupby="leiden_res0p5", within="status"
    )
    assert returned is adata
    assert isinstance(gene_df, pd.DataFrame)
    if not gene_df.empty:
        assert {"within_col", "within_value", "n_cells"} <= set(gene_df.columns)
        assert set(gene_df["within_col"].unique()) == {"status"}


def test_run_differential_analysis_within(assigned_adata):
    results = differential.run_differential_analysis(
        assigned_adata.copy(),
        groupby="leiden_res0p5",
        within="status",
    )
    assert results.gene_expression is not None
    assert results.rank_key is None
    if not results.gene_expression.empty:
        assert "within_value" in results.gene_expression.columns


def test_run_obsm_de_mode_a(assigned_adata):
    obsm_df = differential.run_obsm_de(
        assigned_adata,
        groupby="status",
        obsm_layer="score_mlm_custom",
        compare_groups=["HIV", "NEG"],
    )
    assert obsm_df is not None
    assert "t_statistic" in obsm_df.columns


def test_run_obsm_de_missing_layer_returns_none(assigned_adata):
    assert (
        differential.run_obsm_de(
            assigned_adata, groupby="status", obsm_layer="does_not_exist"
        )
        is None
    )


def test_run_differential_analysis_combined(assigned_adata):
    results = differential.run_differential_analysis(
        assigned_adata.copy(),
        groupby="status",
        compare_groups=["HIV", "NEG"],
        obsm_layer="score_mlm_custom",
    )
    assert results.gene_expression is not None
    assert results.obsm is not None
    assert results.rank_key == "rank_genes_status"


def test_resume_skips_recompute(assigned_adata):
    adata = assigned_adata.copy()
    # First run populates the rank_genes results.
    differential.run_gene_expression_de(adata, groupby="leiden_res0p5")
    with patch("scanpy.tl.rank_genes_groups") as mock_rank:
        differential.run_gene_expression_de(
            adata, groupby="leiden_res0p5", resume=True
        )
    assert not mock_rank.called


def test_rank_key_stored_correctly(assigned_adata):
    adata = assigned_adata.copy()
    adata, _ = differential.run_gene_expression_de(adata, groupby="leiden_res0p5")
    # plotting.save_de_plots expects rank_genes_{cluster_key}.
    assert adata.uns["rank_genes_groups_key"] == "rank_genes_leiden_res0p5"
