"""
Unit tests for the differential analysis core module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spatial_tk.core import differential


def _normalize(adata):
    import scanpy as sc

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


# ---------------------------------------------------------------------------
# Gene-expression DE
# ---------------------------------------------------------------------------
def test_run_gene_expression_de_mode_b(mock_adata_with_clusters):
    """Mode B finds markers for all groups and stores a unified rank key."""
    adata = _normalize(mock_adata_with_clusters)
    cluster_key = "leiden_res0p5"

    adata, gene_df = differential.run_gene_expression_de(adata, cluster_key)

    rank_key = f"rank_genes_{cluster_key}"
    assert rank_key in adata.uns
    assert adata.uns["rank_genes_groups_key"] == rank_key
    assert isinstance(gene_df, pd.DataFrame)
    assert len(gene_df) > 0
    assert "group" in gene_df.columns


def test_run_gene_expression_de_mode_a(mock_adata_with_clusters):
    """Mode A compares two groups and annotates group1/group2 columns."""
    adata = _normalize(mock_adata_with_clusters)

    subset, gene_df = differential.run_gene_expression_de(
        adata, "status", compare_groups=["HIV", "NEG"]
    )

    assert "HIV" in gene_df["group1"].values
    assert "NEG" in gene_df["group2"].values
    # Mode A runs on a subset copy limited to the two groups.
    assert set(subset.obs["status"].unique()) <= {"HIV", "NEG"}


def test_run_gene_expression_de_resume_skips_recompute(mock_adata_with_clusters):
    """resume=True does not re-run rank_genes_groups when results exist."""
    adata = _normalize(mock_adata_with_clusters)
    cluster_key = "leiden_res0p5"

    adata, _ = differential.run_gene_expression_de(adata, cluster_key)

    with patch("scanpy.tl.rank_genes_groups") as mock_rank:
        adata, gene_df = differential.run_gene_expression_de(
            adata, cluster_key, resume=True
        )

    assert not mock_rank.called
    assert isinstance(gene_df, pd.DataFrame)


def test_rank_key_matches_plotting_expectation(mock_adata_with_clusters):
    """The stored rank key matches what plotting.save_de_plots looks up."""
    adata = _normalize(mock_adata_with_clusters)
    cluster_key = "leiden_res0p5"

    adata, _ = differential.run_gene_expression_de(adata, cluster_key)

    assert f"rank_genes_{cluster_key}" in adata.uns


# ---------------------------------------------------------------------------
# obsm DE
# ---------------------------------------------------------------------------
def test_run_obsm_de_mode_a(mock_adata_with_clusters):
    """Mode A obsm DE returns per-feature t-test statistics."""
    adata = mock_adata_with_clusters
    rng = np.random.default_rng(0)
    obsm_df = pd.DataFrame(
        rng.normal(size=(adata.n_obs, 3)),
        columns=["f1", "f2", "f3"],
        index=adata.obs_names,
    )
    adata.obsm["score_mlm_custom"] = obsm_df

    result = differential.run_obsm_de(
        adata, "status", "score_mlm_custom", compare_groups=["HIV", "NEG"]
    )

    assert "t_statistic" in result.columns
    assert "mean_difference" in result.columns
    assert set(result["feature"]) == {"f1", "f2", "f3"}


def test_run_obsm_de_mode_b_group_means(mock_adata_with_clusters):
    """Mode B obsm DE returns per-group means indexed by group."""
    adata = mock_adata_with_clusters
    rng = np.random.default_rng(1)
    adata.obsm["score_mlm_custom"] = pd.DataFrame(
        rng.normal(size=(adata.n_obs, 2)),
        columns=["f1", "f2"],
        index=adata.obs_names,
    )

    result = differential.run_obsm_de(adata, "status", "score_mlm_custom")

    assert set(result.index) <= {"HIV", "NEG"}
    assert list(result.columns) == ["f1", "f2"]


def test_run_obsm_de_missing_layer_returns_none(mock_adata_with_clusters):
    """A missing obsm layer yields None instead of raising."""
    result = differential.run_obsm_de(
        mock_adata_with_clusters, "status", "does_not_exist"
    )
    assert result is None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def test_run_differential_analysis_combined(mock_adata_with_clusters):
    """The orchestrator returns both gene and obsm results when requested."""
    adata = _normalize(mock_adata_with_clusters)
    rng = np.random.default_rng(2)
    adata.obsm["score_mlm_custom"] = pd.DataFrame(
        rng.normal(size=(adata.n_obs, 2)),
        columns=["f1", "f2"],
        index=adata.obs_names,
    )

    results = differential.run_differential_analysis(
        adata,
        "status",
        compare_groups=["HIV", "NEG"],
        obsm_layer="score_mlm_custom",
    )

    assert results.gene_expression is not None
    assert results.obsm is not None
    assert results.rank_key == "rank_genes_status"


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def test_save_gene_expression_de_results_mode_b(tmp_path):
    df = pd.DataFrame(
        {
            "group": ["0", "0", "1"],
            "names": ["g1", "g2", "g3"],
            "scores": [1.0, 0.5, 2.0],
        }
    )
    differential.save_gene_expression_de_results(
        df, tmp_path, "leiden_res0p5", None, n_genes=1
    )
    assert (tmp_path / "de_genes_all_groups_leiden_res0p5.csv").exists()
    assert (tmp_path / "de_genes_top1_per_group_leiden_res0p5.csv").exists()


def test_save_gene_expression_de_results_mode_a(tmp_path):
    df = pd.DataFrame(
        {
            "names": ["g1", "g2"],
            "scores": [1.0, 0.5],
            "group1": ["HIV", "HIV"],
            "group2": ["NEG", "NEG"],
        }
    )
    differential.save_gene_expression_de_results(
        df, tmp_path, "status", ["HIV", "NEG"], n_genes=1
    )
    assert (tmp_path / "de_genes_HIV_vs_NEG.csv").exists()
    assert (tmp_path / "de_genes_top1_HIV_vs_NEG.csv").exists()


def test_save_obsm_de_results_mode_a(tmp_path):
    df = pd.DataFrame(
        {
            "feature": ["f1", "f2"],
            "mean_difference": [0.2, -0.1],
        }
    )
    differential.save_obsm_de_results(
        df, tmp_path, "score_mlm_custom", "status", ["HIV", "NEG"], n_top=1
    )
    assert (tmp_path / "de_score_mlm_custom_HIV_vs_NEG.csv").exists()
    assert (tmp_path / "de_score_mlm_custom_top1_HIV_vs_NEG.csv").exists()
