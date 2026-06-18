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
# Within-stratified gene-expression DE
# ---------------------------------------------------------------------------
def test_run_gene_expression_de_within_mode_b_combines_strata(mock_adata_with_clusters):
    """Mode B within strata combines per-stratum markers into one frame."""
    adata = _normalize(mock_adata_with_clusters)

    returned, df = differential.run_gene_expression_de(
        adata, "leiden_res0p5", within="status"
    )

    # The original (unchanged) adata is returned in within mode.
    assert returned is adata
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert {"within_col", "within_value", "n_cells"} <= set(df.columns)
    assert set(df["within_col"].unique()) == {"status"}
    assert set(df["within_value"].unique()) == {"HIV", "NEG"}


def test_run_gene_expression_de_within_mode_a_pairwise_per_stratum(
    mock_adata_with_clusters,
):
    """Mode A within strata keeps the pairwise group1/group2 annotations."""
    adata = _normalize(mock_adata_with_clusters)

    _, df = differential.run_gene_expression_de(
        adata, "status", within="leiden_res0p5", compare_groups=["HIV", "NEG"]
    )

    assert "group1" in df.columns and "group2" in df.columns
    assert set(df["group1"].unique()) == {"HIV"}
    assert set(df["group2"].unique()) == {"NEG"}
    assert "within_value" in df.columns and df["within_value"].nunique() >= 1


def test_run_gene_expression_de_within_skips_invalid_strata(mock_adata_with_clusters):
    """Strata missing a requested compare group are skipped, not errored."""
    adata = _normalize(mock_adata_with_clusters)
    n = adata.n_obs

    region = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    adata.obs["region"] = pd.Categorical(region)

    # Region A keeps both statuses; region B is forced to a single status so it
    # cannot support the HIV-vs-NEG comparison and must be skipped.
    status = np.where(np.arange(n) % 2 == 0, "HIV", "NEG").astype(object)
    status[region == "B"] = "HIV"
    adata.obs["status"] = pd.Categorical(status)

    _, df = differential.run_gene_expression_de(
        adata, "status", within="region", compare_groups=["HIV", "NEG"]
    )

    assert set(df["within_value"].unique()) == {"A"}
    assert "group1" in df.columns


def test_run_gene_expression_de_within_no_eligible_strata_returns_empty(
    mock_adata_with_clusters,
):
    """When no stratum is eligible, an empty DataFrame is returned."""
    adata = _normalize(mock_adata_with_clusters)
    adata.obs["status"] = pd.Categorical(["HIV"] * adata.n_obs)

    _, df = differential.run_gene_expression_de(
        adata, "status", within="leiden_res0p5", compare_groups=["HIV", "NEG"]
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_run_gene_expression_de_within_missing_column_raises(
    mock_adata_with_clusters,
):
    """An unknown within column raises a clear KeyError."""
    adata = _normalize(mock_adata_with_clusters)

    with pytest.raises(KeyError):
        differential.run_gene_expression_de(
            adata, "leiden_res0p5", within="does_not_exist"
        )


def test_run_differential_analysis_within_pass_through(mock_adata_with_clusters):
    """The orchestrator passes within through and reports rank_key=None."""
    adata = _normalize(mock_adata_with_clusters)
    rng = np.random.default_rng(3)
    adata.obsm["score_mlm_custom"] = pd.DataFrame(
        rng.normal(size=(adata.n_obs, 2)),
        columns=["f1", "f2"],
        index=adata.obs_names,
    )

    results = differential.run_differential_analysis(
        adata,
        "leiden_res0p5",
        within="status",
        obsm_layer="score_mlm_custom",
    )

    assert results.gene_expression is not None
    assert {"within_col", "within_value"} <= set(results.gene_expression.columns)
    assert results.rank_key is None
    assert results.obsm is not None


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


def test_save_gene_expression_de_results_within(tmp_path):
    df = pd.DataFrame(
        {
            "group": ["0", "1", "0", "1"],
            "names": ["g1", "g2", "g3", "g4"],
            "scores": [1.0, 0.5, 2.0, 1.5],
            "within_col": ["status"] * 4,
            "within_value": ["HIV", "HIV", "NEG", "NEG"],
            "n_cells": [10, 10, 12, 12],
        }
    )
    differential.save_gene_expression_de_results(
        df, tmp_path, "leiden_res0p5", None, n_genes=1, within="status"
    )
    assert (tmp_path / "de_genes_all_groups_leiden_res0p5_within_status.csv").exists()
    top_file = (
        tmp_path / "de_genes_top1_per_group_leiden_res0p5_within_status.csv"
    )
    assert top_file.exists()
    # Top-N is taken per (within_value, group): 2 strata x 2 groups x 1 gene.
    assert len(pd.read_csv(top_file)) == 4


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
