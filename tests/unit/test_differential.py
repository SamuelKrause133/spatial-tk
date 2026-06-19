"""
Unit tests for the differential analysis core module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spatial_tk.core import differential
from spatial_tk.core.differential import DifferentialResults


def _normalize(adata):
    import scanpy as sc

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def _add_obsm(adata, key="score_mlm_custom", n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    adata.obsm[key] = pd.DataFrame(
        rng.normal(size=(adata.n_obs, n_features)),
        columns=[f"f{i+1}" for i in range(n_features)],
        index=adata.obs_names,
    )
    return adata


# ---------------------------------------------------------------------------
# Gene-expression backend (run_gene_expression_de)
# ---------------------------------------------------------------------------
def test_run_gene_expression_de_mode_b(mock_adata_with_clusters):
    """Mode B finds markers for all groups and stores a unified rank key."""
    adata = _normalize(mock_adata_with_clusters)
    cluster_key = "leiden_res0p5"

    adata, gene_df = differential.run_gene_expression_de(adata, cluster_key)

    rank_key = f"rank_genes_{cluster_key}"
    assert rank_key in adata.uns
    assert adata.uns["rank_genes_groups_key"] == rank_key
    assert isinstance(gene_df, pd.DataFrame) and len(gene_df) > 0
    assert "group" in gene_df.columns


def test_run_gene_expression_de_mode_a(mock_adata_with_clusters):
    """Mode A compares two groups and annotates group1/group2 columns."""
    adata = _normalize(mock_adata_with_clusters)

    subset, gene_df = differential.run_gene_expression_de(
        adata, "status", compare_groups=["HIV", "NEG"]
    )

    assert "HIV" in gene_df["group1"].values
    assert "NEG" in gene_df["group2"].values
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


# ---------------------------------------------------------------------------
# run_differential dispatch
# ---------------------------------------------------------------------------
def test_run_differential_ge_mode_b(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(adata, "leiden_res0p5")

    assert isinstance(res, DifferentialResults)
    assert res.source == "gene_expression"
    assert res.method == "wilcoxon"
    assert res.rank_key == "rank_genes_leiden_res0p5"
    assert "names" in res.results.columns and len(res.results) > 0


def test_run_differential_ge_mode_a(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "status", compare_groups=["HIV", "NEG"]
    )
    assert set(res.results["group1"].unique()) == {"HIV"}
    assert set(res.results["group2"].unique()) == {"NEG"}


def test_run_differential_on_layer(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    adata.layers["norm"] = adata.X.copy()

    res = differential.run_differential(adata, "leiden_res0p5", on="norm")
    assert res.source == "norm"
    assert len(res.results) > 0


def test_run_differential_invalid_on_raises(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    with pytest.raises(KeyError):
        differential.run_differential(adata, "leiden_res0p5", on="does_not_exist")


def test_run_differential_within_subset_requires_within(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    with pytest.raises(ValueError):
        differential.run_differential(
            adata, "leiden_res0p5", within_subset=["HIV"]
        )


# ---------------------------------------------------------------------------
# Within-stratified analysis (through run_differential)
# ---------------------------------------------------------------------------
def test_run_differential_within_mode_b_combines_strata(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(adata, "leiden_res0p5", within="status")

    assert res.adata is adata
    assert res.rank_key is None
    df = res.results
    assert len(df) > 0
    assert {"within_col", "within_value", "n_cells"} <= set(df.columns)
    assert set(df["within_col"].unique()) == {"status"}
    assert set(df["within_value"].unique()) == {"HIV", "NEG"}


def test_run_differential_within_mode_a_pairwise(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "status", within="leiden_res0p5", compare_groups=["HIV", "NEG"]
    )
    df = res.results
    assert "group1" in df.columns and "group2" in df.columns
    assert set(df["group1"].unique()) == {"HIV"}
    assert df["within_value"].nunique() >= 1


def test_run_differential_within_skips_invalid_strata(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    n = adata.n_obs

    region = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    adata.obs["region"] = pd.Categorical(region)
    status = np.where(np.arange(n) % 2 == 0, "HIV", "NEG").astype(object)
    status[region == "B"] = "HIV"
    adata.obs["status"] = pd.Categorical(status)

    res = differential.run_differential(
        adata, "status", within="region", compare_groups=["HIV", "NEG"]
    )
    assert set(res.results["within_value"].unique()) == {"A"}


def test_run_differential_within_no_eligible_returns_empty(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    adata.obs["status"] = pd.Categorical(["HIV"] * adata.n_obs)

    res = differential.run_differential(
        adata, "status", within="leiden_res0p5", compare_groups=["HIV", "NEG"]
    )
    assert res.results.empty


def test_run_differential_within_missing_column_raises(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    with pytest.raises(KeyError):
        differential.run_differential(
            adata, "leiden_res0p5", within="does_not_exist"
        )


def test_run_differential_within_subset_restricts(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "leiden_res0p5", within="status", within_subset=["HIV"]
    )
    assert set(res.results["within_value"].unique()) == {"HIV"}


def test_run_differential_within_subset_unknown_value_skipped(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "leiden_res0p5", within="status", within_subset=["HIV", "NOPE"]
    )
    assert set(res.results["within_value"].unique()) == {"HIV"}


def test_run_differential_within_over_obsm_means(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    _add_obsm(adata)

    res = differential.run_differential(
        adata, "leiden_res0p5", on="score_mlm_custom", within="status", method="means"
    )
    df = res.results
    assert len(df) > 0
    assert {"group", "feature", "mean", "within_value"} <= set(df.columns)


# ---------------------------------------------------------------------------
# obsm engines (run_obsm_de)
# ---------------------------------------------------------------------------
def test_run_obsm_de_ttest_tidy_long(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters)
    result = differential.run_obsm_de(
        adata, "status", "score_mlm_custom",
        compare_groups=["HIV", "NEG"], method="ttest",
    )
    assert {"feature", "mean_difference", "stat", "pval", "group1", "group2"} <= set(
        result.columns
    )
    assert set(result["feature"]) == {"f1", "f2", "f3"}


def test_run_obsm_de_means_long(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters, n_features=2)
    result = differential.run_obsm_de(
        adata, "status", "score_mlm_custom", method="means"
    )
    assert list(result.columns) == ["group", "feature", "mean"]
    assert set(result["group"]) <= {"HIV", "NEG"}
    assert set(result["feature"]) == {"f1", "f2"}


def test_run_obsm_de_ttest_requires_compare_groups(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters)
    with pytest.raises(ValueError):
        differential.run_obsm_de(adata, "status", "score_mlm_custom", method="ttest")


def test_run_obsm_de_missing_layer_returns_none(mock_adata_with_clusters):
    result = differential.run_obsm_de(
        mock_adata_with_clusters, "status", "does_not_exist", method="means"
    )
    assert result is None


def test_run_obsm_de_invalid_method_raises(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters)
    with pytest.raises(ValueError):
        differential.run_obsm_de(adata, "status", "score_mlm_custom", method="bogus")


def test_run_obsm_de_rankby(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters, n_features=4)
    result = differential.run_obsm_de(
        adata, "status", "score_mlm_custom", method="rankby"
    )
    assert {"feature", "obs_col", "stat", "pval", "padj"} <= set(result.columns)
    assert set(result["feature"]) == {"f1", "f2", "f3", "f4"}
    assert set(result["obs_col"]) == {"status"}


def test_run_differential_obsm_rankby(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters, n_features=3)
    res = differential.run_differential(
        adata, "status", on="score_mlm_custom", method="rankby"
    )
    assert res.source == "score_mlm_custom"
    assert res.method == "rankby"
    assert {"feature", "obs_col", "padj"} <= set(res.results.columns)


def test_run_differential_obsm_default_method(mock_adata_with_clusters):
    """obsm defaults to means without compare_groups, ttest with."""
    adata = _add_obsm(mock_adata_with_clusters, n_features=2)

    res_means = differential.run_differential(adata, "status", on="score_mlm_custom")
    assert res_means.method == "means"

    res_ttest = differential.run_differential(
        adata, "status", on="score_mlm_custom", compare_groups=["HIV", "NEG"]
    )
    assert res_ttest.method == "ttest"


# ---------------------------------------------------------------------------
# save_differential_results
# ---------------------------------------------------------------------------
def test_save_differential_results_ge_all_groups(tmp_path):
    df = pd.DataFrame(
        {"group": ["0", "0", "1"], "names": ["g1", "g2", "g3"], "scores": [1.0, 0.5, 2.0]}
    )
    res = DifferentialResults(
        adata=None, results=df, source="gene_expression", method="wilcoxon"
    )
    differential.save_differential_results(
        res, tmp_path, groupby="leiden_res0p5", n_top=1
    )
    assert (tmp_path / "de_genes_all_groups_leiden_res0p5.csv").exists()
    top = tmp_path / "de_genes_top1_all_groups_leiden_res0p5.csv"
    assert top.exists()
    # Top-1 per group => 2 rows.
    assert len(pd.read_csv(top)) == 2


def test_save_differential_results_ge_within(tmp_path):
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
    res = DifferentialResults(
        adata=None, results=df, source="gene_expression", method="wilcoxon"
    )
    differential.save_differential_results(
        res, tmp_path, groupby="leiden_res0p5", within="status", n_top=1
    )
    assert (tmp_path / "de_genes_all_groups_leiden_res0p5_within_status.csv").exists()
    top = tmp_path / "de_genes_top1_all_groups_leiden_res0p5_within_status.csv"
    assert top.exists()
    # Top-1 per (within_value, group) => 4 rows.
    assert len(pd.read_csv(top)) == 4


def test_save_differential_results_obsm(tmp_path):
    df = pd.DataFrame(
        {
            "feature": ["f1", "f2"],
            "mean_difference": [0.2, -0.1],
            "group1": ["HIV", "HIV"],
            "group2": ["NEG", "NEG"],
        }
    )
    res = DifferentialResults(
        adata=None, results=df, source="score_mlm_custom", method="ttest"
    )
    differential.save_differential_results(
        res, tmp_path, groupby="status", compare_groups=["HIV", "NEG"], n_top=1
    )
    assert (tmp_path / "de_score_mlm_custom_HIV_vs_NEG.csv").exists()
    assert (tmp_path / "de_score_mlm_custom_top1_HIV_vs_NEG.csv").exists()


def test_save_differential_results_empty_noop(tmp_path):
    res = DifferentialResults(
        adata=None, results=pd.DataFrame(), source="gene_expression", method="wilcoxon"
    )
    differential.save_differential_results(res, tmp_path, groupby="leiden_res0p5")
    assert list(tmp_path.iterdir()) == []
