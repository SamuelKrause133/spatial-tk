"""
API integration tests for the differential step.

Mirrors the differential steps in tests/functional/test_full_pipeline.py but
exercises the unified ``run_differential`` API directly instead of the CLI.
"""

import pandas as pd
import pytest

from spatial_tk.core import differential
from spatial_tk.core.differential import DifferentialResults

pytestmark = pytest.mark.api


# ---------------------------------------------------------------------------
# Gene expression
# ---------------------------------------------------------------------------
def test_run_differential_ge_mode_b(assigned_adata):
    res = differential.run_differential(assigned_adata.copy(), groupby="leiden_res0p5")
    assert isinstance(res, DifferentialResults)
    assert res.rank_key == "rank_genes_leiden_res0p5"
    assert "names" in res.results.columns and len(res.results) > 0


def test_run_differential_ge_mode_a(assigned_adata):
    res = differential.run_differential(
        assigned_adata.copy(), groupby="status", compare_groups=["HIV", "NEG"]
    )
    assert set(res.results["group1"].unique()) == {"HIV"}
    assert set(res.results["group2"].unique()) == {"NEG"}


def test_run_differential_ge_within(assigned_adata):
    res = differential.run_differential(
        assigned_adata.copy(), groupby="leiden_res0p5", within="status"
    )
    assert res.rank_key is None
    if not res.results.empty:
        assert {"within_col", "within_value", "n_cells"} <= set(res.results.columns)
        assert set(res.results["within_col"].unique()) == {"status"}


def test_run_differential_ge_within_subset(assigned_adata):
    res = differential.run_differential(
        assigned_adata.copy(),
        groupby="leiden_res0p5",
        within="status",
        within_subset=["HIV"],
    )
    if not res.results.empty:
        assert set(res.results["within_value"].unique()) <= {"HIV"}


# ---------------------------------------------------------------------------
# obsm engines
# ---------------------------------------------------------------------------
def test_run_differential_obsm_ttest(assigned_adata):
    res = differential.run_differential(
        assigned_adata.copy(),
        groupby="status",
        on="score_mlm_custom",
        compare_groups=["HIV", "NEG"],
        method="ttest",
    )
    assert res.method == "ttest"
    assert {"feature", "mean_difference", "stat", "pval"} <= set(res.results.columns)


def test_run_differential_obsm_rankby(assigned_adata):
    res = differential.run_differential(
        assigned_adata.copy(),
        groupby="status",
        on="score_mlm_custom",
        method="rankby",
    )
    assert res.method == "rankby"
    assert {"feature", "obs_col", "padj"} <= set(res.results.columns)


def test_run_obsm_de_missing_layer_returns_none(assigned_adata):
    assert (
        differential.run_obsm_de(
            assigned_adata, "status", "does_not_exist", method="means"
        )
        is None
    )


def test_invalid_on_raises(assigned_adata):
    with pytest.raises(KeyError):
        differential.run_differential(
            assigned_adata.copy(), groupby="status", on="not_a_source"
        )


# ---------------------------------------------------------------------------
# resume + rank key
# ---------------------------------------------------------------------------
def test_resume_skips_recompute(assigned_adata):
    from unittest.mock import patch

    adata = assigned_adata.copy()
    differential.run_gene_expression_de(adata, groupby="leiden_res0p5")
    with patch("scanpy.tl.rank_genes_groups") as mock_rank:
        differential.run_gene_expression_de(
            adata, groupby="leiden_res0p5", resume=True
        )
    assert not mock_rank.called


def test_rank_key_stored_correctly(assigned_adata):
    res = differential.run_differential(assigned_adata.copy(), groupby="leiden_res0p5")
    assert res.adata.uns["rank_genes_groups_key"] == "rank_genes_leiden_res0p5"
