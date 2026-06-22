"""
Unit tests for the differential analysis core module.
"""

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
    """Mode B finds one-vs-all markers for all groups."""
    adata = _normalize(mock_adata_with_clusters)
    cluster_key = "leiden_res0p5"

    adata, gene_df = differential.run_gene_expression_de(adata, cluster_key)

    assert isinstance(gene_df, pd.DataFrame) and len(gene_df) > 0
    assert {"group", "group1", "group2", "feature"} <= set(gene_df.columns)


def test_run_gene_expression_de_mode_a(mock_adata_with_clusters):
    """Mode A compares two groups and annotates group1/group2 columns."""
    adata = _normalize(mock_adata_with_clusters)

    subset, gene_df = differential.run_gene_expression_de(
        adata, "status", compare_groups=["HIV", "NEG"]
    )

    assert "HIV" in gene_df["group1"].values
    assert "NEG" in gene_df["group2"].values
    assert set(subset.obs["status"].unique()) <= {"HIV", "NEG"}


# ---------------------------------------------------------------------------
# run_differential dispatch
# ---------------------------------------------------------------------------
def test_run_differential_ge_mode_b(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(adata, "leiden_res0p5")

    assert isinstance(res, DifferentialResults)
    assert res.source == "gene_expression"
    assert res.method == "wilcoxon"
    assert {"group", "group1", "group2", "feature", "padj"} <= set(res.results.columns)
    assert len(res.results) > 0


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


def test_run_obsm_de_ttest_without_compare_groups_runs_one_vs_all(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters)
    result = differential.run_obsm_de(adata, "status", "score_mlm_custom", method="ttest")
    assert {"group", "group1", "group2", "feature", "padj"} <= set(result.columns)


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
# Generic matrix-statistics kernel
# ---------------------------------------------------------------------------
def _matrix_obs(seed=0, n=120, n_features=4):
    """Synthetic (matrix_df, obs_df) with planted signal in feature f1."""
    rng = np.random.default_rng(seed)
    group = np.where(np.arange(n) % 2 == 0, "A", "B")
    cont = rng.normal(size=n)
    base = rng.normal(size=(n, n_features))
    # f1 carries a strong group effect and correlates with cont.
    base[:, 0] += (group == "A") * 3.0 + cont * 2.0
    idx = [f"c{i}" for i in range(n)]
    matrix_df = pd.DataFrame(
        base, columns=[f"f{i+1}" for i in range(n_features)], index=idx
    )
    obs_df = pd.DataFrame(
        {
            "grp": pd.Categorical(group),
            "cont": cont,
            "batch": pd.Categorical(rng.choice(["x", "y"], size=n)),
        },
        index=idx,
    )
    return matrix_df, obs_df


def test_extract_matrix_source_x_layer_obsm(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters, n_features=3)
    adata.layers["norm"] = adata.X.copy()

    mx, kind = differential._extract_matrix_source(adata, "gene_expression")
    assert kind == "ge" and mx.shape[0] == adata.n_obs
    assert list(mx.columns) == list(adata.var_names)

    ml, kind = differential._extract_matrix_source(adata, "norm")
    assert kind == "layer" and ml.shape == mx.shape

    mo, kind = differential._extract_matrix_source(adata, "score_mlm_custom")
    assert kind == "obsm" and list(mo.columns) == ["f1", "f2", "f3"]


def test_matrix_stats_ttest_signal():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(test="ttest", groupby="grp", compare_groups=["A", "B"])
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    assert {"feature", "stat", "pval", "padj", "mean_difference"} <= set(df.columns)
    f1 = df[df["feature"] == "f1"].iloc[0]
    assert f1["pval"] < 1e-3
    assert (df["padj"] >= df["pval"] - 1e-9).all()


def test_matrix_stats_wilcoxon_signal():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(
        test="wilcoxon", groupby="grp", compare_groups=["A", "B"]
    )
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    assert set(df["test"]) == {"wilcoxon"}
    assert df[df["feature"] == "f1"].iloc[0]["pval"] < 1e-3


def test_matrix_stats_anova_signal():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(test="anova", groupby="grp")
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    assert {"feature", "predictor", "stat", "pval", "padj"} <= set(df.columns)
    assert df[df["feature"] == "f1"].iloc[0]["pval"] < 1e-3


def test_matrix_stats_spearman_signal():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(test="spearman", groupby="cont")
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    assert set(df["test"]) == {"spearman"}
    assert df[df["feature"] == "f1"].iloc[0]["pval"] < 1e-3


def test_matrix_stats_spearman_requires_numeric():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(test="spearman", groupby="grp")
    with pytest.raises(ValueError):
        differential._run_matrix_stats(matrix_df, obs_df, spec)


def test_matrix_stats_means_schema():
    matrix_df, obs_df = _matrix_obs(n_features=2)
    spec = differential.StatSpec(test="means", groupby="grp")
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    assert list(df.columns) == ["group", "feature", "mean"]
    assert set(df["group"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# Regression (statsmodels)
# ---------------------------------------------------------------------------
def test_regression_auto_design_signal():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(test="regression", groupby="grp")
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    assert {"feature", "predictor", "coef", "stderr", "stat", "pval", "padj"} <= set(
        df.columns
    )
    # Default target_coef selects the grp term.
    assert df["predictor"].str.contains("grp").all()
    assert df[df["feature"] == "f1"].iloc[0]["pval"] < 1e-3


def test_regression_with_covariates():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(
        test="regression", groupby="grp", covariates=["batch"]
    )
    df = differential._run_matrix_stats(matrix_df, obs_df, spec)
    # Only the grp term is reported by default, not the batch covariate.
    assert df["predictor"].str.contains("grp").all()
    assert df[df["feature"] == "f1"].iloc[0]["pval"] < 1e-3


def test_regression_missing_covariate_raises():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(
        test="regression", groupby="grp", covariates=["nope"]
    )
    with pytest.raises(KeyError):
        differential._run_matrix_stats(matrix_df, obs_df, spec)


def test_regression_formula_matches_auto_design():
    matrix_df, obs_df = _matrix_obs()
    auto = differential._run_matrix_stats(
        matrix_df, obs_df, differential.StatSpec(test="regression", groupby="grp")
    )
    formula = differential._run_matrix_stats(
        matrix_df,
        obs_df,
        differential.StatSpec(
            test="regression",
            groupby="grp",
            formula="C(grp)",
            target_coef="C(grp)[T.B]",
        ),
    )
    a = auto[auto["feature"] == "f1"].iloc[0]
    f = formula[formula["feature"] == "f1"].iloc[0]
    assert a["coef"] == pytest.approx(f["coef"], rel=1e-6)
    assert a["pval"] == pytest.approx(f["pval"], rel=1e-6)


def test_regression_formula_requires_target_coef():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(
        test="regression", groupby="grp", formula="C(grp)"
    )
    with pytest.raises(ValueError):
        differential._run_matrix_stats(matrix_df, obs_df, spec)


def test_regression_target_coef_list_and_all():
    matrix_df, obs_df = _matrix_obs()
    as_list = differential._run_matrix_stats(
        matrix_df,
        obs_df,
        differential.StatSpec(
            test="regression",
            groupby="grp",
            formula="C(grp) + cont",
            target_coef=["C(grp)[T.B]", "cont"],
        ),
    )
    assert set(as_list[as_list["feature"] == "f1"]["predictor"]) == {
        "C(grp)[T.B]",
        "cont",
    }

    as_all = differential._run_matrix_stats(
        matrix_df,
        obs_df,
        differential.StatSpec(
            test="regression",
            groupby="grp",
            formula="C(grp) + cont",
            target_coef="all",
        ),
    )
    assert {"C(grp)[T.B]", "cont"} <= set(as_all["predictor"].unique())


def test_regression_unknown_target_coef_raises():
    matrix_df, obs_df = _matrix_obs()
    spec = differential.StatSpec(
        test="regression",
        groupby="grp",
        formula="C(grp)",
        target_coef="C(grp)[T.ZZZ]",
    )
    with pytest.raises(ValueError):
        differential._run_matrix_stats(matrix_df, obs_df, spec)


# ---------------------------------------------------------------------------
# Multiple-testing correction helper
# ---------------------------------------------------------------------------
def test_adjust_pvalues_basic():
    df = pd.DataFrame({"pval": [0.001, 0.5, 0.5, 0.5, np.nan]})
    out = differential._adjust_pvalues(df)
    assert "padj" in out.columns
    valid = out.dropna(subset=["pval"])
    assert (valid["padj"] >= valid["pval"] - 1e-12).all()
    assert pd.isna(out["padj"].iloc[-1])


def test_adjust_pvalues_grouped_scope():
    df = pd.DataFrame(
        {
            "predictor": ["a", "a", "b", "b"],
            "pval": [0.01, 0.04, 0.01, 0.04],
        }
    )
    out = differential._adjust_pvalues(df, group_cols=["predictor"])
    # Each predictor corrected independently over 2 tests.
    a = out[out["predictor"] == "a"].sort_values("pval")
    assert a["padj"].iloc[0] == pytest.approx(0.02, rel=1e-6)


# ---------------------------------------------------------------------------
# run_differential through the generic kernel
# ---------------------------------------------------------------------------
def test_run_differential_ge_anova(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "leiden_res0p5", method="anova"
    )
    assert res.method == "anova"
    assert {"feature", "predictor", "padj"} <= set(res.results.columns)


def test_run_differential_ge_regression(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "status", method="regression"
    )
    assert res.method == "regression"
    assert {"feature", "predictor", "coef", "padj"} <= set(res.results.columns)


def test_run_differential_obsm_regression_formula(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters, n_features=3)
    res = differential.run_differential(
        adata,
        "status",
        on="score_mlm_custom",
        method="regression",
        formula="C(status)",
        target_coef="all",
    )
    assert res.method == "regression"
    assert {"feature", "predictor", "coef", "padj"} <= set(res.results.columns)


def test_run_differential_covariates_require_regression(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    with pytest.raises(ValueError):
        differential.run_differential(
            adata, "status", method="ttest", compare_groups=["HIV", "NEG"],
            covariates=["sample"],
        )


def test_run_differential_within_regression(mock_adata_with_clusters):
    adata = _normalize(mock_adata_with_clusters)
    res = differential.run_differential(
        adata, "status", within="leiden_res0p5", method="regression"
    )
    if not res.results.empty:
        assert {"within_value", "predictor", "coef", "padj"} <= set(
            res.results.columns
        )


def test_run_differential_invalid_method_for_source(mock_adata_with_clusters):
    adata = _add_obsm(mock_adata_with_clusters)
    with pytest.raises(ValueError):
        differential.run_differential(
            adata, "status", on="score_mlm_custom", method="logreg"
        )


# ---------------------------------------------------------------------------
# save_differential_results
# ---------------------------------------------------------------------------
def test_save_differential_results_ge_all_groups(tmp_path):
    df = pd.DataFrame(
        {
            "group": ["0", "0", "1"],
            "feature": ["g1", "g2", "g3"],
            "padj": [0.01, 0.20, 0.05],
        }
    )
    res = DifferentialResults(
        adata=None, results=df, source="gene_expression", method="wilcoxon"
    )
    differential.save_differential_results(
        res, tmp_path, groupby="leiden_res0p5", n_top=1
    )
    assert (tmp_path / "de_gene_expression_all_groups_leiden_res0p5.csv").exists()
    top = tmp_path / "de_gene_expression_top1_all_groups_leiden_res0p5.csv"
    assert top.exists()
    # Top-1 per group => 2 rows.
    assert len(pd.read_csv(top)) == 2


def test_save_differential_results_ge_within(tmp_path):
    df = pd.DataFrame(
        {
            "group": ["0", "1", "0", "1"],
            "feature": ["g1", "g2", "g3", "g4"],
            "padj": [0.01, 0.5, 0.02, 0.4],
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
    assert (
        tmp_path / "de_gene_expression_all_groups_leiden_res0p5_within_status.csv"
    ).exists()
    top = (
        tmp_path
        / "de_gene_expression_top1_all_groups_leiden_res0p5_within_status.csv"
    )
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
