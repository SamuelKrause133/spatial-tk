#!/usr/bin/env python3
"""
Differential analysis core API.

This module owns all differential logic for spatial-tk behind a single
notebook-friendly entrypoint, :func:`run_differential`, which dispatches by
data source:

- **Gene expression** (``on="gene_expression"``/``"X"`` or a layer name)
- **obsm embeddings** (``on=<obsm key>``, e.g. MLM/ULM enrichment scores)

The module uses a **single generic matrix-statistics kernel** for *any* source
(``.X``, a layer, or an ``obsm`` key): ``ttest``, ``wilcoxon``, ``spearman``,
``anova``, ``regression`` (statsmodels OLS with covariates or a user-supplied
formula), ``rankby`` (in-module ANOVA/Spearman selector), and descriptive ``means``. All
inferential tests return a tidy frame with Benjamini-Hochberg corrected
``padj`` values.

Any source can be stratified with ``within`` (run independently within each
category of an ``obs`` column) and optionally limited to specific strata with
``within_subset``.

Functions here are pure compute (no file I/O) and return ``AnnData`` and/or
``pandas.DataFrame`` objects so they can be composed in notebooks and scripts.
:func:`save_differential_results` handles CSV export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class DifferentialResults:
    """Container for the output of :func:`run_differential`."""

    adata: ad.AnnData
    results: pd.DataFrame
    source: str
    method: str


@dataclass
class StatSpec:
    """Configuration for the generic per-feature statistics engine.

    Attributes:
        test: One of ``ttest``, ``wilcoxon``, ``spearman``, ``anova``,
            ``regression``, ``means``.
        groupby: ``obs`` column that defines the predictor of interest
            (categorical for grouped tests, numeric for ``spearman`` and
            continuous regression).
        compare_groups: Optional two-element list selecting the directional
            comparison for ``ttest``/``wilcoxon``.
        covariates: Optional ``obs`` columns added to the regression design.
        formula: Optional Patsy right-hand-side string for ``regression``.
            Takes precedence over ``groupby`` + ``covariates``.
        target_coef: Coefficient selection for ``regression``: a single term
            name, a list of names, or the literal ``"all"``. Required when
            ``formula`` is provided.
        correction: Multiple-testing correction method passed to
            :func:`statsmodels.stats.multitest.multipletests`.
    """

    test: str
    groupby: str
    compare_groups: Optional[List[str]] = None
    covariates: Optional[List[str]] = None
    formula: Optional[str] = None
    target_coef: Optional[Union[str, List[str]]] = None
    correction: str = "fdr_bh"


# Grouped inferential tests need at least two cells per group.
_MIN_CELLS_PER_GROUP = 2

# Methods served by the generic matrix-statistics kernel (any source).
_GENERIC_METHODS = (
    "ttest",
    "wilcoxon",
    "spearman",
    "anova",
    "regression",
    "means",
    "rankby",
)

# Two-group methods need ``compare_groups``; descriptive ``means`` has no test.
_TWO_GROUP_METHODS = ("ttest", "wilcoxon")

# Methods whose predictor is continuous (no per-group eligibility gating).
_CONTINUOUS_METHODS = ("spearman", "regression")

# Methods valid for an ``obsm`` source (all in-module).
_OBSM_METHODS = _GENERIC_METHODS


# --------------------------------------------------------------------------- #
# Gene-expression differential expression (backend)
# --------------------------------------------------------------------------- #
def run_gene_expression_de(
    adata: ad.AnnData,
    groupby: str,
    *,
    compare_groups: Optional[List[str]] = None,
    method: str = "wilcoxon",
    layer: Optional[str] = None,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """
    Run gene-expression differential analysis through the generic kernel.

    Two modes:

    - **Pairwise**: when ``compare_groups`` is two values, the analysis runs on
      a subset copy of ``adata`` (returned alongside the results) comparing
      ``compare_groups[0]`` vs ``compare_groups[1]``.
    - **One-vs-all**: when ``compare_groups`` is ``None``, marker features are
      found for every group in ``groupby``.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        compare_groups: Optional list of exactly two groups to compare.
        method: Statistical test (e.g. ``wilcoxon``, ``ttest``, ``anova``,
            ``spearman``, ``regression``, ``means``, ``rankby``).
        layer: Optional expression layer (default ``None`` uses ``.X``).

    Returns:
        Tuple of ``(adata, results_df)``. In pairwise mode the first value is
        the subset copy used for the comparison.
    """
    on = layer if layer is not None else "gene_expression"
    spec = StatSpec(
        test=_normalize_method(method),
        groupby=groupby,
        compare_groups=compare_groups,
    )
    if compare_groups and len(compare_groups) == 2:
        mask = adata.obs[groupby].isin(compare_groups)
        adata_subset = adata[mask].copy()
        return adata_subset, _run_generic(adata_subset, on, spec)
    return adata, _run_generic(adata, on, spec)


# --------------------------------------------------------------------------- #
# obsm embedding differential analysis (backend)
# --------------------------------------------------------------------------- #
def run_obsm_de(
    adata: ad.AnnData,
    groupby: str,
    obsm_layer: str,
    *,
    compare_groups: Optional[List[str]] = None,
    method: str = "ttest",
    covariates: Optional[List[str]] = None,
    formula: Optional[str] = None,
    target_coef: Optional[Union[str, List[str]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Run single-shot differential analysis on an ``obsm`` embedding.

    Always returns a *tidy long* DataFrame so results compose across strata.

    Engines (``method``):

    - ``"ttest"`` / ``"wilcoxon"``: per-feature two-group test (requires
      ``compare_groups``); columns ``feature, group1, group2, mean_group1,
      mean_group2, mean_difference, stat, pval, padj``.
    - ``"means"``: per-group feature means; columns ``group, feature, mean``.
    - ``"anova"`` / ``"spearman"`` / ``"regression"`` / ``"rankby"``: association
      tests against ``groupby`` (see :func:`_run_matrix_stats`).

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        obsm_layer: Key in ``adata.obsm`` to analyze.
        compare_groups: Optional list of exactly two groups to compare
            (required for ``method="ttest"``/``"wilcoxon"``).
        method: One of :data:`_OBSM_METHODS`.
        covariates: Optional covariate columns for ``method="regression"``.
        formula: Optional Patsy formula for ``method="regression"``.
        target_coef: Coefficient selection for ``method="regression"``.

    Returns:
        Tidy long results DataFrame, or ``None`` if ``obsm_layer`` is absent.
    """
    if method not in _OBSM_METHODS:
        raise ValueError(
            f"obsm method must be one of {_OBSM_METHODS}, got {method!r}"
        )

    logging.info(f"obsm DE on {obsm_layer} (method={method})")

    if obsm_layer not in adata.obsm:
        logging.warning(f"  obsm layer '{obsm_layer}' not found. Skipping.")
        return None

    spec = StatSpec(
        test=_normalize_method(method),
        groupby=groupby,
        compare_groups=compare_groups,
        covariates=covariates,
        formula=formula,
        target_coef=target_coef,
    )
    return _run_generic(adata, obsm_layer, spec)


def _obsm_to_dataframe(
    adata: ad.AnnData, obsm_layer: str
) -> tuple[List[str], pd.DataFrame]:
    """Coerce an ``obsm`` entry into a named feature DataFrame."""
    embedding = adata.obsm[obsm_layer]

    if hasattr(embedding, "var_names"):
        feature_names = list(embedding.var_names)
        embedding_df = pd.DataFrame(
            embedding.X, columns=feature_names, index=adata.obs_names
        )
    elif isinstance(embedding, pd.DataFrame):
        feature_names = list(embedding.columns)
        embedding_df = embedding.copy()
        embedding_df.index = adata.obs_names
    else:
        feature_names = [f"feature_{i}" for i in range(embedding.shape[1])]
        embedding_df = pd.DataFrame(
            embedding, columns=feature_names, index=adata.obs_names
        )

    return feature_names, embedding_df


# --------------------------------------------------------------------------- #
# Generic matrix-statistics kernel (source-agnostic)
# --------------------------------------------------------------------------- #
def _extract_matrix_source(
    adata: ad.AnnData, on: str
) -> Tuple[pd.DataFrame, str]:
    """
    Extract a ``cells x features`` DataFrame for any supported source.

    ``on`` may be ``"gene_expression"``/``"X"``, a ``layers`` key, or an
    ``obsm`` key. Returns ``(matrix_df, kind)`` where ``kind`` is one of
    ``"ge"``, ``"layer"`` or ``"obsm"`` and ``matrix_df`` is indexed by
    ``adata.obs_names`` with named feature columns.
    """
    import scipy.sparse as sp

    if on in ("gene_expression", "X"):
        matrix = adata.X
        columns = list(adata.var_names)
        kind = "ge"
    elif on in adata.layers:
        matrix = adata.layers[on]
        columns = list(adata.var_names)
        kind = "layer"
    elif on in adata.obsm:
        feature_names, embedding_df = _obsm_to_dataframe(adata, on)
        return embedding_df[feature_names], "obsm"
    else:
        raise KeyError(
            f"on='{on}' is not 'gene_expression'/'X', a layer "
            f"({list(adata.layers)}), or an obsm key ({list(adata.obsm)})"
        )

    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix_df = pd.DataFrame(
        np.asarray(matrix), columns=columns, index=adata.obs_names
    )
    return matrix_df, kind


def _adjust_pvalues(
    df: pd.DataFrame,
    *,
    method: str = "fdr_bh",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Add a ``padj`` column with multiple-testing-corrected p-values.

    Correction is applied independently within each group defined by
    ``group_cols`` (e.g. per regression term), so unrelated scopes are not
    pooled. ``NaN`` p-values are passed through untouched.
    """
    from statsmodels.stats.multitest import multipletests

    df = df.copy()
    if df.empty or "pval" not in df.columns:
        df["padj"] = np.nan if not df.empty else pd.Series(dtype=float)
        return df

    padj = np.full(len(df), np.nan)

    def _correct(positions: np.ndarray) -> None:
        pvals = df["pval"].to_numpy(dtype=float)[positions]
        ok = ~np.isnan(pvals)
        if ok.sum() == 0:
            return
        corrected = np.full(pvals.shape, np.nan)
        corrected[ok] = multipletests(pvals[ok], method=method)[1]
        padj[positions] = corrected

    if group_cols:
        for positions in df.groupby(list(group_cols), sort=False).indices.values():
            _correct(np.asarray(positions))
    else:
        _correct(np.arange(len(df)))

    df["padj"] = padj
    return df


def _run_generic(adata: ad.AnnData, on: str, spec: StatSpec) -> pd.DataFrame:
    """Extract ``on`` as a matrix and run the generic stats kernel."""
    matrix_df, _ = _extract_matrix_source(adata, on)
    return _run_matrix_stats(matrix_df, adata.obs, spec)


def _run_matrix_stats(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """
    Single generic statistical entrypoint over a ``cells x features`` matrix.

    Dispatches on ``spec.test`` and returns a tidy long DataFrame. All
    inferential tests include a Benjamini-Hochberg ``padj`` column; the
    descriptive ``means`` summary returns ``group, feature, mean`` only.
    """
    if spec.test not in _GENERIC_METHODS:
        raise ValueError(
            f"generic test must be one of {_GENERIC_METHODS}, got {spec.test!r}"
        )

    matrix_df = matrix_df.loc[obs_df.index]

    if spec.test == "means":
        return _stats_means(matrix_df, obs_df, spec)
    if spec.test in _TWO_GROUP_METHODS:
        return _stats_two_group(matrix_df, obs_df, spec)
    if spec.test == "rankby":
        return _stats_rankby(matrix_df, obs_df, spec)
    if spec.test == "anova":
        return _stats_anova(matrix_df, obs_df, spec)
    if spec.test == "spearman":
        return _stats_spearman(matrix_df, obs_df, spec)
    return _stats_regression(matrix_df, obs_df, spec)


def _stats_means(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """Per-group feature means; columns ``group, feature, mean``."""
    logging.info(f"  Computing per-group means over {matrix_df.shape[1]} features")
    work = matrix_df.copy()
    work[spec.groupby] = obs_df[spec.groupby].values
    group_means = work.groupby(spec.groupby, observed=True)[
        list(matrix_df.columns)
    ].mean()
    return (
        group_means.reset_index()
        .melt(id_vars=spec.groupby, var_name="feature", value_name="mean")
        .rename(columns={spec.groupby: "group"})
    )


def _stats_two_group(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """Per-feature two-group test, pairwise or one-vs-all (tidy long)."""
    from scipy import stats

    groups = obs_df[spec.groupby]
    eligible = [g for g, n in groups.value_counts().items() if n >= _MIN_CELLS_PER_GROUP]
    if spec.compare_groups and len(spec.compare_groups) == 2:
        comparisons = [(spec.compare_groups[0], spec.compare_groups[1], "pairwise")]
    else:
        # One-vs-all markers across every eligible group.
        comparisons = [(g, "rest", "one-vs-all") for g in eligible]

    if not comparisons:
        raise ValueError(f"method='{spec.test}' found no eligible group comparisons")

    rows = []
    gvals = groups.to_numpy()
    for group1, group2, mode in comparisons:
        logging.info(f"  {spec.test}: {mode} comparison {group1} vs {group2}")
        m1 = gvals == group1
        m2 = (gvals == group2) if group2 != "rest" else (gvals != group1)
        if m1.sum() < _MIN_CELLS_PER_GROUP or m2.sum() < _MIN_CELLS_PER_GROUP:
            continue
        for feature in matrix_df.columns:
            vals = matrix_df[feature].to_numpy(dtype=float)
            a = vals[m1]
            b = vals[m2]
            if spec.test == "ttest":
                stat, pval = stats.ttest_ind(a, b)
            else:
                stat, pval = stats.ranksums(a, b)
            rows.append(
                {
                    "feature": feature,
                    "test": spec.test,
                    "group": group1,
                    "group1": group1,
                    "group2": group2,
                    "predictor": spec.groupby,
                    "mean_group1": float(np.mean(a)) if a.size else np.nan,
                    "mean_group2": float(np.mean(b)) if b.size else np.nan,
                    "mean_difference": (
                        float(np.mean(a) - np.mean(b)) if a.size and b.size else np.nan
                    ),
                    "stat": stat,
                    "pval": pval,
                    "n_obs": int(a.size + b.size),
                }
            )

    df = _adjust_pvalues(pd.DataFrame(rows))
    if df.empty:
        return df
    if "group" in df.columns:
        return df.sort_values(["group", "padj", "pval"], na_position="last")
    return df.sort_values("mean_difference", ascending=False, key=abs)


def _stats_rankby(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """
    In-module rankby behavior:
    - numeric predictor -> Spearman
    - categorical predictor -> one-way ANOVA
    """
    is_numeric = pd.api.types.is_numeric_dtype(obs_df[spec.groupby])
    if is_numeric:
        out = _stats_spearman(matrix_df, obs_df, spec)
    else:
        out = _stats_anova(matrix_df, obs_df, spec)
    out = out.copy()
    out["obs_col"] = spec.groupby
    return out


def _stats_anova(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """One-way ANOVA of each feature across categories of ``groupby``."""
    from scipy import stats

    groups = obs_df[spec.groupby]
    levels = [g for g in groups.dropna().unique()]
    masks = {lvl: (groups == lvl).to_numpy() for lvl in levels}
    masks = {lvl: m for lvl, m in masks.items() if m.sum() >= 2}
    if len(masks) < 2:
        raise ValueError(
            f"method='anova' needs >= 2 groups (>= 2 cells each) in "
            f"'{spec.groupby}', found {len(masks)}"
        )

    logging.info(f"  anova: {len(masks)} groups in '{spec.groupby}'")
    rows = []
    for feature in matrix_df.columns:
        vals = matrix_df[feature].to_numpy(dtype=float)
        samples = [vals[m] for m in masks.values()]
        stat, pval = stats.f_oneway(*samples)
        rows.append(
            {
                "feature": feature,
                "test": "anova",
                "predictor": spec.groupby,
                "stat": stat,
                "pval": pval,
                "n_obs": int(sum(s.size for s in samples)),
            }
        )

    df = _adjust_pvalues(pd.DataFrame(rows))
    return df.sort_values("pval")


def _stats_spearman(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """Spearman correlation of each feature against a continuous ``groupby``."""
    from scipy import stats

    predictor = pd.to_numeric(obs_df[spec.groupby], errors="coerce")
    if predictor.isna().all():
        raise ValueError(
            f"method='spearman' requires a numeric predictor; "
            f"'{spec.groupby}' is not numeric"
        )

    logging.info(f"  spearman: correlating features with '{spec.groupby}'")
    valid = ~predictor.isna()
    x = predictor[valid].to_numpy(dtype=float)

    rows = []
    for feature in matrix_df.columns:
        y = matrix_df.loc[valid.values, feature].to_numpy(dtype=float)
        stat, pval = stats.spearmanr(y, x)
        rows.append(
            {
                "feature": feature,
                "test": "spearman",
                "predictor": spec.groupby,
                "stat": stat,
                "pval": pval,
                "n_obs": int(x.size),
            }
        )

    df = _adjust_pvalues(pd.DataFrame(rows))
    return df.sort_values("pval")


def _design_term(obs_df: pd.DataFrame, column: str) -> str:
    """Wrap categorical columns in Patsy ``C()`` for a regression formula."""
    series = obs_df[column]
    if pd.api.types.is_numeric_dtype(series):
        return column
    return f"C({column})"


def _select_target_coefs(
    available: List[str],
    target_coef: Optional[Union[str, List[str]]],
    *,
    groupby: str,
) -> List[str]:
    """Resolve which fitted (non-intercept) terms to report."""
    if target_coef == "all":
        return available
    if target_coef is None:
        # Auto-built design: default to terms involving the predictor of
        # interest, falling back to every non-intercept term.
        related = [name for name in available if groupby in name]
        return related or available
    requested = [target_coef] if isinstance(target_coef, str) else list(target_coef)
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"target_coef {missing} not found among model terms {available}"
        )
    return requested


def _stats_regression(
    matrix_df: pd.DataFrame, obs_df: pd.DataFrame, spec: StatSpec
) -> pd.DataFrame:
    """Per-feature statsmodels OLS regression (formula or auto-built design)."""
    import statsmodels.formula.api as smf

    covariates = spec.covariates or []
    if spec.formula:
        if spec.target_coef is None:
            raise ValueError("target_coef is required when formula is provided")
        rhs = spec.formula
    else:
        for col in [spec.groupby, *covariates]:
            if col not in obs_df.columns:
                raise KeyError(f"regression column '{col}' not found in adata.obs")
        terms = [_design_term(obs_df, spec.groupby)]
        terms += [_design_term(obs_df, c) for c in covariates]
        rhs = " + ".join(terms)

    logging.info(f"  regression: feature ~ {rhs}")
    base = obs_df.copy()

    rows = []
    for feature in matrix_df.columns:
        data = base.copy()
        data["feature"] = matrix_df[feature].to_numpy(dtype=float)
        try:
            fit = smf.ols(f"feature ~ {rhs}", data=data).fit()
        except Exception as exc:  # singular design, all-NaN, etc.
            logging.warning(f"    regression failed for feature '{feature}': {exc}")
            continue

        available = [name for name in fit.params.index if name != "Intercept"]
        if not available:
            continue
        selected = _select_target_coefs(
            available,
            spec.target_coef,
            groupby=spec.groupby,
        )
        for term in selected:
            rows.append(
                {
                    "feature": feature,
                    "test": "regression",
                    "predictor": term,
                    "coef": float(fit.params[term]),
                    "stderr": float(fit.bse[term]),
                    "stat": float(fit.tvalues[term]),
                    "pval": float(fit.pvalues[term]),
                    "n_obs": int(fit.nobs),
                }
            )

    if not rows:
        return pd.DataFrame()
    df = _adjust_pvalues(pd.DataFrame(rows), group_cols=["predictor"])
    return df.sort_values(["predictor", "pval"])


# --------------------------------------------------------------------------- #
# Generic within-stratification wrapper
# --------------------------------------------------------------------------- #
def _eligible_groups(adata: ad.AnnData, groupby: str, min_cells: int) -> List:
    """Return ``groupby`` values with enough cells for a statistical test."""
    counts = adata.obs[groupby].value_counts()
    return [g for g, n in counts.items() if n >= min_cells]


def _run_within(
    adata: ad.AnnData,
    groupby: str,
    within: str,
    *,
    single_fn: Callable[[ad.AnnData], Optional[pd.DataFrame]],
    within_subset: Optional[List] = None,
    compare_groups: Optional[List[str]] = None,
    min_cells_per_group: int = _MIN_CELLS_PER_GROUP,
    group_gate: bool = True,
) -> pd.DataFrame:
    """
    Run a differential engine independently within each category of ``within``.

    ``single_fn`` is invoked on a per-stratum subset copy and must return a
    tidy DataFrame (or ``None``/empty to skip). Each stratum's rows are
    annotated with ``within_col``, ``within_value`` and ``n_cells`` and then
    concatenated.

    When ``group_gate`` is True (grouped tests), strata lacking enough groups
    (each with at least ``min_cells_per_group`` cells) for the requested
    comparison are skipped with a warning, and in all-groups mode groups that
    are too small are dropped so the remaining comparison stays valid. When
    ``group_gate`` is False (continuous predictors such as ``spearman`` /
    ``regression``), each stratum is simply handed to ``single_fn``.
    ``within_subset`` restricts the strata that are processed.

    Returns the combined results DataFrame (empty if no stratum was eligible).
    """
    if within not in adata.obs.columns:
        raise KeyError(f"within column '{within}' not found in adata.obs")

    strata = list(adata.obs[within].dropna().unique())

    if within_subset is not None:
        available = set(strata)
        missing = [s for s in within_subset if s not in available]
        if missing:
            logging.warning(
                f"  within_subset values not found in '{within}' and skipped: "
                f"{missing}"
            )
        requested = set(within_subset)
        strata = [s for s in strata if s in requested]

    logging.info(
        f"Differential: stratifying by '{within}' "
        f"(groupby='{groupby}', {len(strata)} strata)"
    )

    results: List[pd.DataFrame] = []

    for stratum in strata:
        sub = adata[adata.obs[within] == stratum].copy()

        if hasattr(sub.obs[groupby], "cat"):
            sub.obs[groupby] = sub.obs[groupby].cat.remove_unused_categories()

        if group_gate:
            counts = sub.obs[groupby].value_counts()

            if compare_groups and len(compare_groups) == 2:
                too_small = [
                    g for g in compare_groups if counts.get(g, 0) < min_cells_per_group
                ]
                if too_small:
                    logging.warning(
                        f"  Skipping {within}='{stratum}': compare group(s) "
                        f"{too_small} have < {min_cells_per_group} cells in '{groupby}'"
                    )
                    continue
            else:
                eligible = _eligible_groups(sub, groupby, min_cells_per_group)
                if len(eligible) < 2:
                    logging.warning(
                        f"  Skipping {within}='{stratum}': fewer than 2 groups with "
                        f">= {min_cells_per_group} cells in '{groupby}'"
                    )
                    continue
                present = [g for g, n in counts.items() if n > 0]
                if len(eligible) < len(present):
                    sub = sub[sub.obs[groupby].isin(eligible)].copy()
                    if hasattr(sub.obs[groupby], "cat"):
                        sub.obs[groupby] = sub.obs[
                            groupby
                        ].cat.remove_unused_categories()

        try:
            stratum_df = single_fn(sub)
        except (ValueError, KeyError) as exc:
            logging.warning(f"  Skipping {within}='{stratum}': {exc}")
            continue
        if stratum_df is None or stratum_df.empty:
            continue

        stratum_df = stratum_df.copy()
        stratum_df["within_col"] = within
        stratum_df["within_value"] = stratum
        stratum_df["n_cells"] = sub.n_obs
        results.append(stratum_df)

    if not results:
        logging.warning(
            f"  No eligible strata for within='{within}'; returning empty results"
        )
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    logging.info(
        f"  Stratified differential completed for {len(results)} of "
        f"{len(strata)} '{within}' strata"
    )
    return combined


# --------------------------------------------------------------------------- #
# Single entrypoint
# --------------------------------------------------------------------------- #
def _resolve_source(adata: ad.AnnData, on: str) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve ``on`` to ``(kind, layer, obsm_layer)`` where kind is ge/obsm."""
    if on in ("gene_expression", "X"):
        return "ge", None, None
    if on in adata.layers:
        return "ge", on, None
    if on in adata.obsm:
        return "obsm", None, on
    raise KeyError(
        f"on='{on}' is not 'gene_expression'/'X', a layer "
        f"({list(adata.layers)}), or an obsm key ({list(adata.obsm)})"
    )


def _normalize_method(method: str) -> str:
    """Normalize the ``t-test`` spelling to the in-module ``ttest`` name."""
    return "ttest" if method == "t-test" else method


def _default_method(kind: str, compare_groups: Optional[List[str]]) -> str:
    """Pick the default engine for a source when ``method`` is unset."""
    if kind == "ge":
        return "wilcoxon"
    return "ttest" if compare_groups else "means"


def _validate_method_for_source(kind: str, method: str) -> None:
    """Raise ``ValueError`` when ``method`` is invalid for the resolved source."""
    if kind == "obsm":
        allowed = _OBSM_METHODS
    else:
        allowed = _GENERIC_METHODS
    if method not in allowed:
        raise ValueError(
            f"method '{method}' is not valid for a {kind} source; "
            f"allowed: {sorted(allowed)}"
        )


def run_differential(
    adata: ad.AnnData,
    groupby: str,
    *,
    on: str = "gene_expression",
    compare_groups: Optional[List[str]] = None,
    within: Optional[str] = None,
    within_subset: Optional[List] = None,
    method: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    formula: Optional[str] = None,
    target_coef: Optional[Union[str, List[str]]] = None,
) -> DifferentialResults:
    """
    Run differential analysis from a single dispatching entrypoint.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by (or the continuous
            predictor for ``spearman`` / continuous regression).
        on: Data source. ``"gene_expression"``/``"X"`` or a layer name selects
            the gene-expression path; an ``adata.obsm`` key selects the
            embedding path. All methods apply to any source.
        compare_groups: Optional list of exactly two groups for a pairwise,
            directional comparison. ``None`` runs the all-groups mode.
        within: Optional ``obs`` column whose categories define strata; the
            analysis is run independently within each.
        within_subset: Optional list restricting which ``within`` categories
            are computed. Requires ``within``.
        method: Statistical engine (any source): ``ttest``, ``wilcoxon``,
            ``spearman``, ``anova``, ``regression``, ``rankby`` or ``means``.
            ``t-test`` is accepted as a spelling of ``ttest``. Defaults to
            ``wilcoxon`` for GE and to ``ttest``/``means`` for obsm depending
            on ``compare_groups``.
        covariates: Optional ``obs`` columns added to the ``regression`` design.
        formula: Optional Patsy right-hand-side formula for ``regression``
            (e.g. ``"C(status) + n_counts"``). Takes precedence over
            ``groupby`` + ``covariates``; requires ``target_coef``.
        target_coef: Coefficient selection for ``regression``: a single term
            name, a list of names, or the literal ``"all"``.

    Returns:
        :class:`DifferentialResults` with a tidy long ``results`` DataFrame.
    """
    if within_subset is not None and within is None:
        raise ValueError("within_subset requires within to be set")

    kind, layer, _obsm_layer = _resolve_source(adata, on)

    resolved_method = _normalize_method(method) if method else _default_method(kind, compare_groups)
    _validate_method_for_source(kind, resolved_method)

    if (covariates or formula) and resolved_method != "regression":
        raise ValueError(
            "covariates/formula are only valid with method='regression'"
        )
    if formula is not None and target_coef is None:
        raise ValueError("target_coef is required when formula is provided")

    spec = StatSpec(
        test=resolved_method,
        groupby=groupby,
        compare_groups=compare_groups,
        covariates=covariates,
        formula=formula,
        target_coef=target_coef,
    )
    group_gate = resolved_method not in _CONTINUOUS_METHODS
    min_cells = 1 if resolved_method == "means" else _MIN_CELLS_PER_GROUP

    def single_fn(sub: ad.AnnData) -> Optional[pd.DataFrame]:
        source = layer if (kind == "ge" and layer is not None) else on
        return _run_generic(sub, source, spec)

    if within is not None:
        results_df = _run_within(
            adata,
            groupby,
            within,
            single_fn=single_fn,
            within_subset=within_subset,
            compare_groups=compare_groups if group_gate else None,
            min_cells_per_group=min_cells,
            group_gate=group_gate,
        )
        return DifferentialResults(
            adata=adata,
            results=results_df,
            source=on,
            method=resolved_method,
        )

    df = single_fn(adata)
    return DifferentialResults(
        adata=adata,
        results=df if df is not None else pd.DataFrame(),
        source=on,
        method=resolved_method,
    )


# --------------------------------------------------------------------------- #
# Save helper (CSV export)
# --------------------------------------------------------------------------- #
def save_differential_results(
    results: DifferentialResults,
    output_dir: Path,
    *,
    groupby: str,
    compare_groups: Optional[List[str]] = None,
    within: Optional[str] = None,
    n_top: int = 100,
) -> None:
    """
    Write a :class:`DifferentialResults` tidy frame to CSV.

    Writes a full results file plus a top-N file. When stratified, filenames
    get a ``_within_{within}`` suffix and the top-N selection is computed per
    stratum (and per group where applicable).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results.results
    if df is None or df.empty:
        logging.warning("  No differential results to save")
        return

    stem = f"de_{results.source}"
    suffix = f"_within_{within}" if within else ""

    if compare_groups and len(compare_groups) == 2:
        cmp_tag = f"_{compare_groups[0]}_vs_{compare_groups[1]}"
    else:
        cmp_tag = f"_all_groups_{groupby}"

    full_file = output_dir / f"{stem}{cmp_tag}{suffix}.csv"
    df.to_csv(full_file, index=False)
    logging.info(f"  Saved differential results to {full_file}")

    df = _rank_for_top(df)
    top_keys = [k for k in ("within_value", "predictor", "group") if k in df.columns]
    top_file = output_dir / f"{stem}_top{n_top}{cmp_tag}{suffix}.csv"
    if top_keys:
        df.groupby(top_keys, sort=False).head(n_top).to_csv(top_file, index=False)
    else:
        df.head(n_top).to_csv(top_file, index=False)
    logging.info(f"  Saved top-{n_top} differential results to {top_file}")


def _rank_for_top(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order rows so a per-scope ``head(n)`` keeps the most relevant features.

    Prefers ascending ``padj`` (then ``pval``); otherwise falls back to the
    magnitude of an available effect/score column.
    """
    sort_within = [k for k in ("within_value", "predictor", "group") if k in df.columns]

    for pcol in ("padj", "pval"):
        if pcol in df.columns:
            return df.sort_values(sort_within + [pcol], na_position="last")

    for ecol in ("mean_difference", "stat", "coef"):
        if ecol in df.columns:
            key = df[ecol].abs()
            order = key.sort_values(ascending=False, na_position="last").index
            return df.loc[order]

    return df
