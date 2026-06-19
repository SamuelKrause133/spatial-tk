#!/usr/bin/env python3
"""
Differential analysis core API.

This module owns all differential logic for spatial-tk behind a single
notebook-friendly entrypoint, :func:`run_differential`, which dispatches by
data source:

- **Gene expression** (``on="gene_expression"``/``"X"`` or a layer name):
  cluster markers or pairwise group comparisons via Scanpy ``rank_genes_groups``.
- **obsm embeddings** (``on=<obsm key>``, e.g. MLM/ULM enrichment scores):
  per-feature pairwise t-tests (``method="ttest"``), per-group means
  (``method="means"``), or decoupler association ranking
  (``method="rankby"``, ANOVA/Spearman via :func:`decoupler.tl.rankby_obsm`).

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
from typing import Callable, List, Optional

import anndata as ad
import pandas as pd


@dataclass
class DifferentialResults:
    """Container for the output of :func:`run_differential`."""

    adata: ad.AnnData
    results: pd.DataFrame
    source: str
    method: str
    rank_key: Optional[str] = None


# Scanpy's ``rank_genes_groups`` (and the obsm t-test / ANOVA engines) cannot
# compute statistics for a group with fewer than 2 cells.
_MIN_CELLS_PER_GROUP = 2

_OBSM_METHODS = ("ttest", "means", "rankby")


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
    key_added: Optional[str] = None,
    resume: bool = False,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """
    Run single-shot gene-expression differential expression.

    Two modes:

    - **Mode A (pairwise)**: when ``compare_groups`` is a list of exactly two
      values, ``compare_groups[0]`` is compared against ``compare_groups[1]``
      as reference. The computation runs on a *subset copy* of ``adata`` and
      that subset is returned alongside the results DataFrame.
    - **Mode B (markers)**: when ``compare_groups`` is None, marker genes are
      found for every group in ``groupby``. ``adata`` is mutated in place.

    Results are stored under ``rank_genes_{groupby}`` (or the explicit
    ``key_added``) and ``adata.uns["rank_genes_groups_key"]`` records that key.

    For stratified (``within``) analysis use :func:`run_differential`.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        compare_groups: Optional list of exactly two groups to compare.
        method: Statistical test (``wilcoxon``, ``t-test``, ``logreg``).
        layer: Optional expression layer (default ``None`` uses ``.X``).
        key_added: Optional explicit ``uns`` key; defaults to
            ``rank_genes_{groupby}``.
        resume: If True, skip recomputation when results already exist.

    Returns:
        Tuple of ``(adata, results_df)``. In Mode A ``adata`` is the subset
        copy used for the comparison.
    """
    import scanpy as sc

    rank_key = key_added or f"rank_genes_{groupby}"

    if compare_groups and len(compare_groups) == 2:
        logging.info(
            f"Gene expression DE: comparing {compare_groups[0]} vs {compare_groups[1]}"
        )

        mask = adata.obs[groupby].isin(compare_groups)
        adata_subset = adata[mask].copy()

        if resume and _de_already_computed(adata_subset, rank_key):
            logging.info(f"  DE already computed for {rank_key} (resuming)")
            result_df = sc.get.rank_genes_groups_df(
                adata_subset, group=compare_groups[0], key=rank_key
            )
        else:
            sc.tl.rank_genes_groups(
                adata_subset,
                groupby=groupby,
                groups=[compare_groups[0]],
                reference=compare_groups[1],
                method=method,
                layer=layer,
                use_raw=False,
                key_added=rank_key,
            )
            adata_subset.uns["rank_genes_groups_key"] = rank_key
            result_df = sc.get.rank_genes_groups_df(
                adata_subset, group=compare_groups[0], key=rank_key
            )

        result_df["group1"] = compare_groups[0]
        result_df["group2"] = compare_groups[1]
        return adata_subset, result_df

    logging.info(f"Gene expression DE: finding marker genes for all groups in {groupby}")

    if resume and _de_already_computed(adata, rank_key):
        logging.info(f"  DE already computed for {rank_key} (resuming)")
        result_df = sc.get.rank_genes_groups_df(adata, group=None, key=rank_key)
        return adata, result_df

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        layer=layer,
        use_raw=False,
        key_added=rank_key,
    )
    adata.uns["rank_genes_groups_key"] = rank_key
    result_df = sc.get.rank_genes_groups_df(adata, group=None, key=rank_key)

    n_groups = adata.obs[groupby].nunique()
    logging.info(f"  Differential expression completed for {n_groups} groups")
    return adata, result_df


def _de_already_computed(adata: ad.AnnData, rank_key: str) -> bool:
    """Return True when ``rank_key`` results are already present in ``adata.uns``."""
    return rank_key in adata.uns and adata.uns.get("rank_genes_groups_key") == rank_key


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
) -> Optional[pd.DataFrame]:
    """
    Run single-shot differential analysis on an ``obsm`` embedding.

    Always returns a *tidy long* DataFrame so results compose across strata.

    Engines (``method``):

    - ``"ttest"``: per-feature two-group Welch-style t-test (requires
      ``compare_groups``); columns ``feature, group1, group2, mean_group1,
      mean_group2, mean_difference, stat, pval``.
    - ``"means"``: per-group feature means; columns ``group, feature, mean``.
    - ``"rankby"``: decoupler :func:`decoupler.tl.rankby_obsm` association
      ranking (ANOVA for categorical / Spearman for continuous, BH-corrected);
      columns ``feature, obs_col, stat, pval, padj``.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        obsm_layer: Key in ``adata.obsm`` to analyze.
        compare_groups: Optional list of exactly two groups to compare
            (required for ``method="ttest"``).
        method: One of ``"ttest"``, ``"means"``, ``"rankby"``.

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

    if method == "rankby":
        return _run_obsm_rankby(adata, groupby, obsm_layer)

    feature_names, embedding_df = _obsm_to_dataframe(adata, obsm_layer)
    embedding_df[groupby] = adata.obs[groupby].values

    if method == "ttest":
        if not (compare_groups and len(compare_groups) == 2):
            raise ValueError(
                "method='ttest' requires compare_groups with exactly two groups"
            )
        return _run_obsm_ttest(embedding_df, feature_names, groupby, compare_groups)

    # method == "means"
    logging.info(f"  Calculating mean {obsm_layer} values per group")
    group_means = embedding_df.groupby(groupby, observed=True)[feature_names].mean()
    long = (
        group_means.reset_index()
        .melt(id_vars=groupby, var_name="feature", value_name="mean")
        .rename(columns={groupby: "group"})
    )
    return long


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


def _run_obsm_ttest(
    embedding_df: pd.DataFrame,
    feature_names: List[str],
    groupby: str,
    compare_groups: List[str],
) -> pd.DataFrame:
    """Per-feature two-group t-test, returned as a tidy long DataFrame."""
    from scipy import stats

    logging.info(f"  Comparing {compare_groups[0]} vs {compare_groups[1]}")

    group1_data = embedding_df[embedding_df[groupby] == compare_groups[0]][feature_names]
    group2_data = embedding_df[embedding_df[groupby] == compare_groups[1]][feature_names]

    results = []
    for feature in feature_names:
        g1_vals = group1_data[feature].values
        g2_vals = group2_data[feature].values

        statistic, pval = stats.ttest_ind(g1_vals, g2_vals)
        results.append(
            {
                "feature": feature,
                "group1": compare_groups[0],
                "group2": compare_groups[1],
                "mean_group1": g1_vals.mean(),
                "mean_group2": g2_vals.mean(),
                "mean_difference": g1_vals.mean() - g2_vals.mean(),
                "stat": statistic,
                "pval": pval,
            }
        )

    result_df = pd.DataFrame(results)
    return result_df.sort_values("mean_difference", ascending=False, key=abs)


def _run_obsm_rankby(
    adata: ad.AnnData, groupby: str, obsm_layer: str
) -> pd.DataFrame:
    """Decoupler association ranking of obsm features vs ``groupby``."""
    import decoupler as dc

    logging.info(f"  Ranking {obsm_layer} features by association with {groupby}")
    # uns_key=None returns the DataFrame instead of writing to adata.uns.
    df = dc.tl.rankby_obsm(adata, obsm_layer, uns_key=None, obs_keys=[groupby])
    return df.rename(columns={"obsm": "feature", "obs": "obs_col"})


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
) -> pd.DataFrame:
    """
    Run a differential engine independently within each category of ``within``.

    ``single_fn`` is invoked on a per-stratum subset copy and must return a
    tidy DataFrame (or ``None``/empty to skip). Each stratum's rows are
    annotated with ``within_col``, ``within_value`` and ``n_cells`` and then
    concatenated.

    Strata lacking enough groups (each with at least ``min_cells_per_group``
    cells) for the requested comparison are skipped with a warning. In
    all-groups mode, groups that are too small are dropped from the stratum so
    the remaining comparison stays valid. ``within_subset`` restricts the
    strata that are processed.

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
                    sub.obs[groupby] = sub.obs[groupby].cat.remove_unused_categories()

        stratum_df = single_fn(sub)
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


def run_differential(
    adata: ad.AnnData,
    groupby: str,
    *,
    on: str = "gene_expression",
    compare_groups: Optional[List[str]] = None,
    within: Optional[str] = None,
    within_subset: Optional[List] = None,
    method: Optional[str] = None,
    resume: bool = False,
) -> DifferentialResults:
    """
    Run differential analysis from a single dispatching entrypoint.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        on: Data source. ``"gene_expression"``/``"X"`` or a layer name selects
            the Scanpy gene-expression path; an ``adata.obsm`` key selects the
            embedding path.
        compare_groups: Optional list of exactly two groups for a pairwise,
            directional comparison. ``None`` runs the all-groups mode.
        within: Optional ``obs`` column whose categories define strata; the
            analysis is run independently within each.
        within_subset: Optional list restricting which ``within`` categories
            are computed. Requires ``within``.
        method: Statistical engine. Gene expression: ``wilcoxon`` (default),
            ``t-test``, ``logreg``. obsm: ``ttest``, ``means``, ``rankby``
            (defaults to ``ttest`` when ``compare_groups`` is given, else
            ``means``).
        resume: If True, skip gene-expression recomputation when present
            (non-stratified gene-expression path only).

    Returns:
        :class:`DifferentialResults` with a tidy long ``results`` DataFrame.
        ``rank_key`` is the Scanpy ``uns`` key for the non-stratified
        gene-expression path, otherwise ``None``.
    """
    if within_subset is not None and within is None:
        raise ValueError("within_subset requires within to be set")

    kind, layer, obsm_layer = _resolve_source(adata, on)

    if kind == "ge":
        resolved_method = method or "wilcoxon"
        min_cells = _MIN_CELLS_PER_GROUP

        def single_fn(sub: ad.AnnData) -> pd.DataFrame:
            return run_gene_expression_de(
                sub,
                groupby,
                compare_groups=compare_groups,
                method=resolved_method,
                layer=layer,
                key_added=f"rank_genes_{groupby}__within",
            )[1]
    else:
        resolved_method = method or ("ttest" if compare_groups else "means")
        min_cells = 1 if resolved_method == "means" else _MIN_CELLS_PER_GROUP

        def single_fn(sub: ad.AnnData) -> Optional[pd.DataFrame]:
            return run_obsm_de(
                sub,
                groupby,
                obsm_layer,
                compare_groups=compare_groups,
                method=resolved_method,
            )

    if within is not None:
        results_df = _run_within(
            adata,
            groupby,
            within,
            single_fn=single_fn,
            within_subset=within_subset,
            compare_groups=compare_groups,
            min_cells_per_group=min_cells,
        )
        return DifferentialResults(
            adata=adata,
            results=results_df,
            source=on,
            method=resolved_method,
            rank_key=None,
        )

    if kind == "ge":
        de_adata, df = run_gene_expression_de(
            adata,
            groupby,
            compare_groups=compare_groups,
            method=resolved_method,
            layer=layer,
            resume=resume,
        )
        return DifferentialResults(
            adata=de_adata,
            results=df,
            source=on,
            method=resolved_method,
            rank_key=de_adata.uns.get("rank_genes_groups_key"),
        )

    df = run_obsm_de(
        adata,
        groupby,
        obsm_layer,
        compare_groups=compare_groups,
        method=resolved_method,
    )
    return DifferentialResults(
        adata=adata,
        results=df if df is not None else pd.DataFrame(),
        source=on,
        method=resolved_method,
        rank_key=None,
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

    is_ge = "names" in df.columns
    stem = "de_genes" if is_ge else f"de_{results.source}"
    suffix = f"_within_{within}" if within else ""

    if compare_groups and len(compare_groups) == 2:
        cmp_tag = f"_{compare_groups[0]}_vs_{compare_groups[1]}"
    else:
        cmp_tag = f"_all_groups_{groupby}"

    full_file = output_dir / f"{stem}{cmp_tag}{suffix}.csv"
    df.to_csv(full_file, index=False)
    logging.info(f"  Saved differential results to {full_file}")

    top_keys = [k for k in ("within_value", "group") if k in df.columns]
    top_file = output_dir / f"{stem}_top{n_top}{cmp_tag}{suffix}.csv"
    if top_keys:
        df.groupby(top_keys, sort=False).head(n_top).to_csv(top_file, index=False)
    else:
        df.head(n_top).to_csv(top_file, index=False)
    logging.info(f"  Saved top-{n_top} differential results to {top_file}")
