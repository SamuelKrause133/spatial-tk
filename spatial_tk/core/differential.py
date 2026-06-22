#!/usr/bin/env python3
"""
Differential analysis core API.

This module owns all differential-expression logic for spatial-tk:

- Gene-expression differential expression (cluster markers or pairwise group
  comparisons) via Scanpy ``rank_genes_groups``.
- ``obsm`` embedding differential analysis (e.g. MLM/ULM enrichment scores)
  via per-feature Welch t-tests or per-group means.

Functions here are pure compute (no file I/O) and return ``AnnData`` and/or
``pandas.DataFrame`` objects so they can be composed in notebooks and scripts.
Optional ``save_*`` helpers handle CSV export with the historical CLI naming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import anndata as ad
import pandas as pd


@dataclass
class DifferentialResults:
    """Container for the outputs of :func:`run_differential_analysis`."""

    adata: ad.AnnData
    gene_expression: Optional[pd.DataFrame]
    obsm: Optional[pd.DataFrame]
    rank_key: Optional[str]


# --------------------------------------------------------------------------- #
# Gene-expression differential expression
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
    Run gene-expression differential expression.

    Two modes:

    - **Mode A (pairwise)**: when ``compare_groups`` is a list of exactly two
      values, ``compare_groups[0]`` is compared against ``compare_groups[1]``
      as reference. The computation runs on a *subset copy* of ``adata`` and
      that subset is returned alongside the results DataFrame.
    - **Mode B (markers)**: when ``compare_groups`` is None, marker genes are
      found for every group in ``groupby``. ``adata`` is mutated in place.

    Results are always stored under ``rank_genes_{groupby}`` (or the explicit
    ``key_added``) and ``adata.uns["rank_genes_groups_key"]`` records that key.

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
# obsm embedding differential analysis
# --------------------------------------------------------------------------- #
def run_obsm_de(
    adata: ad.AnnData,
    groupby: str,
    obsm_layer: str,
    *,
    compare_groups: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Run differential analysis on an ``obsm`` embedding (e.g. enrichment scores).

    Two modes:

    - **Mode A (pairwise)**: per-feature Welch t-test comparing two groups,
      returning a DataFrame with mean difference, t-statistic and p-value.
    - **Mode B (means)**: per-group feature means, returning a DataFrame
      indexed by group.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        obsm_layer: Key in ``adata.obsm`` to analyze.
        compare_groups: Optional list of exactly two groups to compare.

    Returns:
        Results DataFrame, or ``None`` if ``obsm_layer`` is not present.
    """
    logging.info(f"obsm DE on {obsm_layer}")

    if obsm_layer not in adata.obsm:
        logging.warning(f"  obsm layer '{obsm_layer}' not found. Skipping.")
        return None

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

    embedding_df[groupby] = adata.obs[groupby].values

    if compare_groups and len(compare_groups) == 2:
        from scipy import stats

        logging.info(f"  Comparing {compare_groups[0]} vs {compare_groups[1]}")

        group1_data = embedding_df[embedding_df[groupby] == compare_groups[0]][feature_names]
        group2_data = embedding_df[embedding_df[groupby] == compare_groups[1]][feature_names]

        results = []
        for feature in feature_names:
            g1_vals = group1_data[feature].values
            g2_vals = group2_data[feature].values

            mean_diff = g1_vals.mean() - g2_vals.mean()
            statistic, pval = stats.ttest_ind(g1_vals, g2_vals)

            results.append(
                {
                    "feature": feature,
                    "mean_group1": g1_vals.mean(),
                    "mean_group2": g2_vals.mean(),
                    "mean_difference": mean_diff,
                    "t_statistic": statistic,
                    "pvalue": pval,
                    "group1": compare_groups[0],
                    "group2": compare_groups[1],
                }
            )

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("mean_difference", ascending=False, key=abs)
        return result_df

    logging.info(f"  Calculating mean {obsm_layer} values per group")
    group_means = embedding_df.groupby(groupby)[feature_names].mean()
    return group_means


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_differential_analysis(
    adata: ad.AnnData,
    groupby: str,
    *,
    compare_groups: Optional[List[str]] = None,
    method: str = "wilcoxon",
    layer: Optional[str] = None,
    obsm_layer: Optional[str] = None,
    resume: bool = False,
) -> DifferentialResults:
    """
    Run gene-expression DE and (optionally) ``obsm`` DE in one call.

    Args:
        adata: AnnData object.
        groupby: Column in ``adata.obs`` to group by.
        compare_groups: Optional list of exactly two groups to compare.
        method: Statistical test for gene-expression DE.
        layer: Optional expression layer for gene-expression DE.
        obsm_layer: Optional ``obsm`` key for enrichment-based DE.
        resume: If True, skip gene-expression recomputation when present.

    Returns:
        :class:`DifferentialResults` with the gene-expression DataFrame, the
        optional ``obsm`` DataFrame, the (possibly subset) AnnData, and the
        ``rank_key`` used for gene-expression results.
    """
    de_adata, gene_df = run_gene_expression_de(
        adata,
        groupby,
        compare_groups=compare_groups,
        method=method,
        layer=layer,
        resume=resume,
    )
    rank_key = de_adata.uns.get("rank_genes_groups_key")

    obsm_df = None
    if obsm_layer:
        obsm_df = run_obsm_de(
            adata, groupby, obsm_layer, compare_groups=compare_groups
        )

    return DifferentialResults(
        adata=de_adata,
        gene_expression=gene_df,
        obsm=obsm_df,
        rank_key=rank_key,
    )


# --------------------------------------------------------------------------- #
# Save helpers (CSV export)
# --------------------------------------------------------------------------- #
def save_gene_expression_de_results(
    df: pd.DataFrame,
    output_dir: Path,
    groupby: str,
    compare_groups: Optional[List[str]] = None,
    n_genes: int = 100,
) -> None:
    """Write gene-expression DE results to CSV using the historical naming."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if compare_groups and len(compare_groups) == 2:
        g1, g2 = compare_groups[0], compare_groups[1]
        output_file = output_dir / f"de_genes_{g1}_vs_{g2}.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"  Saved gene DE results to {output_file}")

        top_file = output_dir / f"de_genes_top{n_genes}_{g1}_vs_{g2}.csv"
        df.head(n_genes).to_csv(top_file, index=False)
        return

    output_file = output_dir / f"de_genes_all_groups_{groupby}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"  Saved gene DE results to {output_file}")

    top_file = output_dir / f"de_genes_top{n_genes}_per_group_{groupby}.csv"
    df.groupby("group").head(n_genes).to_csv(top_file, index=False)


def save_obsm_de_results(
    df: pd.DataFrame,
    output_dir: Path,
    obsm_layer: str,
    groupby: str,
    compare_groups: Optional[List[str]] = None,
    n_top: int = 50,
) -> None:
    """Write ``obsm`` DE results to CSV using the historical naming."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if compare_groups and len(compare_groups) == 2:
        g1, g2 = compare_groups[0], compare_groups[1]
        output_file = output_dir / f"de_{obsm_layer}_{g1}_vs_{g2}.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"  Saved obsm DE results to {output_file}")

        top_file = output_dir / f"de_{obsm_layer}_top{n_top}_{g1}_vs_{g2}.csv"
        df.head(n_top).to_csv(top_file, index=False)
        return

    output_file = output_dir / f"mean_{obsm_layer}_per_group_{groupby}.csv"
    df.to_csv(output_file)
    logging.info(f"  Saved obsm group means to {output_file}")

