#!/usr/bin/env python3
"""
Cell type annotation and enrichment scoring functions.

This module handles marker-based cell type annotation using decoupler,
differential expression analysis, and MLM/ULM score calculation for
custom gene lists and multiple built-in pathway/TF resources.
"""

import logging
from typing import Dict, List, Optional, Tuple

from spatial_tk.core.cli_constants import PRESET_RESOURCE_NAMES

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------------
# Strategy registry for cluster assignment
# ---------------------------------------------------------------------------

def _strategy_top_positive(
    acts: ad.AnnData,
    adata: ad.AnnData,
    cluster_key: str,
) -> Dict[str, str]:
    """
    Default assignment strategy: for each cluster pick the cell type with
    the highest positive enrichment statistic (rankby_group).  Clusters with
    no positive stat are labelled "Unknown".

    Args:
        acts: AnnData of enrichment scores (cells × cell_types)
        adata: Full AnnData object (used to read cluster labels)
        cluster_key: obs column with cluster assignments

    Returns:
        Dict mapping cluster label → assigned cell type name
    """
    enr = dc.tl.rankby_group(acts, groupby=cluster_key)

    annotation_dict = (
        enr[enr["stat"] > 0]
        .groupby("group", observed=True)
        .head(1)
        .set_index("group")["name"]
        .to_dict()
    )

    # Clusters with no positive score get "Unknown"
    for cluster in adata.obs[cluster_key].unique():
        if cluster not in annotation_dict:
            annotation_dict[cluster] = "Unknown"

    return annotation_dict


def _strategy_threshold(
    acts: ad.AnnData,
    adata: ad.AnnData,
    cluster_key: str,
    threshold: float = 0.0,
) -> Dict[str, str]:
    """
    Stub: assign the top cell type per cluster only when its mean score
    exceeds *threshold*; otherwise assign "Unknown".
    """
    raise NotImplementedError(
        "The 'threshold' strategy is not yet implemented. "
        "Use 'top_positive' for now."
    )


def _strategy_top_n_vote(
    acts: ad.AnnData,
    adata: ad.AnnData,
    cluster_key: str,
    n: int = 3,
) -> Dict[str, str]:
    """
    Stub: consensus vote among top-N scoring cell types per cluster.
    """
    raise NotImplementedError(
        "The 'top_n_vote' strategy is not yet implemented. "
        "Use 'top_positive' for now."
    )


STRATEGY_REGISTRY: Dict = {
    "top_positive": _strategy_top_positive,
    "threshold": _strategy_threshold,
    "top_n_vote": _strategy_top_n_vote,
}


# ---------------------------------------------------------------------------
# Cell filtering
# ---------------------------------------------------------------------------

def filter_cells_by_obs(
    adata: ad.AnnData,
    expr: str,
) -> Tuple[np.ndarray, ad.AnnData]:
    """
    Parse a simple equality expression and return a boolean mask + subset.

    Args:
        adata: AnnData object
        expr: Expression of the form "column==value", e.g. "cell_type==Fibroblast"

    Returns:
        (mask, adata_sub) where mask is a boolean array of length n_obs
        and adata_sub is adata[mask].

    Raises:
        ValueError: If expr does not contain exactly one "==" separator.
        KeyError: If the referenced column does not exist in adata.obs.
    """
    if "==" not in expr:
        raise ValueError(
            f"filter_cells_by_obs: expression must contain '==', got: {expr!r}"
        )

    col, val = expr.split("==", 1)
    col = col.strip()
    val = val.strip()

    if col not in adata.obs.columns:
        raise KeyError(
            f"filter_cells_by_obs: column {col!r} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    mask = (adata.obs[col] == val).values

    n_match = mask.sum()
    if n_match == 0:
        logging.warning(
            f"filter_cells_by_obs: filter '{expr}' matched 0 cells. "
            "Scores will be all NaN."
        )
    else:
        logging.info(
            f"filter_cells_by_obs: filter '{expr}' matched {n_match} / {adata.n_obs} cells"
        )

    adata_sub = adata[mask].copy()
    return mask, adata_sub


# ---------------------------------------------------------------------------
# Preset resource loading
# ---------------------------------------------------------------------------

def load_preset_resource(
    name: str,
    panglao_min_sensitivity: float = 0.5,
    panglao_canonical_only: bool = True,
    organism: str = "human",
) -> pd.DataFrame:
    """
    Load a named decoupler built-in gene-set resource as a DataFrame.

    Supported names: panglao, hallmark, collectri, dorothea, progeny.

    Args:
        name: Resource name (case-insensitive match against PRESET_RESOURCE_NAMES)
        panglao_min_sensitivity: Sensitivity threshold for PanglaoDB filtering
        panglao_canonical_only: Only use canonical PanglaoDB markers
        organism: Organism name (default: "human")

    Returns:
        DataFrame with at least columns "source" and "target" in decoupler format.

    Raises:
        ValueError: If name is not a recognised preset resource.
    """
    name_lower = name.lower()

    if name_lower == "panglao":
        return get_panglao_markers(
            organism=organism,
            min_sensitivity=panglao_min_sensitivity,
            canonical_only=panglao_canonical_only,
        )
    elif name_lower == "hallmark":
        logging.info(f"Loading Hallmark gene sets (organism={organism})")
        return dc.op.hallmark(organism=organism)
    elif name_lower == "collectri":
        logging.info(f"Loading CollectRI TF regulons (organism={organism})")
        return dc.op.collectri(organism=organism)
    elif name_lower == "dorothea":
        logging.info(f"Loading DoRothEA TF regulons (organism={organism})")
        return dc.op.dorothea(organism=organism)
    elif name_lower == "progeny":
        logging.info(f"Loading PROGENy pathway gene sets (organism={organism})")
        return dc.op.progeny(organism=organism)
    else:
        raise ValueError(
            f"Unknown preset resource: {name!r}. "
            f"Supported names: {PRESET_RESOURCE_NAMES}"
        )


# ---------------------------------------------------------------------------
# Core enrichment scoring
# ---------------------------------------------------------------------------

def run_enrichment_scoring(
    adata: ad.AnnData,
    net_df: pd.DataFrame,
    score_key: str,
    method: str = "mlm",
    tmin: int = 2,
    mask: Optional[np.ndarray] = None,
) -> ad.AnnData:
    """
    Run MLM or ULM enrichment scoring and store the result in adata.obsm.

    When *mask* is provided, scoring is performed only on the masked subset
    and the results are written back into the full adata (NaN for excluded
    cells).

    Args:
        adata: Full AnnData object (normalised expression in .X)
        net_df: Gene-set network in decoupler format (source, target[, weight])
        score_key: Key suffix; result stored at obsm[f"score_{method}_{score_key}"]
        method: "mlm" or "ulm"
        tmin: Minimum number of targets per source
        mask: Optional boolean array of length n_obs; if provided, scoring
              runs on adata[mask] and is merged back with NaN fill.

    Returns:
        adata with obsm key added/updated in-place (also returned for chaining).

    Raises:
        ValueError: If method is not "mlm" or "ulm".
    """
    if method not in ("mlm", "ulm"):
        raise ValueError(f"method must be 'mlm' or 'ulm', got {method!r}")

    obsm_key = f"score_{method}_{score_key}"
    raw_obsm_key = f"score_{method}"  # decoupler always writes to this key

    if mask is not None:
        adata_run = adata[mask].copy()
    else:
        adata_run = adata

    logging.info(
        f"Running {method.upper()} scoring on {adata_run.n_obs} cells "
        f"with {len(net_df)} network entries (score_key={score_key!r})"
    )

    try:
        if method == "mlm":
            dc.mt.mlm(data=adata_run, net=net_df, verbose=False, tmin=tmin)
        else:
            dc.mt.ulm(data=adata_run, net=net_df, verbose=False, tmin=tmin)
    except Exception as exc:
        logging.warning(f"  Enrichment scoring failed: {exc}")
        return adata

    scores = adata_run.obsm[raw_obsm_key]

    if mask is not None:
        # Build a full-sized DataFrame initialised to NaN, fill in scored rows
        if isinstance(scores, pd.DataFrame):
            full = pd.DataFrame(
                np.nan,
                index=adata.obs_names,
                columns=scores.columns,
                dtype=float,
            )
            full.loc[adata_run.obs_names] = scores.values
        else:
            full = np.full((adata.n_obs, scores.shape[1]), np.nan, dtype=float)
            full[mask] = scores
        adata.obsm[obsm_key] = full
    else:
        adata.obsm[obsm_key] = scores

    n_sources = scores.shape[1] if hasattr(scores, "shape") and len(scores.shape) > 1 else 1
    logging.info(f"  Stored {n_sources} source scores at obsm['{obsm_key}']")

    return adata


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------

def assign_clusters(
    adata: ad.AnnData,
    score_key: str,
    cluster_key: str,
    annotation_key: str,
    strategy: str = "top_positive",
) -> ad.AnnData:
    """
    Assign a cell type label to each cluster based on enrichment scores.

    Args:
        adata: AnnData object with scores in obsm[score_key]
        score_key: Full obsm key produced by run_enrichment_scoring
                   (e.g. "score_mlm_custom")
        cluster_key: obs column containing cluster assignments
        annotation_key: obs column name to write labels into
        strategy: Assignment strategy name from STRATEGY_REGISTRY

    Returns:
        adata with annotation_key column added to obs.

    Raises:
        ValueError: If strategy is not in STRATEGY_REGISTRY or score_key missing.
    """
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Available strategies: {list(STRATEGY_REGISTRY)}"
        )

    if score_key not in adata.obsm:
        raise ValueError(
            f"Score key {score_key!r} not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    logging.info(
        f"Assigning clusters using strategy={strategy!r}, "
        f"cluster_key={cluster_key!r}, score_key={score_key!r}"
    )

    acts = dc.pp.get_obsm(adata, score_key)

    strategy_fn = STRATEGY_REGISTRY[strategy]
    annotation_dict = strategy_fn(acts, adata, cluster_key)

    adata.obs[annotation_key] = adata.obs[cluster_key].map(annotation_dict)
    adata.obs[annotation_key] = adata.obs[annotation_key].astype("category")

    annotation_counts = adata.obs[annotation_key].value_counts()
    logging.info("Cell type assignment summary:")
    for cell_type, count in annotation_counts.items():
        logging.info(f"  {cell_type}: {count} cells")

    return adata


# ---------------------------------------------------------------------------
# Legacy / convenience functions (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_marker_genes(marker_path: str) -> Dict[str, List[str]]:
    """
    Load marker genes from CSV file.

    Args:
        marker_path: Path to CSV file with columns: cell_type, gene

    Returns:
        Dictionary mapping cell type to list of marker genes
    """
    logging.info(f"Loading marker genes from {marker_path}")

    df = pd.read_csv(marker_path)

    if not all(col in df.columns for col in ["cell_type", "gene"]):
        raise ValueError("Marker CSV must have 'cell_type' and 'gene' columns")

    markers = df.groupby("cell_type")["gene"].apply(list).to_dict()

    total_markers = sum(len(genes) for genes in markers.values())
    logging.info(f"Loaded {len(markers)} cell types with {total_markers} total marker genes")

    return markers


def markers_dict_to_dataframe(markers: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Convert a {cell_type: [gene, ...]} dict to decoupler network format.

    Returns:
        DataFrame with columns: source, target, weight (all weights = 1)
    """
    rows = [
        {"source": cell_type, "target": gene}
        for cell_type, genes in markers.items()
        for gene in genes
    ]
    df = pd.DataFrame(rows)
    df["weight"] = 1
    return df


def get_panglao_markers(
    organism: str = "human",
    min_sensitivity: float = 0.5,
    canonical_only: bool = True,
) -> pd.DataFrame:
    """
    Get PanglaoDB markers with filtering.

    Args:
        organism: Organism name ('human' or 'mouse')
        min_sensitivity: Minimum sensitivity threshold (0-1)
        canonical_only: If True, only use canonical markers

    Returns:
        DataFrame with columns: source (cell_type), target (gene)
    """
    logging.info(
        f"Loading PanglaoDB markers (organism={organism}, min_sensitivity={min_sensitivity})"
    )

    markers = dc.op.resource("PanglaoDB", organism=organism)

    filters = markers[organism].astype(bool)
    if canonical_only:
        filters &= markers["canonical_marker"].astype(bool)
    filters &= markers[f"{organism}_sensitivity"].astype(float) > min_sensitivity

    markers = markers[filters]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
    markers = markers.rename(columns={"cell_type": "source", "genesymbol": "target"})

    logging.info(f"  Filtered to {len(markers)} PanglaoDB markers")
    return markers[["source", "target"]]


def annotate_with_markers(
    adata: ad.AnnData,
    markers: Dict[str, List[str]],
    cluster_key: str = "leiden",
    annotation_key: str = "cell_type",
    resume: bool = False,
    tmin: int = 2,
) -> ad.AnnData:
    """
    Annotate clusters with cell types based on marker gene expression using
    decoupler's multivariate linear model (MLM) approach.

    This is kept for backward compatibility; internally it delegates to
    run_enrichment_scoring and assign_clusters.

    Args:
        adata: AnnData object
        markers: Dictionary mapping cell type to list of marker genes
        cluster_key: Key in adata.obs containing cluster assignments
        annotation_key: Key name for storing cell type annotations
        resume: If True, skip if annotation already exists
        tmin: Minimum number of targets per source (default: 2)

    Returns:
        AnnData object with cell type annotations added
    """
    if resume and annotation_key in adata.obs.columns:
        logging.info("Cell type annotation already exists (resuming)")
        return adata

    # Log coverage
    all_marker_genes = {g for genes in markers.values() for g in genes}
    missing = all_marker_genes - set(adata.var_names)
    if missing:
        logging.info(f"Note: {len(missing)} marker genes not found in dataset")

    net_df = markers_dict_to_dataframe(markers)

    score_key = annotation_key  # re-use annotation_key as score_key suffix
    adata = run_enrichment_scoring(adata, net_df, score_key=score_key, method="mlm", tmin=tmin)

    obsm_key = f"score_mlm_{score_key}"
    adata = assign_clusters(
        adata,
        score_key=obsm_key,
        cluster_key=cluster_key,
        annotation_key=annotation_key,
        strategy="top_positive",
    )

    return adata


def calculate_mlm_scores(
    adata: ad.AnnData,
    use_panglao: bool = True,
    panglao_min_sensitivity: float = 0.5,
    tmin: int = 5,
    resume: bool = False,
) -> ad.AnnData:
    """
    Pre-calculate MLM scores for multiple decoupler resources.

    Resources include:
    - hallmark: Hallmark gene sets
    - collectri: Transcription factor regulons
    - dorothea: TF activity inference
    - progeny: Pathway activity
    - PanglaoDB: Cell type markers (optional, filtered)

    Scores are stored in adata.obsm[f'score_mlm_{resource}']

    Args:
        adata: AnnData object with normalized data
        use_panglao: If True, include PanglaoDB markers
        panglao_min_sensitivity: Minimum sensitivity for PanglaoDB markers
        tmin: Minimum number of targets per source (default: 5)
        resume: If True, skip resources that already have scores

    Returns:
        AnnData object with MLM scores added to obsm
    """
    logging.info("Calculating MLM scores for pathway/TF resources")

    resource_names = ["hallmark", "collectri", "dorothea", "progeny"]
    if use_panglao:
        resource_names.append("panglao")

    for name in resource_names:
        obsm_key = f"score_mlm_{name}"
        if resume and obsm_key in adata.obsm:
            logging.info(f"  MLM scores for {name} already calculated (resuming)")
            continue

        try:
            net_df = load_preset_resource(
                name,
                panglao_min_sensitivity=panglao_min_sensitivity,
            )
            adata = run_enrichment_scoring(adata, net_df, score_key=name, method="mlm", tmin=tmin)
        except Exception as exc:
            logging.warning(f"  Failed to calculate MLM for {name}: {exc}")

    logging.info("MLM score calculation complete")
    return adata


def run_differential_expression(
    adata: ad.AnnData,
    cluster_key: str,
    method: str = "wilcoxon",
    resume: bool = False,
) -> ad.AnnData:
    """
    Run differential expression analysis to find marker genes for each cluster.

    Args:
        adata: AnnData object
        cluster_key: Key in adata.obs containing cluster assignments
        method: Statistical test to use (default: wilcoxon)
        resume: If True, skip if differential expression already computed

    Returns:
        AnnData object with differential expression results added
    """
    rank_key = f"rank_genes_{cluster_key}"

    if resume and "rank_genes_groups" in adata.uns and adata.uns.get("rank_genes_groups_key") == rank_key:
        logging.info(f"Differential expression already computed for {cluster_key} (resuming)")
        return adata

    logging.info(f"Running differential expression analysis for {cluster_key}")

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method=method,
        use_raw=False,
        key_added=rank_key,
        layer=None,
    )

    adata.uns["rank_genes_groups_key"] = rank_key

    n_clusters = adata.obs[cluster_key].nunique()
    logging.info(f"  Differential expression completed for {n_clusters} clusters")

    return adata


def save_differential_expression_results(
    adata: ad.AnnData,
    cluster_key: str,
    output_dir,
    n_genes: int = 100,
) -> None:
    """
    Save differential expression results to CSV files.

    Args:
        adata: AnnData object with differential expression results
        cluster_key: Key in adata.obs containing cluster assignments
        output_dir: Directory to save output files
        n_genes: Number of top genes to save per cluster
    """
    rank_key = f"rank_genes_{cluster_key}"

    if rank_key not in adata.uns:
        logging.warning(f"  No differential expression results found for {cluster_key}")
        return

    logging.info(f"  Saving differential expression results for {cluster_key}")

    result = sc.get.rank_genes_groups_df(adata, group=None, key=rank_key)

    de_dir = output_dir / "differential_expression"
    de_dir.mkdir(exist_ok=True)

    res_str = cluster_key.replace("leiden_res", "")

    all_results_path = de_dir / f"deg_all_clusters_res{res_str}.csv"
    result.to_csv(all_results_path, index=False)
    logging.info(f"    Saved all DE genes to {all_results_path}")

    top_results_path = de_dir / f"deg_top{n_genes}_per_cluster_res{res_str}.csv"
    top_result = result.groupby("group").head(n_genes)
    top_result.to_csv(top_results_path, index=False)
    logging.info(f"    Saved top {n_genes} DE genes per cluster to {top_results_path}")
