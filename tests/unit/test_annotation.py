"""
Unit tests for annotation module.
"""

import pytest
import logging
import numpy as np
import pandas as pd
import anndata as ad
from spatial_tk.core import annotation


def test_load_marker_genes(test_markers_csv):
    """Test loading marker genes from CSV."""
    markers = annotation.load_marker_genes(str(test_markers_csv))
    
    # Check that markers were loaded
    assert isinstance(markers, dict)
    assert len(markers) > 0
    
    # Check structure
    for cell_type, genes in markers.items():
        assert isinstance(cell_type, str)
        assert isinstance(genes, list)
        assert len(genes) > 0


# Differential-expression tests moved to tests/unit/test_differential.py
# (gene-expression DE now lives in spatial_tk.core.differential.run_gene_expression_de)


# ---------------------------------------------------------------------------
# filter_cells_by_obs
# ---------------------------------------------------------------------------

def test_filter_cells_by_obs_basic(mock_adata):
    """filter_cells_by_obs returns correct mask and subset for a valid expression."""
    adata = mock_adata
    # mock_adata has a 'status' column with values 'HIV' and 'NEG'
    mask, adata_sub = annotation.filter_cells_by_obs(adata, "status==HIV")

    expected_count = (adata.obs["status"] == "HIV").sum()
    assert mask.sum() == expected_count
    assert adata_sub.n_obs == expected_count


def test_filter_cells_by_obs_invalid_column(mock_adata):
    """filter_cells_by_obs raises KeyError for a non-existent column."""
    with pytest.raises(KeyError, match="nonexistent_col"):
        annotation.filter_cells_by_obs(mock_adata, "nonexistent_col==foo")


def test_filter_cells_by_obs_no_matches_warns(mock_adata, caplog):
    """filter_cells_by_obs logs a warning when 0 cells match."""
    with caplog.at_level(logging.WARNING, logger="root"):
        mask, adata_sub = annotation.filter_cells_by_obs(mock_adata, "status==NONEXISTENT")

    assert mask.sum() == 0
    assert adata_sub.n_obs == 0
    assert any("0 cells" in msg for msg in caplog.messages)


def test_filter_cells_by_obs_invalid_expr_format(mock_adata):
    """filter_cells_by_obs raises ValueError when '==' is missing."""
    with pytest.raises(ValueError, match="=="):
        annotation.filter_cells_by_obs(mock_adata, "status")


# ---------------------------------------------------------------------------
# run_enrichment_scoring
# ---------------------------------------------------------------------------

def _make_net_df(mock_adata):
    """Return a minimal network DataFrame using genes that exist in mock_adata."""
    genes = list(mock_adata.var_names[:4])
    return pd.DataFrame(
        {
            "source": ["TypeA", "TypeA", "TypeB", "TypeB"],
            "target": genes,
            "weight": [1, 1, 1, 1],
        }
    )


def test_run_enrichment_scoring_mlm_stores_obsm(mock_adata_with_clusters):
    """After MLM scoring, obsm contains the expected key with shape (n_obs, n_sources)."""
    adata = mock_adata_with_clusters
    import scanpy as sc
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    net_df = _make_net_df(adata)
    adata = annotation.run_enrichment_scoring(adata, net_df, score_key="test", method="mlm", tmin=1)

    key = "score_mlm_test"
    assert key in adata.obsm
    scores = adata.obsm[key]
    assert scores.shape[0] == adata.n_obs
    assert scores.shape[1] == 2  # TypeA and TypeB


def test_run_enrichment_scoring_ulm_stores_obsm(mock_adata_with_clusters):
    """After ULM scoring, obsm contains key prefixed with 'score_ulm_'."""
    adata = mock_adata_with_clusters
    import scanpy as sc
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    net_df = _make_net_df(adata)
    adata = annotation.run_enrichment_scoring(adata, net_df, score_key="test", method="ulm", tmin=1)

    key = "score_ulm_test"
    assert key in adata.obsm
    assert adata.obsm[key].shape[0] == adata.n_obs


def test_run_enrichment_scoring_with_mask_fills_nan(mock_adata_with_clusters):
    """When a mask is provided, excluded rows contain NaN in the score matrix."""
    adata = mock_adata_with_clusters
    import scanpy as sc
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    net_df = _make_net_df(adata)

    # Only first 50 cells
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[:50] = True

    adata = annotation.run_enrichment_scoring(
        adata, net_df, score_key="subset", method="mlm", tmin=1, mask=mask
    )

    key = "score_mlm_subset"
    assert key in adata.obsm
    scores = adata.obsm[key]
    # Convert to numpy for easier NaN checking
    if hasattr(scores, "values"):
        scores_np = scores.values
    else:
        scores_np = np.asarray(scores)
    assert scores_np.shape[0] == adata.n_obs
    # Excluded cells (indices 50+) should be NaN
    assert np.all(np.isnan(scores_np[50:]))
    # Included cells (indices 0–49) should NOT all be NaN
    assert not np.all(np.isnan(scores_np[:50]))


# ---------------------------------------------------------------------------
# assign_clusters
# ---------------------------------------------------------------------------

def _adata_with_scores(mock_adata_with_clusters):
    """Return adata with a fake score matrix in obsm ready for assign_clusters."""
    adata = mock_adata_with_clusters
    import scanpy as sc
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    net_df = _make_net_df(adata)
    adata = annotation.run_enrichment_scoring(adata, net_df, score_key="custom", method="mlm", tmin=1)
    return adata


def test_assign_clusters_top_positive_adds_obs_column(mock_adata_with_clusters):
    """assign_clusters with top_positive strategy adds annotation column as categorical."""
    adata = _adata_with_scores(mock_adata_with_clusters)

    adata = annotation.assign_clusters(
        adata,
        score_key="score_mlm_custom",
        cluster_key="leiden_res0p5",
        annotation_key="cell_type_test",
        strategy="top_positive",
    )

    assert "cell_type_test" in adata.obs.columns
    assert str(adata.obs["cell_type_test"].dtype) == "category"
    assert adata.obs["cell_type_test"].notna().all()


def test_assign_clusters_unknown_for_no_positive_stat(mock_adata_with_clusters):
    """Clusters with no positive enrichment score are labelled 'Unknown'."""
    adata = mock_adata_with_clusters

    # Create all-zero / all-negative score matrix → no positive stats
    import pandas as pd
    n_obs = adata.n_obs
    zero_scores = pd.DataFrame(
        np.full((n_obs, 2), -1.0),
        index=adata.obs_names,
        columns=["TypeA", "TypeB"],
    )
    adata.obsm["score_mlm_zeros"] = zero_scores

    adata = annotation.assign_clusters(
        adata,
        score_key="score_mlm_zeros",
        cluster_key="leiden_res0p5",
        annotation_key="test_annotation",
        strategy="top_positive",
    )

    assert "test_annotation" in adata.obs.columns
    # All clusters have no positive stat, so all should be "Unknown"
    assert (adata.obs["test_annotation"] == "Unknown").all()


def test_assign_clusters_invalid_strategy_raises(mock_adata_with_clusters):
    """assign_clusters raises ValueError for an unrecognised strategy."""
    adata = _adata_with_scores(mock_adata_with_clusters)

    with pytest.raises(ValueError, match="nonexistent"):
        annotation.assign_clusters(
            adata,
            score_key="score_mlm_custom",
            cluster_key="leiden_res0p5",
            annotation_key="test_annotation",
            strategy="nonexistent",
        )

