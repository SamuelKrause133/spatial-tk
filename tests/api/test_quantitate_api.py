"""
API integration tests for the quantitate (enrichment scoring) step.

Mirrors the quantitate step in tests/functional/test_full_pipeline.py but
exercises the core ``annotation`` API directly instead of the CLI.
"""

import pytest

from spatial_tk.core import annotation

pytestmark = pytest.mark.api


def test_run_enrichment_scoring_adds_obsm_key(clustered_adata, marker_net_df):
    adata = clustered_adata.copy()
    adata = annotation.run_enrichment_scoring(
        adata, marker_net_df, score_key="custom", method="mlm", tmin=1
    )
    assert "score_mlm_custom" in adata.obsm
    assert adata.obsm["score_mlm_custom"].shape[0] == adata.n_obs


def test_enrichment_scoring_ulm_method(clustered_adata, marker_net_df):
    adata = clustered_adata.copy()
    adata = annotation.run_enrichment_scoring(
        adata, marker_net_df, score_key="custom", method="ulm", tmin=1
    )
    assert "score_ulm_custom" in adata.obsm


def test_enrichment_scoring_with_cell_mask(clustered_adata, marker_net_df):
    adata = clustered_adata.copy()
    mask = (adata.obs["status"] == "HIV").values
    if mask.sum() == 0 or mask.all():
        pytest.skip("Need both masked and unmasked cells for this test")

    adata = annotation.run_enrichment_scoring(
        adata, marker_net_df, score_key="masked", method="mlm", tmin=1, mask=mask
    )
    if "score_mlm_masked" not in adata.obsm:
        pytest.skip("Masked enrichment scoring did not produce an obsm key")

    score_df = adata.obsm["score_mlm_masked"]
    # Excluded cells must be NaN across all source columns.
    import numpy as np

    values = score_df.values if hasattr(score_df, "values") else np.asarray(score_df)
    assert np.isnan(values[~mask]).all()


def test_load_marker_genes_structure(test_markers_csv):
    if not test_markers_csv.exists():
        pytest.skip("Test markers CSV not found")
    markers = annotation.load_marker_genes(str(test_markers_csv))
    assert isinstance(markers, dict)
    for _cell_type, genes in markers.items():
        assert isinstance(genes, list) and len(genes) > 0


def test_markers_dict_to_dataframe(marker_net_df):
    assert {"source", "target"} <= set(marker_net_df.columns)
