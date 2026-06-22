"""
API integration tests for the assign step.

Mirrors the assign step in tests/functional/test_full_pipeline.py but
exercises the core ``annotation`` / ``differential`` API directly.
"""

import pytest

from spatial_tk.core import annotation, differential

pytestmark = pytest.mark.api


def test_assign_clusters_adds_annotation_column(scored_adata):
    adata = scored_adata.copy()
    adata = annotation.assign_clusters(
        adata,
        score_key="score_mlm_custom",
        cluster_key="leiden_res0p5",
        annotation_key="cell_type_res0p5",
        strategy="top_positive",
    )
    assert "cell_type_res0p5" in adata.obs.columns
    assert adata.obs["cell_type_res0p5"].notna().all()


def test_assign_auto_discovers_leiden_keys(assigned_adata):
    leiden_cols = [c for c in assigned_adata.obs.columns if c.startswith("leiden_res")]
    assert leiden_cols
    for col in leiden_cols:
        res_str = col.replace("leiden_res", "")
        assert f"cell_type_res{res_str}" in assigned_adata.obs.columns


def test_assign_clusters_with_de(assigned_adata):
    adata = assigned_adata.copy()
    _, de_df = differential.run_gene_expression_de(adata, "leiden_res0p5")
    assert len(de_df) > 0
    assert {"group", "feature", "padj"} <= set(de_df.columns)


def test_assign_result_matches_cluster_count(assigned_adata):
    n_clusters = assigned_adata.obs["leiden_res0p5"].nunique()
    n_labels = assigned_adata.obs["cell_type_res0p5"].nunique()
    # One label is assigned per cluster (labels may repeat, never exceed clusters).
    assert n_labels <= n_clusters
