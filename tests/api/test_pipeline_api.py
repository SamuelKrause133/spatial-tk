"""
End-to-end API pipeline tests.

This is the primary regression guard for programmatic pipelines. It rebuilds
the same sequences as tests/functional/test_full_pipeline.py using only core
Python calls (no subprocesses), ensuring notebook/script pipelines keep working
across refactors.
"""

import pandas as pd
import pytest
from matplotlib.figure import Figure

from spatial_tk.core import (
    annotation,
    clustering,
    data_io,
    differential,
    preprocessing,
    spatial_clustering,
    spatial_neighbors,
    visualization,
)
from spatial_tk.utils.helpers import get_table

pytestmark = pytest.mark.api


@pytest.mark.slow
def test_full_pipeline_api_end_to_end(test_samples_csv, test_markers_csv):
    if not test_samples_csv.exists():
        pytest.skip("Test samples CSV not found")
    if not test_markers_csv.exists():
        pytest.skip("Test markers CSV not found")

    # 1. concat (single sample for speed)
    sample_df = pd.read_csv(test_samples_csv).iloc[[0]]
    sdata_list = data_io.load_spatial_datasets(sample_df, load_images=False)
    sdata = data_io.concatenate_spatial_data(sdata_list, sample_df)
    adata = get_table(sdata)
    assert adata is not None and adata.n_obs > 0
    assert "sample" in adata.obs.columns

    # 2. normalize
    adata = preprocessing.calculate_qc_metrics(adata)
    adata = preprocessing.filter_cells_and_genes(adata, min_genes=10, min_cells=3)
    adata = preprocessing.normalize_and_log(adata)
    adata = preprocessing.select_variable_genes(adata, n_top_genes=500)
    assert "highly_variable" in adata.var.columns

    # 3. cluster
    adata = clustering.run_pca(adata)
    adata = clustering.compute_neighbors_and_umap(adata)
    adata = clustering.cluster_leiden(adata, 0.5, key_added="leiden_res0p5")
    assert "leiden_res0p5" in adata.obs.columns

    # 4. quantitate
    markers = annotation.load_marker_genes(str(test_markers_csv))
    net_df = annotation.markers_dict_to_dataframe(markers)
    adata = annotation.run_enrichment_scoring(
        adata, net_df, score_key="custom", method="mlm", tmin=1
    )
    if "score_mlm_custom" not in adata.obsm:
        pytest.skip("Enrichment scoring produced no obsm key for this panel")

    # 5. assign
    adata = annotation.assign_clusters(
        adata,
        score_key="score_mlm_custom",
        cluster_key="leiden_res0p5",
        annotation_key="cell_type_res0p5",
    )
    assert "cell_type_res0p5" in adata.obs.columns

    # 6. differential (mode B)
    results = differential.run_differential_analysis(adata, groupby="leiden_res0p5")
    assert results.gene_expression is not None

    # 7. spatial neighbors
    adata = spatial_neighbors.compute_spatial_neighbors(
        adata, spatial_key="spatial", n_neighs=6, key_added="spatial"
    )

    # 8. spatial cluster (kmeans)
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        max_clusters=20,
        output_key="spatial_cluster",
        results_key="spatial_cluster",
    )
    assert "spatial_cluster" in adata.obs.columns
    assert "spatial_cluster" in adata.uns

    # 9. visualization
    import matplotlib.pyplot as plt

    plots = visualization.run_roi_visualization(
        adata.obsm["spatial"], adata.obs, view="full"
    )
    assert len(plots) == 1 and isinstance(plots[0].fig, Figure)
    plt.close(plots[0].fig)


@pytest.mark.slow
def test_pipeline_api_group_comparison(assigned_adata):
    # Reuse the staged assigned_adata fixture (HIV + NEG) for the Mode A path.
    if assigned_adata.obs["status"].nunique() < 2:
        pytest.skip("Need at least two status groups for group comparison")

    results = differential.run_differential_analysis(
        assigned_adata.copy(), groupby="status", compare_groups=["HIV", "NEG"]
    )
    assert results.gene_expression is not None
    comparison_df = results.gene_expression
    assert set(comparison_df["group1"].unique()) == {"HIV"}
    assert set(comparison_df["group2"].unique()) == {"NEG"}
