"""
Unit tests for spatial_cluster command.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


def _make_args(**overrides):
    defaults = dict(
        input="test.zarr",
        output="output.zarr",
        inplace=False,
        table_key=None,
        cell_type_key="cell_type",
        connectivities_key="spatial_connectivities",
        neighbor_k=None,
        spatial_key="spatial",
        library_key=None,
        output_key="spatial_cluster",
        results_key="spatial_cluster",
        mode="kmeans",
        min_clusters=2,
        max_clusters=20,
        force_n_clusters=None,
        random_state=0,
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=None,
        hdbscan_cluster_selection_epsilon=0.0,
        hdbscan_metric="euclidean",
        hdbscan_allow_single_cluster=False,
        include_self=True,
        normalize_composition=True,
        config=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_spatial_cluster_requires_input():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(input=None)
    with pytest.raises(SystemExit) as exc_info:
        spatial_cluster.main(args)
    assert exc_info.value.code == 1


def test_spatial_cluster_requires_cell_type_key():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(cell_type_key=None)
    with pytest.raises(SystemExit) as exc_info:
        spatial_cluster.main(args)
    assert exc_info.value.code == 1


def test_spatial_cluster_forwards_args_to_core():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(force_n_clusters=7, min_clusters=2, max_clusters=12)

    with patch("spatial_tk.commands.spatial_cluster.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_cluster.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_cluster.save_command_output"), \
         patch("spatial_tk.commands.spatial_cluster.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_cluster.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.build_neighborhood_composition") as mock_build, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.run_spatial_kmeans") as mock_kmeans, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.store_spatial_cluster_results") as mock_store:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obsp = {"spatial_connectivities": MagicMock()}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_build.return_value = {
            "composition": MagicMock(),
            "cell_type_categories": ["A", "B"],
            "neighbor_counts": MagicMock(),
        }
        mock_kmeans.return_value = {
            "mode": "kmeans",
            "best_labels": [0, 1],
            "n_clusters": [2, 3],
            "silhouette_scores": [0.2, 0.1],
            "inertia": [1.0, 0.8],
            "best_n_clusters": 2,
            "best_silhouette_score": 0.2,
            "selection_method": "forced",
            "force_n_clusters": 7,
            "silhouette_best_n_clusters": 2,
            "labels_by_n_clusters": {"2": [0, 1]},
        }
        mock_store.return_value = mock_adata

        spatial_cluster.main(args)

        kwargs = mock_kmeans.call_args.kwargs
        assert kwargs["min_clusters"] == 2
        assert kwargs["max_clusters"] == 12
        assert kwargs["force_n_clusters"] == 7


def test_spatial_cluster_computes_neighbors_when_missing_graph_and_neighbor_k_set():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(connectivities_key="my_connectivities", neighbor_k=9)

    with patch("spatial_tk.commands.spatial_cluster.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_cluster.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_cluster.save_command_output"), \
         patch("spatial_tk.commands.spatial_cluster.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_cluster.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_cluster.spatial_neighbors_core.compute_spatial_neighbors") as mock_compute_neighbors, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.build_neighborhood_composition") as mock_build, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.run_spatial_kmeans") as mock_kmeans, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.store_spatial_cluster_results") as mock_store:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obsp = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_build.return_value = {
            "composition": MagicMock(),
            "cell_type_categories": ["A", "B"],
            "neighbor_counts": MagicMock(),
        }
        mock_kmeans.return_value = {
            "mode": "kmeans",
            "best_labels": [0, 1],
            "n_clusters": [2],
            "silhouette_scores": [0.2],
            "inertia": [1.0],
            "best_n_clusters": 2,
            "best_silhouette_score": 0.2,
            "selection_method": "silhouette_max",
            "force_n_clusters": None,
            "silhouette_best_n_clusters": 2,
            "labels_by_n_clusters": {"2": [0, 1]},
        }
        mock_store.return_value = mock_adata

        spatial_cluster.main(args)
        assert mock_compute_neighbors.called


def test_spatial_cluster_config_merge_applies_values():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(config="config.toml", max_clusters=20, random_state=0)
    config_dict = {
        "spatial_cluster": {
            "max_clusters": 12,
            "random_state": 123,
        }
    }

    with patch("spatial_tk.commands.spatial_cluster.load_config", return_value=config_dict), \
         patch("spatial_tk.commands.spatial_cluster.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_cluster.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_cluster.save_command_output"), \
         patch("spatial_tk.commands.spatial_cluster.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_cluster.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.build_neighborhood_composition") as mock_build, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.run_spatial_kmeans") as mock_kmeans, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.store_spatial_cluster_results") as mock_store:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obsp = {"spatial_connectivities": MagicMock()}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_build.return_value = {
            "composition": MagicMock(),
            "cell_type_categories": ["A", "B"],
            "neighbor_counts": MagicMock(),
        }
        mock_kmeans.return_value = {
            "mode": "kmeans",
            "best_labels": [0, 1],
            "n_clusters": [2],
            "silhouette_scores": [0.2],
            "inertia": [1.0],
            "best_n_clusters": 2,
            "best_silhouette_score": 0.2,
            "selection_method": "silhouette_max",
            "force_n_clusters": None,
            "silhouette_best_n_clusters": 2,
            "labels_by_n_clusters": {"2": [0, 1]},
        }
        mock_store.return_value = mock_adata

        spatial_cluster.main(args)
        kwargs = mock_kmeans.call_args.kwargs
        assert kwargs["max_clusters"] == 12
        assert kwargs["random_state"] == 123


def test_cli_registers_spatial_cluster_subcommand():
    from spatial_tk.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(
        ["spatial_cluster", "--input", "data.zarr", "--inplace", "--cell-type-key", "cell_type"]
    )
    assert args.command == "spatial_cluster"
    assert callable(args.func)


def test_spatial_cluster_hdbscan_mode_dispatches_to_hdbscan():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(mode="hdbscan", force_n_clusters=None)

    with patch("spatial_tk.commands.spatial_cluster.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_cluster.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_cluster.save_command_output"), \
         patch("spatial_tk.commands.spatial_cluster.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_cluster.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.build_neighborhood_composition") as mock_build, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.run_spatial_kmeans") as mock_kmeans, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.run_spatial_hdbscan") as mock_hdbscan, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.store_spatial_cluster_results") as mock_store:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obsp = {"spatial_connectivities": MagicMock()}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_build.return_value = {
            "composition": MagicMock(),
            "cell_type_categories": ["A", "B"],
            "neighbor_counts": MagicMock(),
        }
        mock_hdbscan.return_value = {
            "mode": "hdbscan",
            "best_labels": [0, 1],
            "labels": [0, 1],
            "n_clusters_found": 1,
            "n_noise": 0,
            "noise_fraction": 0.0,
            "silhouette_score": None,
            "hdbscan_params": {},
        }
        mock_store.return_value = mock_adata

        spatial_cluster.main(args)
        assert mock_hdbscan.called
        assert not mock_kmeans.called


def test_spatial_cluster_rejects_force_n_clusters_for_hdbscan():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(mode="hdbscan", force_n_clusters=5)
    with pytest.raises(SystemExit) as exc_info:
        spatial_cluster.main(args)
    assert exc_info.value.code == 1


def test_spatial_cluster_config_merge_applies_mode_and_hdbscan_params():
    from spatial_tk.commands import spatial_cluster

    args = _make_args(config="config.toml", mode="kmeans")
    config_dict = {
        "spatial_cluster": {
            "mode": "hdbscan",
            "hdbscan_min_cluster_size": 9,
            "hdbscan_min_samples": 3,
        }
    }

    with patch("spatial_tk.commands.spatial_cluster.load_config", return_value=config_dict), \
         patch("spatial_tk.commands.spatial_cluster.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_cluster.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_cluster.save_command_output"), \
         patch("spatial_tk.commands.spatial_cluster.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_cluster.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.build_neighborhood_composition") as mock_build, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.run_spatial_hdbscan") as mock_hdbscan, \
         patch("spatial_tk.commands.spatial_cluster.spatial_clustering.store_spatial_cluster_results") as mock_store:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obsp = {"spatial_connectivities": MagicMock()}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata
        mock_build.return_value = {
            "composition": MagicMock(),
            "cell_type_categories": ["A", "B"],
            "neighbor_counts": MagicMock(),
        }
        mock_hdbscan.return_value = {
            "mode": "hdbscan",
            "best_labels": [0, 1],
            "labels": [0, 1],
            "n_clusters_found": 1,
            "n_noise": 0,
            "noise_fraction": 0.0,
            "silhouette_score": None,
            "hdbscan_params": {},
        }
        mock_store.return_value = mock_adata

        spatial_cluster.main(args)
        kwargs = mock_hdbscan.call_args.kwargs
        assert kwargs["min_cluster_size"] == 9
        assert kwargs["min_samples"] == 3

