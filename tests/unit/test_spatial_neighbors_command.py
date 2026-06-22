"""
Unit tests for the spatial_neighbors command.
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
        spatial_key="spatial",
        library_key=None,
        library_id=None,
        coord_type=None,
        n_neighs=6,
        radius=None,
        transform="none",
        key_added="spatial",
        config=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_spatial_neighbors_passes_arguments_to_core():
    from spatial_tk.commands import spatial_neighbors

    args = _make_args(
        spatial_key="my_spatial",
        library_key="sample",
        coord_type="generic",
        n_neighs=8,
        radius="10,20",
        transform="cosine",
        key_added="custom",
    )

    with patch("spatial_tk.commands.spatial_neighbors.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_neighbors.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_neighbors.save_command_output"), \
         patch("spatial_tk.commands.spatial_neighbors.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_neighbors.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_neighbors.spatial_neighbors_core.compute_spatial_neighbors") as mock_compute:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obs = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata

        spatial_neighbors.main(args)

        assert mock_compute.called
        kwargs = mock_compute.call_args.kwargs
        assert kwargs["spatial_key"] == "my_spatial"
        assert kwargs["library_key"] == "sample"
        assert kwargs["coord_type"] == "generic"
        assert kwargs["n_neighs"] == 8
        assert kwargs["radius"] == (10.0, 20.0)
        assert kwargs["transform"] == "cosine"
        assert kwargs["key_added"] == "custom"


def test_spatial_neighbors_library_id_creates_temp_library_column():
    from spatial_tk.commands import spatial_neighbors

    args = _make_args(library_id="sample_a")

    with patch("spatial_tk.commands.spatial_neighbors.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_neighbors.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_neighbors.save_command_output"), \
         patch("spatial_tk.commands.spatial_neighbors.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_neighbors.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_neighbors.spatial_neighbors_core.compute_spatial_neighbors") as mock_compute:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obs = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata

        spatial_neighbors.main(args)

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["library_key"] == "__spatial_tk_library_id"
        assert mock_adata.obs["__spatial_tk_library_id"] == "sample_a"


def test_spatial_neighbors_requires_input():
    from spatial_tk.commands import spatial_neighbors

    args = _make_args(input=None)
    with pytest.raises(SystemExit) as exc_info:
        spatial_neighbors.main(args)
    assert exc_info.value.code == 1


def test_spatial_neighbors_invalid_radius_exits():
    from spatial_tk.commands import spatial_neighbors

    args = _make_args(radius="10,20,30")

    with patch("spatial_tk.commands.spatial_neighbors.Path") as mock_path_cls:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj

        with pytest.raises(SystemExit) as exc_info:
            spatial_neighbors.main(args)

    assert exc_info.value.code == 1


def test_spatial_neighbors_table_key_forwarded():
    from spatial_tk.commands import spatial_neighbors

    args = _make_args(table_key="rna_table")

    with patch("spatial_tk.commands.spatial_neighbors.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_neighbors.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_neighbors.save_command_output") as mock_save_output, \
         patch("spatial_tk.commands.spatial_neighbors.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_neighbors.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_neighbors.spatial_neighbors_core.compute_spatial_neighbors"):
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obs = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata

        spatial_neighbors.main(args)

        mock_get_table.assert_called_once_with(mock_sdata, table_key="rna_table")
        mock_save_output.assert_called_once_with(
            mock_adata,
            mock_path_obj,
            mock_path_obj,
            inplace=False,
            table_key="rna_table",
        )


def test_spatial_neighbors_config_merge_applies_values():
    from spatial_tk.commands import spatial_neighbors

    args = _make_args(
        config="config.toml",
        n_neighs=6,
        transform="none",
    )
    config_dict = {
        "spatial_neighbors": {
            "n_neighs": 12,
            "transform": "spectral",
        }
    }

    with patch("spatial_tk.commands.spatial_neighbors.load_config", return_value=config_dict), \
         patch("spatial_tk.commands.spatial_neighbors.Path") as mock_path_cls, \
         patch("spatial_tk.commands.spatial_neighbors.load_existing_spatial_data") as mock_load, \
         patch("spatial_tk.commands.spatial_neighbors.save_command_output"), \
         patch("spatial_tk.commands.spatial_neighbors.get_output_path") as mock_out, \
         patch("spatial_tk.commands.spatial_neighbors.get_table") as mock_get_table, \
         patch("spatial_tk.commands.spatial_neighbors.spatial_neighbors_core.compute_spatial_neighbors") as mock_compute:
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj
        mock_out.return_value = mock_path_obj

        mock_sdata = MagicMock()
        mock_adata = MagicMock()
        mock_adata.obs = {}
        mock_load.return_value = mock_sdata
        mock_get_table.return_value = mock_adata

        spatial_neighbors.main(args)

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["n_neighs"] == 12
        assert kwargs["transform"] == "spectral"


def test_cli_registers_spatial_neighbors_subcommand():
    from spatial_tk.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["spatial_neighbors", "--input", "data.zarr", "--inplace"])
    assert args.command == "spatial_neighbors"
    assert callable(args.func)


def test_parse_radius_single_value():
    from spatial_tk.core import spatial_neighbors as spatial_neighbors_core

    assert spatial_neighbors_core.parse_radius("42") == 42.0
