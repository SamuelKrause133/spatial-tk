"""Tests for argv routing and lazy-loaded image namespace."""

import argparse
import sys
from unittest import mock

import pytest


def test_main_routes_image_namespace_to_image_main():
    """First token `image` must dispatch to `image_main`, not the analysis parser."""
    import spatial_tk.cli as cli

    with mock.patch.object(cli, "image_main") as mock_image_main:
        with mock.patch.object(sys, "argv", ["spatial-tk", "image", "--help"]):
            cli.main()
            mock_image_main.assert_called_once()


def test_create_analysis_parser_exposes_core_commands():
    """Needs analysis stack (spatialdata, etc.) loaded by concat → data_io."""
    pytest.importorskip("spatialdata")

    import spatial_tk.cli as cli

    p = cli.create_parser()
    names: set[str] = set()
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction) and a.choices:
            names |= set(a.choices.keys())
    assert {"concat", "normalize", "cluster", "quantitate", "differential"}.issubset(names)


def test_image_parser_lists_image_subcommands():
    from spatial_tk.commands.image_group import create_image_parser

    p = create_image_parser()
    names: set[str] = set()
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction) and a.choices:
            names |= set(a.choices.keys())
    assert names == {"import-bioformat", "segment", "quantify", "extract"}
