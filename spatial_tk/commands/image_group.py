#!/usr/bin/env python3
"""
Image pipeline CLI namespace (`spatial-tk image ...`).

Heavy optional dependencies (Bio-Formats, Cellpose, etc.) are only imported when
a concrete image subcommand runs — not when this module is loaded or when
`--help` is printed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

_MISSING_MSG = (
    "The 'spatial-tk image' pipeline is optional and not installed in this environment.\n"
    "Use the project image environment:\n"
    "  conda env create -p ./venv_image -f image.env.yaml\n"
    "  ./venv_image/bin/python -m spatial_tk.cli image --help\n"
)


def _missing_subcommand(name: str) -> Callable[[Any], None]:
    def _run(_args: Any) -> None:
        print(
            _MISSING_MSG + f"\n(Subcommand '{name}' is not available — missing modules.)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return _run


def _register_import_bioformat(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import import_bioformat as import_bioformat_mod
    except ImportError:
        p = subparsers.add_parser(
            "import-bioformat",
            help="Convert OME-TIFF / OIR and similar to SpatialData / Zarr (Bio-Formats)",
        )
        p.set_defaults(func=_missing_subcommand("import-bioformat"))
        return

    p = subparsers.add_parser(
        "import-bioformat",
        help=getattr(import_bioformat_mod, "CLI_HELP", "Import microscopy formats via Bio-Formats"),
        description=getattr(
            import_bioformat_mod,
            "CLI_DESCRIPTION",
            "Convert supported microscopy files using Bio-Formats / PIMS.",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    import_bioformat_mod.add_arguments(p)
    p.set_defaults(func=import_bioformat_mod.main)


def _register_segment(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import segment as segment_mod
    except ImportError:
        p = subparsers.add_parser(
            "segment", help="Cell / nucleus segmentation (Cellpose or similar)"
        )
        p.set_defaults(func=_missing_subcommand("segment"))
        return

    p = subparsers.add_parser(
        "segment",
        help=getattr(segment_mod, "CLI_HELP", "Run segmentation on image data"),
        description=getattr(segment_mod, "CLI_DESCRIPTION", "Segment cells or nuclei."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    segment_mod.add_arguments(p)
    p.set_defaults(func=segment_mod.main)


def _register_quantify(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import quantify as quantify_mod
    except ImportError:
        p = subparsers.add_parser(
            "quantify", help="Per-cell / per-object channel quantification"
        )
        p.set_defaults(func=_missing_subcommand("quantify"))
        return

    p = subparsers.add_parser(
        "quantify",
        help=getattr(quantify_mod, "CLI_HELP", "Quantify channels for segmented objects"),
        description=getattr(quantify_mod, "CLI_DESCRIPTION", "Summarize intensities per label."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    quantify_mod.add_arguments(p)
    p.set_defaults(func=quantify_mod.main)


def _register_extract(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import extract as extract_mod
    except ImportError:
        p = subparsers.add_parser(
            "extract", help="Extract per-cell image chips / crops for downstream use"
        )
        p.set_defaults(func=_missing_subcommand("extract"))
        return

    p = subparsers.add_parser(
        "extract",
        help=getattr(extract_mod, "CLI_HELP", "Extract object-center crops from images"),
        description=getattr(extract_mod, "CLI_DESCRIPTION", "Crop chips around segmentations."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    extract_mod.add_arguments(p)
    p.set_defaults(func=extract_mod.main)


def create_image_parser() -> argparse.ArgumentParser:
    """
    Build ``spatial-tk image`` argparse tree. Optional deps are imported only from
    register helpers above when the corresponding command module exists.
    """
    parser = argparse.ArgumentParser(
        prog="spatial-tk image",
        description="Microscopy image import, segmentation, quantification, and chip extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  spatial-tk image import-bioformat --input sample.oir --output sample.zarr\n"
            "  spatial-tk image segment --input sample.zarr --output-mask mask.zarr\n"
            "  spatial-tk image quantify --input sample.zarr --labels-key ...\n"
            "  spatial-tk image extract --input sample.zarr --output crops.zarr\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="image_command", help="Image pipeline steps", required=True)

    _register_import_bioformat(subparsers)
    _register_segment(subparsers)
    _register_quantify(subparsers)
    _register_extract(subparsers)

    return parser


def image_main(argv: list[str] | None = None) -> None:
    """Entry point after ``spatial-tk`` has stripped the leading ``image`` token."""
    if argv is None:
        argv = sys.argv[1:]
    parser = create_image_parser()
    args = parser.parse_args(argv)
    args.func(args)
