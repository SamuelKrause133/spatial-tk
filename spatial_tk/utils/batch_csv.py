"""Read batch manifest CSVs with per-command required columns (extra columns ignored)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_batch_manifest(path: Path, required: set[str]) -> tuple[list[dict[str, str]], Path]:
    """
    Load a batch manifest CSV.

    Returns (rows, base_dir) where base_dir is the resolved parent of the manifest
    (used to resolve relative paths in row values). Each row is a dict mapping
    column names to string cell values; columns not in ``required`` may be present
    and are preserved for callers that need them.
    """
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("batch CSV must have a header row")
        columns = {str(c) for c in fieldnames if c is not None}
        missing = required - columns
        if missing:
            raise ValueError(f"batch CSV missing required columns: {sorted(missing)}")

        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {k: ("" if v is None else str(v)) for k, v in raw.items() if k is not None}
            if not any(v.strip() for v in row.values()):
                continue
            rows.append(row)

    if not rows:
        raise ValueError("batch CSV must contain at least one row")

    return rows, manifest.parent


def resolve_manifest_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
