"""Subset cells without importing the analysis stack (scanpy).

Used by ``concat`` so ``pip install .`` + optional ``image`` deps do not
require scanpy at CLI registration time.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np


def downsample_cells(adata: ad.AnnData, fraction: float) -> ad.AnnData:
    """
    Randomly downsample observations to approximately ``fraction`` of the original count.

    Args:
        adata: AnnData object
        fraction: Fraction of cells to keep (0–1].

    Returns:
        Subset AnnData (copy); input is not modified.
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError("Downsample fraction must be between 0 and 1")

    if fraction == 1.0:
        logging.info("No downsampling (fraction=1.0)")
        return adata

    n_cells_original = int(adata.n_obs)
    n_cells_keep = int(n_cells_original * fraction)

    logging.info(
        "Downsampling from %s to %s cells (fraction=%s)",
        n_cells_original,
        n_cells_keep,
        fraction,
    )

    if n_cells_keep >= n_cells_original:
        logging.info("No downsampling applied (fraction rounds to keep all)")
        return adata.copy()

    if n_cells_keep < 1:
        n_cells_keep = 1

    rng = np.random.default_rng()
    ix = rng.choice(n_cells_original, size=n_cells_keep, replace=False)
    ix_sorted = np.sort(ix)
    out = adata[ix_sorted].copy()

    logging.info("Downsampled: %s cells remaining", out.n_obs)

    return out
