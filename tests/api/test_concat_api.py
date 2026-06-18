"""
API integration tests for the concat step.

Mirrors tests/functional/test_concat_command.py but exercises the core
``data_io`` API directly instead of the CLI.
"""

import numpy as np
import pandas as pd
import pytest

from spatial_tk.core import data_io
from spatial_tk.utils.helpers import get_table

pytestmark = pytest.mark.api


def test_concat_produces_valid_adata(raw_adata):
    assert raw_adata.n_obs > 0
    assert "sample" in raw_adata.obs.columns
    assert raw_adata.obs["sample"].nunique() >= 1


def test_concat_metadata_columns_present(raw_adata):
    assert "status" in raw_adata.obs.columns
    assert "location" in raw_adata.obs.columns


def test_concat_downsample(pipeline_sample_df):
    sample_df = pipeline_sample_df.iloc[[0]]
    sdata_list = data_io.load_spatial_datasets(sample_df, load_images=False)
    sdata = data_io.concatenate_spatial_data(sdata_list, sample_df)
    adata_full = get_table(sdata)
    assert adata_full is not None and adata_full.n_obs > 0

    rng = np.random.default_rng(0)
    keep = rng.choice(
        adata_full.n_obs, size=int(adata_full.n_obs * 0.5), replace=False
    )
    adata_ds = adata_full[keep]
    assert adata_ds.n_obs < adata_full.n_obs
