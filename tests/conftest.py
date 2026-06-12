"""
Pytest fixtures for spatial_tk tests.
"""

import logging
import os
import subprocess
import sys
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import spatialdata as sd
import shutil
import gc


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test(request):
    """Automatically clean up after each test to free disk space."""
    yield
    # Force garbage collection after each test
    gc.collect()


def _fast_samples_csv_path() -> Path:
    """Resolve the fast-tier samples CSV path, honoring env override."""
    override = os.getenv("SPATIAL_TK_FAST_SAMPLES_CSV")
    if override:
        return Path(override)
    return Path(__file__).parent / "test_data" / "test_samples_fast.csv"


def _full_samples_csv_path() -> Path:
    """Resolve the full-tier samples CSV path (used to look up source zarrs)."""
    override = os.getenv("SPATIAL_TK_FULL_SAMPLES_CSV")
    if override:
        return Path(override)
    return Path(__file__).parent / "test_data" / "test_samples_full.csv"


@pytest.fixture(scope="session", autouse=True)
def _materialize_fast_roi_fixtures():
    """
    For the fast test tier, ensure tests/test_data/rois/ contains every ROI zarr
    listed in test_samples_fast.csv. Missing ROIs are generated once per session
    from the matching full-tier source zarrs via generate_roi_subsets.py.

    If source zarrs are unavailable, generation is skipped and downstream tests
    fall back to their existing "Test samples CSV not found" skip behavior.
    """
    tier = os.getenv("SPATIAL_TK_TEST_TIER", "fast").strip().lower()
    if tier != "fast":
        return

    fast_csv = _fast_samples_csv_path()
    if not fast_csv.exists():
        return

    try:
        fast_df = pd.read_csv(fast_csv)
    except Exception:
        return

    if "path" not in fast_df.columns or "sample" not in fast_df.columns:
        return

    repo_root = Path(__file__).resolve().parent.parent
    test_data_dir = Path(__file__).parent / "test_data"
    rois_dir = test_data_dir / "rois"

    def _resolve(roi_path: str) -> Path:
        p = Path(roi_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    fast_df["_resolved_path"] = fast_df["path"].map(_resolve)
    missing = fast_df[~fast_df["_resolved_path"].map(Path.exists)]
    if missing.empty:
        return

    full_csv = _full_samples_csv_path()
    if not full_csv.exists():
        logging.warning(
            "Fast ROI fixtures missing and %s not available; tests that require "
            "tests/test_data/rois/ will be skipped or fail.",
            full_csv,
        )
        return

    try:
        full_df = pd.read_csv(full_csv)
    except Exception:
        return

    full_lookup = {row["sample"]: row for _, row in full_df.iterrows()}

    # Group missing ROIs by their parent sample prefix (e.g. "Drexel-Neg_roi_03" -> "Drexel-Neg").
    parent_samples: dict[str, dict] = {}
    for _, roi_row in missing.iterrows():
        roi_sample = str(roi_row["sample"])
        parent = roi_sample.rsplit("_roi_", 1)[0]
        info = parent_samples.setdefault(
            parent,
            {
                "status": str(roi_row.get("status", "") or ""),
                "location": str(roi_row.get("location", "") or ""),
                "roi_names": set(),
            },
        )
        info["roi_names"].add(roi_sample)

    generator_script = test_data_dir / "generate_roi_subsets.py"
    if not generator_script.exists():
        logging.warning("ROI generator script not found at %s", generator_script)
        return

    rois_dir.mkdir(parents=True, exist_ok=True)

    for parent_sample, info in parent_samples.items():
        full_row = full_lookup.get(parent_sample)
        if full_row is None:
            logging.warning(
                "No entry for parent sample %s in %s; cannot generate ROI fixtures.",
                parent_sample,
                full_csv,
            )
            continue

        source_path = Path(str(full_row["path"]))
        if not source_path.exists():
            logging.warning(
                "Source zarr for %s not found at %s; skipping ROI generation.",
                parent_sample,
                source_path,
            )
            continue

        n_rois = max(len(info["roi_names"]), 5)
        cmd = [
            sys.executable,
            str(generator_script),
            "--input-zarr", str(source_path),
            "--output-dir", str(test_data_dir),
            "--sample-name", parent_sample,
            "--status", info["status"],
            "--location", info["location"],
            "--n-rois", str(n_rois),
            "--overwrite",
        ]
        logging.info("Generating fast ROI fixtures for %s: %s", parent_sample, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logging.warning(
                "ROI generation failed for %s (exit %s); downstream tests may skip.",
                parent_sample,
                exc.returncode,
            )


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def test_samples_csv(test_data_dir):
    """
    Return path to samples CSV for the selected test tier.

    Tiers:
      - fast (default): in-repo ROI fixtures
      - full: out-of-repo full-size zarr fixtures
    """
    tier = os.getenv("SPATIAL_TK_TEST_TIER", "fast").strip().lower()

    if tier == "full":
        override = os.getenv("SPATIAL_TK_FULL_SAMPLES_CSV")
        return Path(override) if override else test_data_dir / "test_samples_full.csv"

    override = os.getenv("SPATIAL_TK_FAST_SAMPLES_CSV")
    return Path(override) if override else test_data_dir / "test_samples_fast.csv"


@pytest.fixture
def test_markers_csv(test_data_dir):
    """Return path to test markers CSV."""
    return test_data_dir / "test_markers.csv"


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for unit tests."""
    n_obs = 100
    n_vars = 300  # Increased to avoid scanpy QC issues with percent_top (needs >200)
    
    # Create random expression matrix
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars)).astype(np.float32)
    
    # Create obs DataFrame
    obs = pd.DataFrame({
        'sample': np.random.choice(['sample1', 'sample2'], n_obs),
        'status': np.random.choice(['HIV', 'NEG'], n_obs),
        'location': np.random.choice(['Drexel', 'OSU'], n_obs)
    })
    obs.index = [f'cell_{i}' for i in range(n_obs)]
    
    # Create var DataFrame with some real gene names for testing annotation
    # Include marker genes that will be used in tests
    real_genes = ['CD3D', 'CD3E', 'MS4A1', 'CD19', 'CD68', 'CD14']
    generic_genes = [f'gene_{i}' for i in range(n_vars - len(real_genes))]
    all_genes = real_genes + generic_genes
    
    var = pd.DataFrame({
        'gene_name': all_genes
    })
    var.index = all_genes
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return adata


@pytest.fixture
def mock_adata_with_clusters(mock_adata):
    """Create a mock AnnData object with clustering results."""
    # Add PCA
    np.random.seed(42)
    mock_adata.obsm['X_pca'] = np.random.randn(mock_adata.n_obs, 50)
    
    # Add UMAP
    mock_adata.obsm['X_umap'] = np.random.randn(mock_adata.n_obs, 2)
    
    # Add clustering - ensure we have multiple clusters by design
    # Divide cells into 5 groups deterministically
    n_obs = mock_adata.n_obs
    cluster_labels = np.array([str(i % 5) for i in range(n_obs)])
    mock_adata.obs['leiden_res0p5'] = pd.Categorical(cluster_labels)
    
    return mock_adata


@pytest.fixture
def mock_markers():
    """Return a mock markers dictionary."""
    return {
        'T cells': ['CD3D', 'CD3E'],
        'B cells': ['MS4A1', 'CD19'],
        'Macrophages': ['CD68', 'CD14']
    }


@pytest.fixture
def subsampled_zarr_path(test_data_dir):
    """Return path to first available ROI zarr fixture."""
    zarr_files = list((test_data_dir / "rois").glob("*.zarr"))
    if zarr_files:
        return zarr_files[0]
    return None


@pytest.fixture
def tmp_zarr_cleanup(tmp_path):
    """
    Provide a temp directory that aggressively cleans up .zarr files.
    Use this instead of tmp_path for tests that create large files.
    """
    yield tmp_path
    # Clean up all .zarr directories immediately after test
    for zarr_dir in tmp_path.glob("*.zarr"):
        if zarr_dir.is_dir():
            shutil.rmtree(zarr_dir, ignore_errors=True)
    gc.collect()

