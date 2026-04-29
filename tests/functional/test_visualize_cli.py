"""
Functional smoke tests for visualize command.
"""

import subprocess
import sys

import pytest


def test_visualize_random_roi_smoke(subsampled_zarr_path, tmp_zarr_cleanup):
    """visualize should generate ROI images and metadata for random ROI mode."""
    if subsampled_zarr_path is None or not subsampled_zarr_path.exists():
        pytest.skip("No ROI fixture zarr found")

    output_dir = tmp_zarr_cleanup / "visualize_out"
    spec_file = tmp_zarr_cleanup / "viz.toml"
    spec_file.write_text(
        """
[points]
default_color = "#999999"
default_marker = "o"
default_size = 4

[[rules]]
where = "sample == 'Drexel-Pos'"
color = "#d73027"
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spatial_tk.cli",
            "visualize",
            "--input",
            str(subsampled_zarr_path),
            "--output",
            str(output_dir),
            "--view",
            "roi",
            "--random-rois",
            "2",
            "--roi-width",
            "250",
            "--roi-height",
            "250",
            "--random-state",
            "7",
            "--spec",
            str(spec_file),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"visualize failed: {result.stderr}"
    assert (output_dir / "roi_001.png").exists()
    assert (output_dir / "roi_002.png").exists()
    assert (output_dir / "rois.csv").exists()
    assert (output_dir / "visualize.resolved.json").exists()
