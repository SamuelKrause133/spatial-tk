# Changelog

All notable changes to the Xenium Spatial Clustering and Annotation Tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **ROI fixture generator script for tests**:
  - Added `tests/test_data/generate_roi_subsets.py` to generate ROI subset `.zarr` files from a single input `.zarr`.
  - Supports configurable ROI count, target cell ranges, coordinate system, overlap/distance constraints, and manifest output for test ingestion.
- **Tiered functional test sample manifests**:
  - Added `tests/test_data/test_samples_fast.csv` for in-repo ROI-based functional testing.
  - Added `tests/test_data/test_samples_full.csv` for full-size out-of-repo functional testing.
- **Spatial neighbors subcommand**:
  - Added `spatial-tk spatial_neighbors` to build Squidpy spatial graphs from existing `.zarr` datasets.
  - Added support for CLI/config options including `spatial_key`, `table_key`, `library_key`, `library_id`, `n_neighs`, `radius`, `transform`, and `key_added`.
  - Added new core module `spatial_tk/core/spatial_neighbors.py` with radius parsing and Squidpy wrapper logic.
  - Added functional test `tests/functional/test_spatial_neighbors_command.py` to verify that spatial graph matrices persist in `obsp` after save/reload.
- **Spatial neighborhood clustering subcommand**:
  - Added `spatial-tk spatial_cluster` to cluster neighborhood cell-type composition vectors derived from spatial connectivities.
  - Added support for configurable graph key, `cell_type_key`, output/result keys, k-means sweep bounds, and `--force-n-clusters`.
  - Added `--mode` with `kmeans` and `hdbscan` options for alternative clustering backends.
  - Added HDBSCAN options (`hdbscan_min_cluster_size`, `hdbscan_min_samples`, `hdbscan_cluster_selection_epsilon`, `hdbscan_metric`, `hdbscan_allow_single_cluster`) and mode-specific result fields in `uns`.
  - Added new core module `spatial_tk/core/spatial_clustering.py` for composition construction, k-means sweep, silhouette/inertia scoring, and uns result schema.
- **Visualization subcommand**:
  - Added `spatial-tk visualize` to render full-slide or ROI spatial point plots with rule-based styling from a supplemental TOML spec.
  - Added support for manual ROIs (`--roi`), ROI CSV input (`--roi-file`), and random ROI generation (`--random-rois`, `--roi-width`, `--roi-height`).
  - Added optional point subsampling (`--max-points`), figure overrides (`--figsize`, `--dpi`, `--title`), and config-file integration via `[visualize]`.
  - Added background image overlay from `SpatialData.images` or an external source (`--overlay-image`, `--image-source`, `--image-layer`, multiscale/channel controls).
  - Added output artifacts: PNG plot(s), `rois.csv` metadata for multi-ROI runs, and `visualize.resolved.json` with merged CLI/spec settings.
  - Added new core module `spatial_tk/core/visualization.py` for ROI resolution, style compilation (direct `where` rules, categorical mapping, continuous colormaps), multiscale image extraction/alignment, and matplotlib rendering.
  - Added `load_image_source()` in `spatial_tk/core/data_io.py` to load image layers from raw Xenium directories or `.zarr` stores.
  - Added `copy_spatial_store()` in `spatial_tk/core/data_io.py` for filesystem-level `.zarr` copying without materializing arrays.
  - Added unit tests `tests/unit/test_visualization.py` and `tests/unit/test_visualize_command.py`.
  - Added functional smoke test `tests/functional/test_visualize_cli.py`.

### Changed
- **Functional test fixture routing**:
  - Updated `tests/conftest.py` `test_samples_csv` fixture to select fast/full manifests using `SPATIAL_TK_TEST_TIER` (`fast` default, `full` optional).
  - Added optional manifest overrides: `SPATIAL_TK_FAST_SAMPLES_CSV` and `SPATIAL_TK_FULL_SAMPLES_CSV`.
  - Updated `subsampled_zarr_path` fixture to use ROI fixtures from `tests/test_data/rois/`.
- **Test data layout**:
  - Moved ROI test fixtures from temporary workspace paths into `tests/test_data/rois/`.
  - Replaced `tests/test_data/test_samples.csv` entries to reference ROI fixtures in `tests/test_data/rois/`.
  - Removed legacy `tests/test_data/subsampled_*.zarr` fixtures in favor of ROI and external full-data manifests.
- **Pytest marker registration**:
  - Added marker declarations in `pyproject.toml` for `slow`, `functional_fast`, and `functional_full`.
- **Makefile test target behavior**:
  - Updated `make test` to run with `SPATIAL_TK_TEST_TIER=full` (full external datasets).
  - Kept `make test-unit` as quick unit-only tests.
  - Updated `make test-functional` to run with `SPATIAL_TK_TEST_TIER=fast` (ROI fixtures).
- **CLI and docs**:
  - Registered `spatial_neighbors` in `spatial_tk/cli.py`.
  - Registered `spatial_cluster` in `spatial_tk/cli.py`.
  - Registered `visualize` in `spatial_tk/cli.py`.
  - Added `[spatial_neighbors]` section to `example_config.toml`.
  - Added `[spatial_cluster]` section to `example_config.toml`.
  - Updated `README.md` with command usage and examples for `spatial_neighbors`, `spatial_cluster`, and `visualize`.
- **Table-only I/O for analysis commands**:
  - Extended `load_table_only()` and `save_table_only()` with optional `table_key` selection.
  - Updated `assign`, `cluster`, `normalize`, `quantitate`, `spatial_neighbors`, and `spatial_cluster` to load/save only the AnnData table.
  - Non-inplace writes now use `copy_spatial_store()` followed by `save_table_only()` to avoid reloading images and shapes.
  - `load_spatial_datasets()` now records `adata.uns["spatial_tk"]["image_source"]` for raw Xenium inputs to support later image overlay in visualize.
  - `setup_squidpy_structure()` no longer eagerly converts multiscale image trees into numpy arrays.
- **Table selection utilities**:
  - Updated `get_table()` and `set_table()` in `spatial_tk/utils/helpers.py` to support optional `table_key` selection.
- **End-to-end pipeline test**:
  - Extended `tests/functional/test_full_pipeline.py` to include `spatial_neighbors` as step 7 in the full pipeline flow.

### Changed (breaking)

- Renamed the Python package from `xenium_process` to `spatial_tk`, the distribution/PyPI name from `xenium-process` to `spatial-tk`, and the CLI entry point from `xenium_process` to `spatial-tk`. Run `pip install spatial-tk` (or install from source) and invoke `spatial-tk ...` or `python -m spatial_tk.cli ...`.

## [1.3.0] - 2025-11-09

### Added
- **Memory-Efficient Table I/O**:
  - `load_table_only()`: Direct AnnData loading from `zarr_path/tables/table` without loading SpatialData
  - `save_table_only()`: Direct AnnData writing to zarr without loading SpatialData
  - Automatic table name detection (handles custom table names)
  - Fallback to SpatialData read if direct path fails (for robustness)
- **Selective Image Loading**:
  - `load_existing_spatial_data()`: Added `load_images=False` parameter to skip image loading
  - `load_spatial_datasets()`: Added `load_images=True` parameter (default True for concat)
  - Images automatically removed from memory when not needed
  - On-demand image loading in visualization script (`load_images_on_demand()`)
- **Raw Xenium Dataset Support**:
  - Automatic detection of `.zarr` vs raw Xenium dataset directories
  - `load_xenium_dataset()`: Loads raw Xenium datasets using `spatialdata_io.xenium()`
  - Automatic image loading from `morphology_focus` directory
  - `setup_squidpy_structure()`: Configures `adata.uns['spatial']` for squidpy compatibility

### Changed
- **Memory Optimization for Processing Commands**:
  - `normalize` and `cluster` commands now use `load_table_only()` for all operations
  - Inplace operations (`--inplace`) use `save_table_only()` - no SpatialData loading at all
  - Non-inplace operations load SpatialData without images for efficiency
  - Significant memory savings (90%+ for image-heavy datasets) during clustering/normalization
- **Visualization Script**:
  - Images only loaded when `--overlay-image` flag is used
  - On-demand image loading when overlay is requested
  - Automatic image path detection and loading from zarr
- **Data Loading Strategy**:
  - Concat command: Loads images by default (may be needed for downstream visualization)
  - Normalize/Cluster commands: Skip images entirely (not needed for processing)
  - Visualization: Load images only when explicitly requested

### Performance Improvements
- **Memory Usage**:
  - Table-only operations avoid loading images, shapes, and SpatialData metadata
  - Direct AnnData I/O eliminates SpatialData object overhead
  - Inplace operations are now most efficient (no SpatialData loading)
- **Loading Speed**:
  - Direct table I/O faster than loading full SpatialData and extracting table
  - Image loading deferred until actually needed
  - Reduced I/O overhead for processing workflows

## [1.2.0] - 2025-11-05

### Changed
- **Enrichment Scoring Method**: Switched from Univariate Linear Model (ULM) to Multivariate Linear Model (MLM)
  - All enrichment scoring now uses `dc.mt.mlm()` instead of `dc.mt.ulm()`
  - Function renamed: `calculate_ulm_scores()` → `calculate_mlm_scores()`
  - Obsm keys updated: `score_ulm_{resource}` → `score_mlm_{resource}`
  - MLM provides more accurate activity estimates by accounting for correlated regulators and their interactions
  - CLI flag `--calculate-ulm` retained for backward compatibility (still calculates MLM scores)
- **Plotting**: Updated enrichment heatmap function to correctly detect MLM score keys in obsm

### Documentation
- Added `notes/ULM_vs_MLM.md` with detailed explanation of differences and toy example
- Updated all documentation references from ULM to MLM
- Updated configuration file examples and help text

## [1.1.0] - 2025-11-04

### Added
- **TOML Configuration File Support**:
  - All commands now accept an optional `--config` argument for TOML configuration files
  - Each command reads its own section from the config file (`[concat]`, `[normalize]`, `[cluster]`, `[annotate]`, `[differential]`)
  - Config values can be overridden by CLI arguments for maximum flexibility
  - Automatic type conversion for config values (strings to ints/floats/bools)
  - Support for underscore/hyphen key mapping (e.g., `min_genes` in config maps to `--min-genes` CLI arg)
- **Makefile Run Target**:
  - New `make run ROOT=/path/to/directory` target for running full pipeline
  - Executes all five pipeline steps sequentially using a single config file
  - Validates config file existence and stops on any step failure
  - Changes working directory to ROOT for relative path resolution
- **Configuration Documentation**:
  - `example_config.toml` - Complete example with all command sections documented
  - `notes/TOML_CONFIG_GUIDE.md` - Comprehensive guide to TOML config files
  - `notes/MAKEFILE_RUN_TARGET.md` - Documentation for the make run target
- **Config Utility Module**:
  - `spatial_tk/utils/config.py` with `load_config()` and `merge_config_with_args()` functions
  - Uses Python 3.11+ built-in `tomllib` module (no additional dependencies)
  - Handles config loading, merging with CLI args, and type conversion

### Changed
- **Command Argument Handling**:
  - Required arguments (`--input`, `--output`, etc.) are now optional when `--config` is provided
  - Arguments are validated after config merge instead of during argparse parsing
  - Clear error messages when required values are missing from both CLI and config
- **CLI Behavior**:
  - Commands can now be run with only `--config` argument when all required parameters are in config file
  - CLI arguments always override config values (standard precedence)
  - Config files support relative paths resolved from config file location

### Testing
- **Unit Tests**:
  - `tests/unit/test_config.py` - Tests for config loading, merging, and type conversion
- **Functional Tests**:
  - `tests/functional/test_config_integration.py` - Integration tests for config files with all commands
  - Tests verify config values override defaults, CLI overrides config, and error handling

### Documentation
- Updated README.md with configuration file section
- Added examples of using config files in workflows
- Documented config key naming conventions (underscores vs hyphens)

## [1.0.0] - 2025-10-28

### Major Refactor - Xenium Spatial Data Support

This is a major rewrite transforming the tool from single-cell RNA-seq to Xenium spatial transcriptomics analysis.

### Added
- **Xenium Spatial Data Support**:
  - Load and concatenate multiple Xenium .zarr datasets
  - CSV input format with sample metadata (sample, path, optional columns)
  - SpatialData integration for spatial coordinate preservation
  - Output as .zarr format preserving spatial information
- **Modular Architecture**:
  - Refactored ~1000 line monolithic script into organized modules:
    - `data_io.py`: Data loading, concatenation, and saving
    - `preprocessing.py`: QC, filtering, normalization, HVG selection
    - `clustering.py`: PCA, neighbors, UMAP, Leiden clustering
    - `annotation.py`: Marker loading, cell type annotation, ULM scores, DE analysis
    - `plotting.py`: All visualization functions
    - `main.py`: CLI and workflow orchestration
- **ULM Enrichment Scoring**:
  - Pre-calculate ULM scores for pathway/TF resources via `--calculate-ulm` flag
  - Resources included: hallmark, collectri, dorothea, progeny, PanglaoDB
  - PanglaoDB markers filtered by canonical status and sensitivity (default: >0.5)
  - Scores stored in `adata.obsm['score_ulm_{resource}']`
  - Configurable PanglaoDB sensitivity via `--panglao-min-sensitivity`
- **Enhanced Dependencies**:
  - Added `spatialdata` for spatial data handling
  - Added `squidpy` for future spatial analysis features
- **Multi-sample Support**:
  - Automatic sample concatenation with metadata preservation
  - Sample-aware batch correction in HVG selection
  - UMAP visualization colored by sample

### Changed
- **Input Format**: Now accepts CSV file with sample paths instead of single h5ad file
- **Output Format**: Saves processed data as .zarr SpatialData object instead of h5ad
- **Main Script**: Renamed from `scanpy_cluster.py` to `main.py` (legacy script preserved)
- **Cell Type Annotation**: Updated to use latest decoupler API:
  - `dc.run_mlm()` instead of `dc.mt.mlm()`
  - `dc.get_acts()` instead of `dc.pp.get_obsm()`
  - `dc.rank_sources_groups()` instead of `dc.tl.rankby_group()`
- **Documentation**: Completely rewritten README for Xenium workflow

### Maintained
- All existing features from v0.2.0:
  - Quality control and filtering
  - Normalization and feature selection
  - PCA, UMAP, Leiden clustering
  - Differential expression analysis
  - Cell type annotation with markers
  - Resume functionality
  - Downsampling
  - Multiple clustering resolutions
  - Comprehensive plotting

### Notes
- Legacy `scanpy_cluster.py` remains available for h5ad single-cell workflows
- Spatial-specific analyses (spatial autocorrelation, niche detection, etc.) planned for future releases

## [0.2.0] - 2025-10-22

### Added
- **Differential Expression Analysis**: Automated identification of marker genes for each cluster
  - Uses `sc.tl.rank_genes_groups()` with Wilcoxon rank-sum test
  - Runs automatically for all clustering resolutions
  - Supports resume functionality to skip already computed analyses
- **Differential Expression Output Files**:
  - `deg_all_clusters_res{resolution}.csv`: Complete DE results for all genes across all clusters
  - `deg_top100_per_cluster_res{resolution}.csv`: Top 100 marker genes per cluster with statistics
  - Files organized in `differential_expression/` subdirectory
- **Differential Expression Visualizations**:
  - Dotplot showing top 5 DE genes per cluster (`deg_dotplot_res{resolution}.png`)
  - Heatmap showing top 10 DE genes with expression patterns (`deg_heatmap_res{resolution}.png`)
  - Both visualizations generated automatically when `--save-plots` flag is used
- **Enhanced Documentation**:
  - Added detailed documentation for differential expression features
  - Updated script docstring to reflect new capabilities

### Changed
- Updated workflow to integrate DE analysis after clustering and before cell type annotation
- Enhanced `save_results()` function to automatically save DE results for each resolution
- Expanded `save_plots()` function to generate DE visualizations

## [0.1.0] - 2025-10-21

### Added
- **Initial Release**: Complete scRNA-seq analysis pipeline
- **Data Loading and Preprocessing**:
  - Support for h5ad format input files
  - Automatic handling of variable and observation name uniqueness
  - Optional downsampling for quick testing
- **Quality Control**:
  - Calculation of QC metrics (mitochondrial, ribosomal, hemoglobin gene percentages)
  - Configurable cell and gene filtering thresholds
  - QC visualization plots (violin plots, scatter plots)
- **Normalization and Feature Selection**:
  - Median total count normalization
  - Log transformation
  - Highly variable gene selection (default: 2000 genes)
  - Batch-aware feature selection when sample information available
- **Dimensionality Reduction**:
  - PCA analysis with variance ratio plots
  - Neighborhood graph computation
  - UMAP embedding for visualization
- **Clustering**:
  - Leiden clustering with configurable resolution(s)
  - Support for multiple clustering resolutions in a single run
  - Results stored with unique keys for each resolution
- **Cell Type Annotation**:
  - Marker-based annotation using decoupler's MLM (multivariate linear model)
  - CSV format for marker gene input (cell_type, gene columns)
  - Automatic enrichment score calculation per cell type
  - Cluster-level annotation based on top enrichment scores
- **Visualizations** (with `--save-plots` flag):
  - QC plots: violin plots, scatter plots
  - Highly variable genes plot
  - PCA variance ratio plot
  - UMAP colored by sample (if available)
  - UMAP colored by clusters for each resolution
  - UMAP colored by cell type annotations for each resolution
  - Marker gene dotplots for each resolution
  - Enrichment score heatmaps for each resolution
- **Output Files**:
  - Processed AnnData object (`processed_data.h5ad`)
  - Cell type annotation CSVs for each resolution
  - Organized plot directory structure
- **Command-Line Interface**:
  - Required arguments: `--input`, `--output-dir`
  - Optional arguments: `--markers`, `--save-plots`, `--min-genes`, `--min-cells`, 
    `--n-top-genes`, `--leiden-resolution`, `--downsample`, `--resume`
  - Comprehensive help documentation and usage examples
- **Resume Functionality**:
  - Ability to resume from existing analysis
  - Skips already computed steps (QC, normalization, PCA, UMAP, clustering, annotation)
  - Useful for adding new markers or resolutions without recomputing everything
- **Logging**:
  - Detailed logging with timestamps
  - Progress tracking for all major steps
  - Error handling with informative messages
- **Performance Features**:
  - Non-interactive backend for plot generation
  - Efficient processing of large datasets
  - Graceful handling of missing marker genes

### Technical Details
- Based on Scanpy and the Scverse ecosystem
- Follows best practices from Scverse tutorials
- Uses igraph-based Leiden algorithm for clustering
- Implements decoupler for marker-based enrichment analysis
- Compatible with standard h5ad file formats

