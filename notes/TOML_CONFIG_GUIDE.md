# TOML Configuration Files for Reproducible Analyses

## Overview

All `spatial-tk` commands support TOML (Tom's Obvious, Minimal Language) configuration files. Using config files makes your analysis pipelines reproducible by capturing all parameters in a single, version-controlled file.

## TOML Format Basics

TOML is a simple, human-readable configuration format. Here are the basics:

### Key-Value Pairs

```toml
# Comments start with #
input = "data.zarr"
output = "results.zarr"
min_genes = 100
save_plots = true
```

### Sections

Sections (tables) are defined with square brackets:

```toml
[normalize]
min_genes = 100
min_cells = 3

[cluster]
leiden_resolution = "0.5"
```

### Data Types

- **Strings**: Use quotes for strings
  ```toml
  input = "data.zarr"
  markers = "markers.csv"
  ```

- **Numbers**: No quotes needed
  ```toml
  min_genes = 100
  downsample = 0.5
  n_top_genes = 2000
  ```

- **Booleans**: `true` or `false` (lowercase)
  ```toml
  inplace = true
  save_plots = false
  ```

- **Optional / unset values**: TOML has no `null` type. Omit the key (or leave it commented) to use the CLI default (`None` in Python).
  ```toml
  # cluster_key = "leiden_res0p5"   # omit → all leiden_res* columns
  # compare_groups = "HIV,NEG"      # omit → markers for all groups
  ```

- **Arrays/Comma-separated values**: For parameters that accept multiple values
  ```toml
  leiden_resolution = "0.2,0.5,1.0"
  ```

## Using Config Files with Commands

### Basic Usage

Each command accepts an optional `--config` argument:

```bash
spatial-tk concat --config config.toml --input samples.csv --output merged.zarr
spatial-tk normalize --config config.toml --input merged.zarr --inplace
spatial-tk cluster --config config.toml --input merged.zarr --inplace
```

### Config File Structure

Each command has its own section in the config file:

```toml
[concat]
input = "samples.csv"
output = "merged.zarr"
downsample = 1.0

[normalize]
input = "merged.zarr"
inplace = true
min_genes = 100
min_cells = 3
n_top_genes = 2000
save_plots = false

[cluster]
input = "merged.zarr"
inplace = true
leiden_resolution = "0.2,0.5,1.0"
save_plots = true

[annotate]
input = "merged.zarr"
inplace = true
markers = "markers.csv"
calculate_ulm = true
panglao_min_sensitivity = 0.5
tmin = 2
save_plots = true

[differential]
input = "merged.zarr"
output_dir = "results/"
groupby = "leiden_res0p5"
method = "wilcoxon"
n_genes = 100
save_plots = false
```

### Config Key Naming Convention

Config keys use **underscores** (e.g., `min_genes`, `n_top_genes`), while CLI arguments use **hyphens** (e.g., `--min-genes`, `--n-top-genes`). The config system automatically handles this conversion.

### CLI Arguments Override Config

**Important**: CLI arguments always take precedence over config values. This allows you to:
- Use config as defaults
- Override specific values on the command line
- Mix config and CLI arguments flexibly

Example:
```bash
# Config has downsample = 0.5, but CLI overrides it to 0.8
spatial-tk concat --config config.toml --input samples.csv --output merged.zarr --downsample 0.8
```

## Workflow for Reproducible Analyses

### Step 1: Create Your Config File

Start with `example_config.toml` as a template:

```bash
cp example_config.toml my_analysis_config.toml
```

### Step 2: Customize Parameters

Edit `my_analysis_config.toml` with your specific parameters:

```toml
[normalize]
min_genes = 200          # Stricter filtering
n_top_genes = 3000      # More variable genes
save_plots = true       # Always save plots

[cluster]
leiden_resolution = "0.2,0.5,1.0,2.0"  # Multiple resolutions
save_plots = true

[annotate]
markers = "custom_markers.csv"
calculate_ulm = true
panglao_min_sensitivity = 0.6
```

### Step 3: Run Pipeline with Config

```bash
# Full pipeline with config file
spatial-tk concat --config my_analysis_config.toml --input samples.csv --output data.zarr
spatial-tk normalize --config my_analysis_config.toml --input data.zarr --inplace
spatial-tk cluster --config my_analysis_config.toml --input data.zarr --inplace
spatial-tk annotate --config my_analysis_config.toml --input data.zarr --inplace
spatial-tk differential --config my_analysis_config.toml --input data.zarr --output-dir results/
```

### Step 4: Version Control

**Commit your config file to version control** (Git):

```bash
git add my_analysis_config.toml
git commit -m "Add analysis configuration"
```

This ensures:
- Anyone can reproduce your analysis
- Parameter changes are tracked
- You can revisit exact settings used

## Best Practices

### 1. One Config File Per Analysis

Create separate config files for different analyses or projects:

```
configs/
├── project1_config.toml
├── project2_config.toml
└── exploratory_analysis_config.toml
```

### 2. Document Your Choices

Add comments explaining parameter choices:

```toml
[normalize]
# Stricter filtering for high-quality cells
min_genes = 200
# More variable genes for better clustering
n_top_genes = 3000
```

### 3. Use Relative Paths

Prefer relative paths for portability:

```toml
[concat]
input = "data/samples.csv"           # ✅ Good
# input = "/absolute/path/to/data.csv"  # ❌ Avoid
```

### 4. Override Only When Needed

Use CLI arguments only for values that change frequently:

```bash
# Good: Use config for most parameters, override only input/output
spatial-tk concat --config config.toml --input new_samples.csv --output new_output.zarr

# Less ideal: Overriding many parameters defeats the purpose
spatial-tk concat --config config.toml --input samples.csv --output merged.zarr --downsample 0.5 --min-genes 200 ...
```

### 5. Validate Your Config

Test your config file syntax:

```bash
# Python can validate TOML syntax
python -c "import tomllib; tomllib.load(open('config.toml', 'rb'))"
```

## Common Patterns

### Pattern 1: Development vs Production

```toml
# development_config.toml
[concat]
downsample = 0.1  # Fast testing

[normalize]
n_top_genes = 500  # Faster processing
```

```toml
# production_config.toml
[concat]
downsample = 1.0  # Full dataset

[normalize]
n_top_genes = 3000  # Full analysis
```

### Pattern 2: Parameter Sweeps

Use different config files for parameter sweeps:

```bash
# Run with different resolutions
for config in config_res0p2.toml config_res0p5.toml config_res1p0.toml; do
    spatial-tk cluster --config $config --input data.zarr --inplace
done
```

### Pattern 3: Multi-Project Analysis

```bash
# Same pipeline, different configs
for project in project1 project2 project3; do
    spatial-tk concat --config ${project}_config.toml --input ${project}_samples.csv --output ${project}_merged.zarr
    spatial-tk normalize --config ${project}_config.toml --input ${project}_merged.zarr --inplace
    # ... continue pipeline
done
```

## Troubleshooting

### Config Not Applied

If config values aren't being applied:

1. **Check section name**: Must match command name exactly (`[concat]`, `[normalize]`, etc.)
2. **Check key names**: Use underscores, not hyphens (`min_genes` not `min-genes`)
3. **Check CLI override**: CLI arguments override config - ensure you're not overriding unintentionally

### Config File Not Found

```bash
# Use absolute path if relative path doesn't work
spatial-tk concat --config /full/path/to/config.toml ...
```

### Invalid TOML Syntax

Common errors:
- Missing quotes around strings: `input = samples.csv` ❌ → `input = "samples.csv"` ✅
- Wrong boolean: `save_plots = True` ❌ → `save_plots = true` ✅
- Missing closing bracket: `[normalize` ❌ → `[normalize]` ✅

## Example: Complete Workflow

```bash
# 1. Create config
cat > my_config.toml << EOF
[concat]
input = "samples.csv"
output = "merged.zarr"
downsample = 1.0

[normalize]
input = "merged.zarr"
inplace = true
min_genes = 100
n_top_genes = 2000

[cluster]
input = "merged.zarr"
inplace = true
leiden_resolution = "0.5"
save_plots = true

[annotate]
input = "merged.zarr"
inplace = true
markers = "markers.csv"
save_plots = true

[differential]
input = "merged.zarr"
output_dir = "results/"
groupby = "leiden_res0p5"
EOF

# 2. Run pipeline
spatial-tk concat --config my_config.toml --input samples.csv --output merged.zarr
spatial-tk normalize --config my_config.toml --input merged.zarr --inplace
spatial-tk cluster --config my_config.toml --input merged.zarr --inplace
spatial-tk annotate --config my_config.toml --input merged.zarr --inplace
spatial-tk differential --config my_config.toml --input merged.zarr --output-dir results/

# 3. Commit to version control
git add my_config.toml
git commit -m "Add reproducible analysis config"
```

## References

- [TOML Specification](https://toml.io/)
- `example_config.toml` - Complete example with all options
- `README.md` - General usage documentation

