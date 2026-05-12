.PHONY: help venv venv-image install install-dev build test test-unit test-functional test-functional-analysis test-bridge-functional test-coverage test-unit-image test-unit-analysis test-all-envs clean clean-all lint format create-test-data run

# Default analysis env: ./venv. Optional image/JVM env: ./venv_image (see image.env.yaml).
VENV_IMAGE_PY := ./venv_image/bin/python
VENV_ANALYSIS_PY := ./venv/bin/python
VENV_IMAGE_JAVA_HOME := $(CURDIR)/venv_image/lib/jvm

# Default target
help:
	@echo "spatial-tk Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make venv              - Create ./venv (analysis stack: conda py3.12 + requirements-analysis.txt)"
	@echo "  make venv-image        - Create ./venv_image: conda (py3.10+JDK) then pip deps from requirements-image.txt"
	@echo "  make install           - Install package in current environment"
	@echo "  make install-dev       - Install package with dev dependencies"
	@echo "  make build             - Build distribution packages"
	@echo "  make test              - Orchestrated tests (./venv + ./venv_image targets below)"
	@echo "  make test-unit         - Run only unit tests (fast, current python)"
	@echo "  make test-unit-analysis - Unit tests using ./venv"
	@echo "  make test-unit-image    - Image CLI + image-side bridge unit tests using ./venv_image"
	@echo "  make test-all-envs      - test-unit-analysis, test-unit-image, test-bridge-functional"
	@echo "  make check-env-analysis - Print versions + validate imports (./venv)"
	@echo "  make check-env-image    - Print versions + validate imports (./venv_image)"
	@echo "  make test-functional    - Fast functional tests in ./venv (excludes functional_full)"
	@echo "  make test-functional-analysis - Same as test-functional"
	@echo "  make test-bridge-functional - OIR bridge functional tests in ./venv (needs ./venv_image + fixture)"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo "  make create-test-data  - Generate subsampled test data (uses ./venv if present)"
	@echo "  make clean-test        - Clean up test temporary files"
	@echo "  make lint              - Run linting checks"
	@echo "  make format            - Format code with black"
	@echo "  make clean             - Remove build artifacts and caches"
	@echo "  make clean-all         - Remove build artifacts, caches, and conda prefixes (venv, venv_image)"
	@echo "  make run ROOT=/path    - Run full pipeline (6 steps) using config.toml in ROOT directory"

# Analysis stack — default environment at ./venv (Python 3.12 + requirements-analysis.txt)
venv:
	conda create -p ./venv python=3.12 pip setuptools wheel -y
	$(VENV_ANALYSIS_PY) -m pip install --upgrade pip wheel packaging "setuptools<81"
	$(VENV_ANALYSIS_PY) -m pip install -r requirements-analysis.txt
	# Install package metadata + console scripts without re-resolving deps.
	$(VENV_ANALYSIS_PY) -m pip install -e . --no-deps

# Microscopy / JVM / Cellpose (same workflow as main: minimal conda + pip; avoids conda-forge YAML brittleness)
# After install: set JAVA_HOME to the JDK inside the prefix (javabridge needs javac + runtime), e.g.
#   export JAVA_HOME=$(pwd)/venv_image/lib/jvm
#   export PATH="$$JAVA_HOME/bin:$$PATH"
venv-image:
	conda create -p ./venv_image python=3.10 openjdk=17 pip setuptools wheel -y
	$(VENV_IMAGE_PY) -m pip install --upgrade pip wheel packaging "setuptools<81"
	$(VENV_IMAGE_PY) -m pip install numpy==1.26.4 scipy==1.11.4
	JAVA_HOME=$(VENV_IMAGE_JAVA_HOME) PATH="$(VENV_IMAGE_JAVA_HOME)/bin:$$PATH" $(VENV_IMAGE_PY) -m pip install --no-build-isolation python-javabridge==4.0.3
	JAVA_HOME=$(VENV_IMAGE_JAVA_HOME) PATH="$(VENV_IMAGE_JAVA_HOME)/bin:$$PATH" $(VENV_IMAGE_PY) -m pip install -r requirements-image.txt
	# Install package metadata + console scripts without re-resolving deps.
	# (Deps are pinned by requirements-image.txt + the numpy/scipy bootstrap above.)
	$(VENV_IMAGE_PY) -m pip install -e . --no-deps

# Install package
install:
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -e ".[analysis,dev]"

# Build distribution packages
build:
	pip install --upgrade build
	python -m build
	@echo "Distribution packages created in dist/"

# Orchestrated: default analysis env + image env + fast functional + full bridge (bridge skips if missing fixture/envs)
test: test-all-envs test-functional-analysis

# Run only unit tests
test-unit:
	pytest tests/unit/ -v --basetemp=.pytest_tmp

test-unit-analysis:
	@$(VENV_ANALYSIS_PY) -m pytest tests/unit/ -v --basetemp=.pytest_tmp

test-unit-image:
	@$(VENV_IMAGE_PY) -m pytest tests/unit/test_lazy_cli.py tests/unit/test_import_bioformat_bridge.py -v --basetemp=.pytest_tmp

check-env-analysis:
	@$(VENV_ANALYSIS_PY) scripts/validate_env.py analysis

check-env-image:
	@JAVA_HOME=$(VENV_IMAGE_JAVA_HOME) PATH="$(VENV_IMAGE_JAVA_HOME)/bin:$$PATH" $(VENV_IMAGE_PY) scripts/validate_env.py image

test-all-envs: test-unit-analysis test-unit-image test-bridge-functional

# Fast functional tests (ROI fixtures); excludes OIR bridge tests marked functional_full
test-functional: test-functional-analysis

test-functional-analysis:
	@SPATIAL_TK_TEST_TIER=fast $(VENV_ANALYSIS_PY) -m pytest tests/functional/ -v --basetemp=.pytest_tmp -m "not functional_full"

test-bridge-functional:
	@SPATIAL_TK_TEST_TIER=full $(VENV_ANALYSIS_PY) -m pytest tests/functional/test_bioformat_csv2zarr_bridge.py -v --basetemp=.pytest_tmp

# Run tests with coverage
test-coverage:
	pytest --cov=spatial_tk --cov-report=html --cov-report=term --basetemp=.pytest_tmp
	@echo "Coverage report generated in htmlcov/"

# Clean up test temporary files
clean-test:
	rm -rf .pytest_tmp/
	rm -rf .pytest_cache/
	@echo "Cleaned test temporary files"

# Create test data (prefer project ./venv when present)
create-test-data:
	@if [ -x "$(VENV_ANALYSIS_PY)" ]; then \
		$(VENV_ANALYSIS_PY) scripts/create_test_data.py \
			--input-csv example.csv \
			--output-dir tests/test_data \
			--n-cells 500; \
	else \
		python scripts/create_test_data.py \
			--input-csv example.csv \
			--output-dir tests/test_data \
			--n-cells 500; \
	fi

# Run linting
lint:
	@echo "Running flake8..."
	-flake8 spatial_tk/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "Running mypy..."
	-mypy spatial_tk/ --ignore-missing-imports

# Format code
format:
	@echo "Formatting with black..."
	black spatial_tk/ tests/ scripts/

# Clean build artifacts and caches
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .pytest_tmp/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned build artifacts and caches"

# Clean everything including venv prefixes
clean-all: clean
	rm -rf venv/
	rm -rf venv_image/
	@echo "Cleaned everything including venv"

# Development workflow shortcuts
dev-setup: venv create-test-data
	@echo "Development environment ready! (activate ./venv)"

# Quick test during development
quick-test: test-unit
	@echo "Quick unit tests passed!"

# Run full pipeline using config file
run:
	@if [ -z "$(ROOT)" ]; then \
		echo "Error: ROOT must be specified. Usage: make run ROOT=/path/to/directory"; \
		exit 1; \
	fi
	@if [ ! -d "$(ROOT)" ]; then \
		echo "Error: Directory $(ROOT) does not exist"; \
		exit 1; \
	fi
	@if [ ! -f "$(ROOT)/config.toml" ]; then \
		echo "Error: config.toml not found in $(ROOT)"; \
		exit 1; \
	fi
	@echo "Running pipeline with config: config.toml"
	@echo "Working directory: $(ROOT)"
	@echo "=========================================="
	@cd "$(ROOT)" && \
	echo "Step 1: Concatenate samples" && \
	spatial-tk concat --config "config.toml" || exit 1
	@cd "$(ROOT)" && \
	echo "" && \
	echo "Step 2: Normalize data" && \
	spatial-tk normalize --config "config.toml" || exit 1
	@cd "$(ROOT)" && \
	echo "" && \
	echo "Step 3: Cluster cells" && \
	spatial-tk cluster --config "config.toml" || exit 1
	@cd "$(ROOT)" && \
	echo "" && \
	echo "Step 4: Quantitate enrichment scores" && \
	spatial-tk quantitate --config "config.toml" || exit 1
	@cd "$(ROOT)" && \
	echo "" && \
	echo "Step 5: Assign cell type labels" && \
	spatial-tk assign --config "config.toml" || exit 1
	@cd "$(ROOT)" && \
	echo "" && \
	echo "Step 6: Differential expression analysis" && \
	spatial-tk differential --config "config.toml" || exit 1
	@echo ""
	@echo "=========================================="
	@echo "Pipeline completed successfully!"
