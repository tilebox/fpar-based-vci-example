# VCI Workflow System

A modular Vegetation Condition Index (VCI) calculation pipeline built with Tilebox workflows.

## Architecture

The system processes satellite FPAR data through four main stages:

1. **FPAR Ingestion** - Downloads and consolidates MODIS/VIIRS data into Zarr format
2. **Min/Max Calculation** - Computes historical min/max values per dekad (10-day periods)
3. **VCI Computation** - Calculates VCI using formula: `(current - min) / (max - min)`
4. **Video Generation** - Creates MP4 visualizations with temporal animation

## Usage

### Run Complete Pipeline

```bash
uv run python cli.py end-to-end --time-range "2022-01-01/2022-12-31"
```

### Run Individual Steps

```bash
# Each step prints the command for the next step
uv run python cli.py ingest --time-range "2022-01-01/2022-12-31" --fpar-zarr-path "cube.zarr"
uv run python cli.py minmax --fpar-zarr-path "cube.zarr" --min-max-zarr-path "minmax.zarr"
uv run python cli.py vci --fpar-zarr-path "cube.zarr" --min-max-zarr-path "minmax.zarr" --vci-zarr-path "vci.zarr"
uv run python cli.py video --vci-zarr-path "vci.zarr" --fpar-zarr-path "cube.zarr"
```

## Development Commands

```bash
# Type checking
uv run mypy .

# Linting and formatting
uv run ruff check .
uv run ruff format .

# Code analysis
uv run pylyzer .

# Add dependencies
uv add <package-name>
```

## Key Improvements

- **Memory Management**: Fixed memory leaks by reducing chunk sizes and properly closing Zarr stores
- **Modularity**: Each workflow step can run independently with job metadata chaining
- **Error Handling**: Improved resource cleanup with try/finally blocks
- **Performance**: Direct Zarr array access instead of xarray for compute-intensive operations
