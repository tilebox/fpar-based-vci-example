# FPAR to VCI Example Workflow

A modular Vegetation Condition Index (VCI) calculation pipeline built with Tilebox workflows.

<p align="center">
  <img src="../VCI.png"></a>
</p>

<div align="center">
  <a href="https://youtu.be/s4wzyX9adWo">
    <img src="https://img.shields.io/badge/FPAR-View_on_Youtube-red?logo=youtube&style=flat-square" alt="VCI Youtube"/>
  </a>
  <a href="https://youtu.be/AGr1OB91ZPk">
    <img src="https://img.shields.io/badge/VCI-View_on_Youtube-red?logo=youtube&style=flat-square" alt="FPAR Youteube"/>
  </a>
</div>

## Architecture

The workflow processes satellite FPAR data through four main stages:

1. **FPAR Conversion** - Downloads and consolidates MODIS/VIIRS FPAR data into the Zarr format
2. **Min/Max Computation** - Computes historical min/max values per dekad (10-day periods)
3. **VCI Computation** - Calculates VCI values using formula: `vci = (fpar - dekad_min) / (dekad_max - dekad_min)`
4. **Video Generation** - Creates MP4 visualizations with temporal animation

## Setup

### Create a Tilebox API Key

Head over to https://console.tilebox.com/account/api-keys and create a new API key.
You'll need this to authenticate with the Tilebox API.

### Install Dependencies

```bash
uv sync
```

> [!TIP]
> If you're not familiar with `uv`, a tool for managing Python installations and project dependencies, check out the [documentation](https://docs.astral.sh/uv/#installation) here.

### Configure a google storage bucket

The workflow uses a Google Cloud Storage bucket to store the FPAR Zarr datacube, as well as intermediate workflow cache
data. Utilizing a dedicated bucket for this is what allows us to easily set it up on a auto-scaling Spot instance cluster
on GCP.

Create a google storage bucket and then modify the `vci_workflow.zarr.GCS_BUCKET` constant to point to your bucket slug.

> [!TIP]
> Check out our [Infrastructure as Code for this workflow](../infrastructure/), which not only provisions the bucket, but also a Runner cluster and all the necessary networking and IAM configurations to run the workflow at scale using cheap GCP Spot instances.

## Usage

### Run A Complete Workflow

```bash
uv run python -m vci_workflow.cli end-to-end --time-range "2022-01-01/2022-12-31" --tilebox-api-key <your-api-key>
```

> [!TIP]
> You can omit the `--tilebox-api-key` argument if you have set the `TILEBOX_API_KEY` environment variable.

### Run Individual Steps

```bash
# Each step prints the command for the next step
uv run python -m vci_workflow.cli convert --time-range "2022-01-01/2022-12-31" --fpar-store "fpar.zarr" --tilebox-api-key <your-api-key>

# --fpar-store is the output of the previous step, so the value needs to match
uv run python -m vci_workflow.cli minmax --fpar-store "fpar.zarr" --min-max-store "minmax.zarr" --tilebox-api-key <your-api-key>

# --fpar-store and --min-max-store are the outputs of the previous steps, so the values need to match
uv run python -m vci_workflow.cli vci --store "fpar.zarr" --min-max-store "minmax.zarr" --vci-store "vci.zarr" --tilebox-api-key <your-api-key>


uv run python cli.py fpar-video --fpar-store "fpar.zarr" --tilebox-api-key <your-api-key>
uv run python cli.py vci-video --vci-store "vci.zarr" --fpar-store "fpar.zarr" --tilebox-api-key <your-api-key>
```

### Monitor progress

After submitting a job by running one of the `uv run` commands above, you can monitor the job's progress by heading over to
the [Tilebox console](https://console.tilebox.com/workflows/jobs).

### Start runners

For the workflow to make progress, you'll need to start one or more runners. To start a runner, run the following command:

```bash
uv run python -m vci_workflow.runner
```

> [!TIP]
> You can start multiple runners in separate terminal windows to increase parallelism. There is also [`call-in-parallel`](https://github.com/tilebox/call-in-parallel) for automating this as a single command.
