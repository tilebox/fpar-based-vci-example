#!/usr/bin/env python3

"""
Modular VCI workflow definitions following Tilebox best practices.
Each workflow can be run independently or as part of a larger pipeline.
"""

from cyclopts import App
from tilebox.workflows import Client as WorkflowsClient
from tilebox.workflows import Task

from vci_workflow.tasks.end_to_end import EndToEndFparToVideos
from vci_workflow.tasks.fpar_to_zarr import WriteFparToZarr
from vci_workflow.tasks.minmax import ComputeMinMaxPerDekad
from vci_workflow.tasks.vci import ComputeVCI
from vci_workflow.tasks.video import ZarrArrayToVideo

app = App()


@app.command()
def end_to_end(time_range: str, cluster: str | None = None) -> None:
    """Run a complete FPAR pipeline producing both FPAR and VCI videos.

    This workflow combines the following steps:
    1. Converting FPAR data from MODIS/VIIRS into Zarr format (convert)
    2. Calculating the min/max values per dekad (minmax)
    3. Calculating the VCI (vci)
    4. Generating the FPAR video (fpar-video)
    5. Generating the VCI video (vci-video)

    To run only individual steps, see the remaining commands of the CLI.

    Args:
        time_range: Time range to process, in the format "YYYY-MM-DD/YYYY-MM-DD"
    """
    task = EndToEndFparToVideos(time_range=time_range)
    submit_job(task, f"fpar-vci-videos-end-to-end-{time_range}", cluster)


@app.command()
def convert(time_range: str, fpar_zarr_path: str, cluster: str | None = None) -> None:
    """Convert FPAR data to Zarr format only.

    Args:
        time_range: Time range to process, in the format "YYYY-MM-DD/YYYY-MM-DD"
        fpar_zarr_path: Output Zarr path for the FPAR data
        cluster: Optional Tilebox cluster to submit the job to
    """
    task = WriteFparToZarr(fpar_zarr_path, time_range)
    submit_job(task, f"ingest-fpar-to-zarr-{time_range}", cluster)


@app.command()
def minmax(fpar_store: str, min_max_store: str, cluster: str | None = None) -> None:
    """Min/max computation only.

    Args:
        fpar_store: Input Zarr path for the FPAR data
        min_max_store: Output Zarr path for the min/max data
        cluster: Optional Tilebox cluster to submit the job to
    """
    task = ComputeMinMaxPerDekad(fpar_store, min_max_store)
    submit_job(task, f"compute-minmax-{fpar_store}", cluster)


@app.command()
def vci(fpar_store: str, min_max_store: str, vci_store: str, cluster: str | None = None) -> None:
    """VCI calculation only.

    Args:
        fpar_store: Input Zarr path for the FPAR data
        min_max_store: Input Zarr path for the min/max data
        vci_store: Output Zarr path for the VCI data
        cluster: Optional Tilebox cluster to submit the job to
    """
    task = ComputeVCI(fpar_store, min_max_store, vci_store)
    submit_job(task, f"compute-vci-{fpar_store}", cluster)


@app.command()
def fpar_video(fpar_store: str, cluster: str | None = None) -> None:
    """FPAR video generation only.

    Args:
        fpar_store: Input Zarr path for the FPAR data
        cluster: Optional Tilebox cluster to submit the job to
    """
    task = ZarrArrayToVideo(
        fpar_store,
        "fpar",
        fpar_store,
        downscale_factors=(5, 4),
        title="FPAR (%)",
        subtitle="Fraction of absorbed photosynthetically active radiation",
    )
    submit_job(task, f"generate-fpar-video-{fpar_store}", cluster)


@app.command()
def vci_video(vci_store: str, fpar_store: str, cluster: str | None = None) -> None:
    """VCI video generation only.

    Args:
        vci_store: Input Zarr path for the VCI data
        fpar_store: Input Zarr path for the FPAR data, needed to look up dekad/year metadata for labelling frames
        cluster: Optional Tilebox cluster to submit the job to
    """
    task = ZarrArrayToVideo(
        vci_store,
        "vci",
        fpar_store,
        downscale_factors=(5, 4),
        title="VCI",
        subtitle="Vegetation Condition Index (Derived from: MODIS/VIIRS FPAR)",
    )
    submit_job(task, f"generate-vci-video-{vci_store}", cluster)


def submit_job(task: Task, job_name: str, cluster: str | None = None) -> None:
    client = WorkflowsClient().jobs()
    job = client.submit(job_name, task, cluster=cluster, max_retries=3)
    print(f"Successfully submitted job {job_name}: {job.id}")  # noqa: T201
    print(f"Monitor at: https://console.tilebox.com/workflows/jobs/{job.id}")  # noqa: T201


if __name__ == "__main__":
    app()
