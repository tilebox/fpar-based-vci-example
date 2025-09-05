import os
import socket

from cyclopts import run
from google.cloud.storage import Client as StorageClient
from tilebox.workflows import Client as WorkflowsClient
from tilebox.workflows.cache import GoogleStorageCache
from tilebox.workflows.observability.logging import (
    configure_console_logging,
    configure_otel_logging_axiom,
    get_logger,
)
from tilebox.workflows.observability.tracing import (
    configure_otel_tracing_axiom,
)

from vci_workflow.tasks.end_to_end import EndToEndFparToVideos
from vci_workflow.tasks.fpar_to_zarr import (
    LoadDekadIntoZarr,
    WriteFparDataIntoEmptyZarr,
    WriteFparToZarr,
)
from vci_workflow.tasks.minmax import (
    ComputeMinMaxForChunk,
    ComputeMinMaxForDekad,
    ComputeMinMaxPerDekad,
)
from vci_workflow.tasks.vci import ComputeVCI, ComputeVCIForDekad, ComputeVCIRecursively
from vci_workflow.tasks.video import CreateFrames, CreateVideoFromFrames, ExportFrame, ZarrArrayToVideo
from vci_workflow.zarr import GCS_BUCKET

logger = get_logger()

def start_runner(cluster: str | None = None, tilebox_api_key: str | None = None) -> None:
    """Start a Tilebox workflow runner for the FPAR to VCI video workflow.

    The runner will run until it is manually stopped, executing tasks as they are submitted.

    For automatically parallelizing the workflow, simply launch multiple instances of the runner as separate processes.

    Args:
        cluster: Optional Tilebox cluster to start the runner on.
        tilebox_api_key: A Tilebox API key to use for authentication. If not set, defaults to the TILEBOX_API_KEY
            environment variable. Go to https://console.tilebox.com/account/api-keys to create one.
    """
    if tilebox_api_key is None and "TILEBOX_API_KEY" not in os.environ:
        raise ValueError(
            "No Tilebox API key provided. Please set the TILEBOX_API_KEY environment variable or pass the --tilebox-api-key argument."
        )

    # Configure logging backends
    configure_console_logging()

    if "AXIOM_API_KEY" in os.environ:
        # Optionally configure logging and tracing to Axiom for observability.
        configure_otel_logging_axiom(f"{socket.gethostname()}-{os.getpid()}")
        configure_otel_tracing_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # Configure a GCS-backed cache for sharing metadata between tasks.
    storage_client = StorageClient()
    gcs_bucket = storage_client.bucket(GCS_BUCKET)
    cache = GoogleStorageCache(gcs_bucket, prefix="vci_workflow_cache")

    # Start the runner
    client = WorkflowsClient(token=tilebox_api_key)
    runner = client.runner(
        tasks=[
            # end to end orchestration task
            EndToEndFparToVideos,
            # fpar to zarr tasks
            WriteFparToZarr,
            WriteFparDataIntoEmptyZarr,
            LoadDekadIntoZarr,
            # minmax computation tasks
            ComputeMinMaxPerDekad,
            ComputeMinMaxForChunk,
            ComputeMinMaxForDekad,
            # vci computation tasks
            ComputeVCI,
            ComputeVCIRecursively,
            ComputeVCIForDekad,
            # video related tasks
            ZarrArrayToVideo,
            CreateFrames,
            ExportFrame,
            CreateVideoFromFrames,
        ],
        cache=cache,
        cluster=cluster,
    )

    logger.info(f"Starting runner on cluster: {runner.tasks_to_run.cluster_slug}")
    runner.run_forever()


if __name__ == "__main__":
    run(start_runner)
