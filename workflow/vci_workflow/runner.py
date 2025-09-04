import os
import socket
import sys

from dotenv import load_dotenv
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

from vci_workflow.cli import EndToEndVciWorkflow
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

if __name__ == "__main__":
    # Load environment variables from .env file
    if len(sys.argv) >= 2 and sys.argv[1] == "--load-dotenv":
        load_dotenv()

    # Configure logging backends
    configure_console_logging()
    configure_otel_logging_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # Configure tracing backends
    configure_otel_tracing_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # Configure a GCS-backed cache for sharing metadata between tasks.
    storage_client = StorageClient()
    gcs_bucket = storage_client.bucket(GCS_BUCKET)
    cache = GoogleStorageCache(gcs_bucket, prefix="vci_workflow_cache")

    # Get cluster configuration from environment variable
    cluster = os.environ.get("TILEBOX_CLUSTER", None)

    # Start the runner
    client = WorkflowsClient()
    runner = client.runner(
        tasks=[
            EndToEndVciWorkflow,
            ComputeVCI,
            ComputeVCIRecursively,
            ComputeVCIForDekad,
            LoadDekadIntoZarr,
            WriteFparToZarr,
            WriteFparDataIntoEmptyZarr,
            ComputeMinMaxPerDekad,
            ComputeMinMaxForDekad,
            ComputeMinMaxForChunk,
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
