import os
import socket

from google.cloud.storage import Client as StorageClient  # type: ignore[import-untyped]
from tilebox.workflows import Client as WorkflowsClient  # type: ignore[import-untyped]
from tilebox.workflows.cache import GoogleStorageCache  # type: ignore[import-untyped]

from config import GCS_BUCKET
from ingest import (
    LoadDekadIntoZarr,
    InitializeZarrStore,
    WriteFparToZarr,
    WriteFparDataIntoEmptyZarr,
)
from minmax import (
    ComputeMinMaxPerDekad,
    CalculateMinMaxForDekad,
    InitializeMinMaxArrays,
    OrchestrateDekadMinMaxCalculation,
)
from tilebox.workflows.observability.logging import (  # type: ignore[import-untyped]
    configure_console_logging,
    configure_otel_logging_axiom,
)
from tilebox.workflows.observability.tracing import (  # type: ignore[import-untyped]
    configure_otel_tracing_axiom,
)
from vci import (
    CalculateVciChunk,
    CalculateVciDekad,
    CalculateVciForYear,
    InitializeVciArray,
    OrchestrateVciByYear,
)
from vci_visualization import (
    CreateVciFramesByYear,
    CreateVciFramesForYear,
    CreateSingleVciFrame,
    CreateVideoFromFrames,
    CreateVciVideo,
    DownloadVideoFromCache,
)
from vci_workflow import VciWorkflow

if __name__ == "__main__":
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
            VciWorkflow,
            InitializeZarrStore,
            LoadDekadIntoZarr,
            ComputeMinMaxPerDekad,
            WriteFparToZarr,
            WriteFparDataIntoEmptyZarr,
            InitializeMinMaxArrays,
            InitializeVciArray,
            OrchestrateVciByYear,
            CalculateVciForYear,
            CalculateVciDekad,
            CalculateVciChunk,
            CalculateMinMaxForDekad,
            OrchestrateDekadMinMaxCalculation,
            CreateVciVideo,
            CreateVciFramesByYear,
            CreateVciFramesForYear,
            CreateSingleVciFrame,
            CreateVideoFromFrames,
            DownloadVideoFromCache,
        ],
        cache=cache,
        cluster=cluster,
    )

    print(f"Starting runner on cluster: {cluster}")
    runner.run_forever()
