import os
import socket

from dotenv import load_dotenv  # type: ignore[import-untyped]
from google.cloud.storage import Client as StorageClient  # type: ignore[import-untyped]

from fpar_to_vci.cli import (
    EndToEndVciWorkflow,
    FparIngestionWorkflow,
    MinMaxWorkflow,
    VciCalculationWorkflow,
    VciVideoWorkflow,
)
from fpar_to_vci.config import GCS_BUCKET
from fpar_to_vci.ingest import (
    InitializeZarrStore,
    LoadDekadIntoZarr,
    WriteFparDataIntoEmptyZarr,
    WriteFparToZarr,
)
from fpar_to_vci.minmax import (
    CalculateMinMaxForDekad,
    ComputeMinMaxPerDekad,
    InitializeMinMaxArrays,
    OrchestrateDekadMinMaxCalculation,
)
from fpar_to_vci.vci import CalculateVciDekad, ComputeVci, ComputeVciSlice, InitializeVciArray
from fpar_to_vci.vci_visualization import CreateSingleVciFrame, CreateVciFrames, CreateVciMp4, CreateVideoFromFrames
from tilebox.workflows import Client as WorkflowsClient  # type: ignore[import-untyped]
from tilebox.workflows.cache import GoogleStorageCache  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import (  # type: ignore[import-untyped]
    configure_console_logging,
    configure_otel_logging_axiom,
)
from tilebox.workflows.observability.tracing import (  # type: ignore[import-untyped]
    configure_otel_tracing_axiom,
)

if __name__ == "__main__":
    # Load environment variables from .env file
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
            FparIngestionWorkflow,
            MinMaxWorkflow,
            VciCalculationWorkflow,
            VciVideoWorkflow,
            ComputeVci,
            ComputeVciSlice,
            InitializeZarrStore,
            LoadDekadIntoZarr,
            ComputeMinMaxPerDekad,
            WriteFparToZarr,
            WriteFparDataIntoEmptyZarr,
            InitializeMinMaxArrays,
            InitializeVciArray,
            CalculateVciDekad,
            CalculateMinMaxForDekad,
            OrchestrateDekadMinMaxCalculation,
            CreateVciMp4,
            CreateVciFrames,
            CreateSingleVciFrame,
            CreateVideoFromFrames,
        ],
        cache=cache,
        cluster=cluster,
    )

    print(f"Starting runner on cluster: {cluster}")
    runner.run_forever()
