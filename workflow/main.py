import os
import pickle
import socket
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import numpy as np
import rasterio  # type: ignore[import-untyped]
import xarray as xr
import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore, HTTPStore
from tilebox.datasets import Client as DatasetsClient  # type: ignore[import-untyped]
from tilebox.workflows import Client as WorkflowsClient  # type: ignore[import-untyped]
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import (  # type: ignore[import-untyped]
    configure_console_logging,
    get_logger,
)
from zarr.codecs import BloscCodec
from zarr.storage import ObjectStore as ZarrObjectStore

from config import (
    COMPRESSOR,
    DATASET_ID,
    FILL_VALUE,
    GCS_BUCKET,
    HEIGHT,
    HEIGHT_CHUNK,
    MODIS_COLLECTION,
    TIME_CHUNK,
    VIIRS_COLLECTION,
    WIDTH,
    WIDTH_CHUNK,
    ZARR_STORE_PATH,
    _calc_time_index,
)
from minmax import InitializeMinMaxArrays
from vci import InitializeVciArray


def _find_closest_datapoint(dataset, time: datetime) -> xr.Dataset:
    """
    Finds the first available data point in the dataset after a given time.
    This is used to determine the precise start and end dekads for the workflow.
    """
    collection = VIIRS_COLLECTION
    if time < datetime(2023, 1, 1):  # MODIS/VIIRS cutoff
        collection = MODIS_COLLECTION
    # Query a small window to find the next available dekad
    data = dataset.collection(collection).query(
        temporal_extent=(time - timedelta(days=1), time + timedelta(days=15)),
        skip_data=False,
        show_progress=False,
    )
    return data.isel(time=0)




class InitializeZarrStore(Task):
    """
    Initializes a Zarr store on GCS to hold the consolidated FPAR datacube.
    """

    time_range: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing Zarr store...")

        datasets_client = DatasetsClient()
        dataset = datasets_client.dataset(DATASET_ID)
        start_str, end_str = self.time_range.split("/")
        start_time = datetime.fromisoformat(start_str)
        end_time = datetime.fromisoformat(end_str)

        start_datapoint = _find_closest_datapoint(dataset, start_time)
        end_datapoint = _find_closest_datapoint(dataset, end_time)

        start_year_dekad = (start_datapoint["year"].item(), start_datapoint["dekad"].item())
        end_year_dekad = (end_datapoint["year"].item(), end_datapoint["dekad"].item())

        

        num_dekads = (
            _calc_time_index(
                end_year_dekad[0], end_year_dekad[1], start_year_dekad[0], start_year_dekad[1]
            )
            + 1
        )
        logger.info(f"Found {num_dekads} assets for the given time range.")

        shape = (num_dekads, HEIGHT, WIDTH)
        chunks = (TIME_CHUNK, HEIGHT_CHUNK, WIDTH_CHUNK)
        context.job_cache["num_dekads"] = pickle.dumps(num_dekads)
        context.job_cache["shape"] = pickle.dumps(shape)
        context.job_cache["chunks"] = pickle.dumps(chunks)

        # Use a job-specific path to prevent concurrent jobs from interfering.
        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"  # type: ignore[attr-defined]
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)

        # Initialize the Zarr group and arrays directly. This is more memory-efficient
        # than creating a large in-memory xarray/dask object and writing it.
        root = zarr.group(store=zarr_store, overwrite=True)
        root.create_array(
            "fpar",
            shape=shape,
            chunks=chunks,
            dimension_names=["time", "y", "x"],
            dtype="u1",
            compressors=COMPRESSOR,
            fill_value=FILL_VALUE,
        )

        # Save the start and end dekads to the cache for the VCI calculation tasks.
        # The cache is automatically scoped to the job, so we can use a simple key.
        context.job_cache["dekad_range"] = pickle.dumps(
            {"start": start_year_dekad, "end": end_year_dekad}
        )

        logger.info(f"Successfully initialized Zarr store at: gs://{GCS_BUCKET}/{zarr_prefix}")
        data_loading_task = context.submit_subtask(
            OrchestrateDataLoading(time_range=self.time_range, start_year_dekad=start_year_dekad)
        )

        # Chain the min/max calculation to run after all data loading is complete.
        min_max_task = context.submit_subtask(InitializeMinMaxArrays(), depends_on=[data_loading_task])

        # Chain the VCI calculation to run after the min/max calculation is complete.
        context.submit_subtask(InitializeVciArray(), depends_on=[min_max_task])



class OrchestrateDataLoading(Task):
    """
    Submits a subtask for each year in the time range.
    """

    time_range: str
    start_year_dekad: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating data loading by year...")
        start_str, end_str = self.time_range.split("/")
        start_year = datetime.fromisoformat(start_str).year
        end_year = datetime.fromisoformat(end_str).year

        for year in range(start_year, end_year + 1):
            context.submit_subtask(
                LoadYearData(
                    year=year, time_range=self.time_range, start_year_dekad=self.start_year_dekad
                )
            )
        logger.info(f"Submitted tasks for years {start_year} to {end_year}.")


class LoadYearData(Task):
    """
    Queries for a single year and submits subtasks for each dekad in that year.
    """

    year: int
    time_range: str
    start_year_dekad: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Loading data for year {self.year}...")

        datasets_client = DatasetsClient()
        dataset = datasets_client.dataset(DATASET_ID)
        start_str, end_str = self.time_range.split("/")
        workflow_start_time = datetime.fromisoformat(start_str)
        workflow_end_time = datetime.fromisoformat(end_str)

        year_start_time = datetime(self.year, 1, 1)
        year_end_time = datetime(self.year + 1, 1, 1)

        query_start_time = max(workflow_start_time, year_start_time)
        query_end_time = min(workflow_end_time, year_end_time)

        if query_start_time >= query_end_time:
            logger.info(f"No data to load for year {self.year} within the given time range.")
            return

        collection_name = MODIS_COLLECTION if self.year <= 2022 else VIIRS_COLLECTION

        logger.info(
            f"Querying {collection_name} for year {self.year} in range {query_start_time} to {query_end_time}"
        )
        datapoints = dataset.collection(collection_name).query(
            temporal_extent=(query_start_time, query_end_time),
            show_progress=False,
        )

        for i in range(len(datapoints.time)):
            dp = datapoints.isel(time=i)
            context.submit_subtask(
                LoadDekadIntoZarr(
                    asset_url=str(dp["asset_url"].item()),
                    year=int(dp["year"].item()),
                    dekad=int(dp["dekad"].item()),
                    start_year_dekad=self.start_year_dekad,
                )
            )
        logger.info(f"Submitted {len(datapoints.time)} subtasks for year {self.year}.")


class LoadDekadIntoZarr(Task):
    """
    Loads a single dekad's GeoTIFF into the correct time slice of the Zarr datacube.
    """

    asset_url: str
    year: int
    dekad: int
    start_year_dekad: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        time_index = _calc_time_index(self.year, self.dekad, *self.start_year_dekad)
        logger = get_logger()
        if time_index < 0:
            logger.error(
                f"Invalid time index for {self.asset_url} for year {self.year} and dekad {self.dekad}"
            )
            return

        logger.info(
            f"Loading asset {self.asset_url} into time index {time_index} ({self.year}-{self.dekad})..."
        )

        # 1. Download the file into an in-memory buffer first.
        # This is significantly faster than streaming directly from the URL with rioxarray,
        # as it avoids multiple HTTP range requests.
        parsed_url = urlparse(self.asset_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        store = HTTPStore.from_url(base_url)
        file_name = parsed_url.path.lstrip("/")
        obj = store.get(file_name)
        buffer = memoryview(obj.bytes())

        # 2. Open the in-memory buffer and process with NumPy.
        with rasterio.MemoryFile(buffer) as memfile:
            with memfile.open() as product:
                # Read the data. rasterio returns a 3D array (band, y, x).
                arr = product.read()
                # Manually replace flagged no-data values (251, 254) with the standard fill value.
                arr = np.where((arr == 251) | (arr == 254), FILL_VALUE, arr)
                # Select the first band to get a 2D array for writing.
                processed_arr = arr[0, :, :]

        # 3. Write the 2D array to the correct Zarr slice.
        # Using zarr directly for writing is more explicit and avoids the overhead
        # of creating an xarray object just for the write operation.
        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"  # type: ignore[attr-defined]
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)
        root = zarr.open_group(store=zarr_store, mode="r+")
        fpar_array = root["fpar"]
        # Use standard NumPy-style slicing, which is highly efficient.
        # mypy struggles with this dynamic assignment, so we ignore the type error.
        fpar_array[time_index, :, :] = processed_arr  # type: ignore[index]

        logger.info(f"Successfully loaded asset {self.asset_url} into time index {time_index}.")


if __name__ == "__main__":
    from google.cloud.storage import Client as StorageClient  # type: ignore[import-untyped]
    from minmax import (
        CalculateChunkMinMax,
        InitializeMinMaxArrays,
        OrchestrateMinMaxCalculation,
    )
    from tilebox.workflows.cache import GoogleStorageCache  # type: ignore[import-untyped]
    from tilebox.workflows.observability.logging import (
        configure_console_logging,
        configure_otel_logging_axiom,
    )
    from tilebox.workflows.observability.tracing import (  # type: ignore[import-untyped]
        configure_otel_tracing_axiom,
    )
    from vci import (
        CalculateVciChunk,
        InitializeVciArray,
        OrchestrateVciCalculation,
    )

    # Configure logging backends
    configure_console_logging()
    configure_otel_logging_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # Configure tracing backends
    configure_otel_tracing_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # Configure a GCS-backed cache for sharing metadata between tasks.
    storage_client = StorageClient()
    gcs_bucket = storage_client.bucket(GCS_BUCKET)
    cache = GoogleStorageCache(gcs_bucket, prefix="vci_workflow_cache")

    # Start the runner
    client = WorkflowsClient()
    runner = client.runner(
        tasks=[
            InitializeZarrStore,
            OrchestrateDataLoading,
            LoadYearData,
            LoadDekadIntoZarr,
            InitializeMinMaxArrays,
            OrchestrateMinMaxCalculation,
            CalculateChunkMinMax,
            InitializeVciArray,
            OrchestrateVciCalculation,
            CalculateVciChunk,
        ],
        cache=cache,
    )
    runner.run_forever()
