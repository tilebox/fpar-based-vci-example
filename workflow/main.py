import os
from datetime import datetime, timedelta, timezone
import socket
from urllib.parse import urlparse

import dask.array
import numpy as np
import xarray as xr
import rasterio
from obstore.store import HTTPStore
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from tilebox.datasets import Client as DatasetsClient  # type: ignore[import-untyped]
from tilebox.workflows import Task, ExecutionContext, Client as WorkflowsClient  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import configure_console_logging, get_logger  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec
from zarr.storage import ObjectStore as ZarrObjectStore

# --- Constants ---
GCS_BUCKET = "vci-datacube-bucket-1513742"
ZARR_STORE_PATH = "vci_fpar.zarr"
DATASET_ID = "tilebox.modis_fpar"
MODIS_COLLECTION = "MODIS"
VIIRS_COLLECTION = "VIIRS"
FILL_VALUE = 255

# --- Configuration ---
WIDTH = 80640
HEIGHT = 29346
TIME_CHUNK = 1
HEIGHT_CHUNK = 8192
WIDTH_CHUNK = 8192
COMPRESSOR = BloscCodec(cname="lz4hc", clevel=5, shuffle="shuffle")


def _find_closest_datapoint(dataset, time: datetime) -> xr.Dataset:
    collection = VIIRS_COLLECTION
    if time < datetime(2023, 1, 1):  # modis cutoff
        collection = MODIS_COLLECTION
    data = dataset.collection(collection).query(
        temporal_extent=(time - timedelta(days=1), time + timedelta(days=15)),
        skip_data=False, show_progress=False,
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

        logger.info(start_datapoint)
        start_year_dekad = (start_datapoint["year"].item(), start_datapoint["dekad"].item())
        end_year_dekad = (end_datapoint["year"].item(), end_datapoint["dekad"].item())

        num_dekads = _calc_time_index(end_year_dekad[0], end_year_dekad[1], start_year_dekad[0], start_year_dekad[1]) + 1
        logger.info(f"Found {num_dekads} assets for the given time range.")

        shape = (num_dekads, HEIGHT, WIDTH)
        chunks = (TIME_CHUNK, HEIGHT_CHUNK, WIDTH_CHUNK)

        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"  # type: ignore[attr-defined]
        object_store = GCSStore(bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider())
        zarr_store = ZarrObjectStore(object_store)

        coords = {"time": np.arange(num_dekads), "y": np.arange(HEIGHT), "x": np.arange(WIDTH)}
        dims = ("time", "y", "x")

        dummy_data = dask.array.zeros(
                shape,
                chunks=chunks,
                dtype=np.uint8,
            )
        ds = xr.Dataset({"fpar": (dims, dummy_data)}, coords=coords)
        # TODO directly use zarr here to initialize the store, not xarray or dask or numpy
        encoding = {"fpar": {"compressors": tuple([COMPRESSOR]), "chunks": chunks, "_FillValue": FILL_VALUE}}
        ds.to_zarr(zarr_store, mode='w', compute=False, encoding=encoding, consolidated=False, zarr_format=3)  # type: ignore[call-overload]
        
        logger.info(f"Successfully initialized Zarr store at: gs://{GCS_BUCKET}/{ZARR_STORE_PATH}")
        context.submit_subtask(OrchestrateDataLoading(time_range=self.time_range, start_year_dekad=start_year_dekad))


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
            context.submit_subtask(LoadYearData(year=year, time_range=self.time_range, start_year_dekad=self.start_year_dekad))
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
        
        logger.info(f"Querying {collection_name} for year {self.year} in range {query_start_time} to {query_end_time}")
        datapoints = dataset.collection(collection_name).query(
            temporal_extent=(query_start_time, query_end_time),
            show_progress=False,
        )

        for i in range(len(datapoints.time)):
            dp = datapoints.isel(time=i)

            context.submit_subtask(LoadDekadIntoZarr(
                asset_url=str(dp["asset_url"].item()),
                year=int(dp["year"].item()),
                dekad=int(dp["dekad"].item()),
                start_year_dekad=self.start_year_dekad,
            ))
        logger.info(f"Submitted {len(datapoints.time)} subtasks for year {self.year}.")


def _calc_time_index(year: int, dekad: int, start_year: int, start_dekad: int) -> int:
    return (year - start_year) * 36 + (dekad - start_dekad)

def _tests():
    assert _calc_time_index(2010, 20, 2010, 20) == 0
    assert _calc_time_index(2010, 21, 2010, 20) == 1
    assert _calc_time_index(2010, 22, 2010, 20) == 2
    assert _calc_time_index(2010, 35, 2010, 20) == 15
    assert _calc_time_index(2010, 36, 2010, 20) == 16
    assert _calc_time_index(2011, 1, 2010, 20) == 17
    assert _calc_time_index(2012, 1, 2010, 20) == 17 + 36
    assert _calc_time_index(2013, 1, 2010, 20) == 17 + 36 + 36

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
            logger.error(f"Invalid time index for {self.asset_url} for year {self.year} and dekad {self.dekad}")
            return

        logger.info(f"Loading asset {self.asset_url} into time index {time_index} ({self.year}-{self.dekad})...")

        # Download the file into an in-memory buffer
        parsed_url = urlparse(self.asset_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        store = HTTPStore.from_url(base_url)
        file_name = parsed_url.path.lstrip('/')
        obj = store.get(file_name)

        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001
        with tracer.start_span(f"{file_name}/download"):
            buffer = memoryview(obj.bytes())

        # Open the in-memory buffer with rioxarray
        with tracer.start_span(f"{file_name}/read"):
            with rasterio.MemoryFile(buffer) as memfile:
                with memfile.open() as product:
                    arr = product.read()

            # Manually replace flagged no-data values with the standard fill value
            arr = np.where((arr == 251) | (arr == 254), FILL_VALUE, arr)
            ds = xr.Dataset({
                "fpar": (["time", "y", "x"], arr)
            })

        # Write to Zarr store
        with tracer.start_span(f"{file_name}/write"):
            zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"  # type: ignore[attr-defined]
            object_store = GCSStore(bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider())
            zarr_store = ZarrObjectStore(object_store)
            # TODO explore using zarr directly instead of xarray (so we have implicit region mapping etc)
            ds.to_zarr(
                zarr_store,
                region={"time": slice(time_index, time_index + 1)},
                mode="r+",
                consolidated=False,
            ) # type: ignore[call-overload]
            logger.info(f"Successfully loaded asset {self.asset_url} into time index {time_index}.")


if __name__ == "__main__":
    from tilebox.workflows.observability.logging import configure_console_logging, configure_otel_logging_axiom
    from tilebox.workflows.observability.tracing import configure_otel_tracing_axiom

    # Configure logging backends
    configure_console_logging()
    configure_otel_logging_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # # Configure tracing backends
    configure_otel_tracing_axiom(f"{socket.gethostname()}-{os.getpid()}")

    # Start the runner
    client = WorkflowsClient()
    runner = client.runner(tasks=[InitializeZarrStore, OrchestrateDataLoading, LoadYearData, LoadDekadIntoZarr])
    runner.run_forever()
