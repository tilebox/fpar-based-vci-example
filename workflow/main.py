import os
from datetime import datetime

import numpy as np
import rioxarray
import xarray as xr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from tilebox.datasets import Client as DatasetsClient
from tilebox.workflows import Task, ExecutionContext, Client as WorkflowsClient
from tilebox.workflows.observability.logging import configure_console_logging, get_logger
from zarr.codecs import BloscCodec
from zarr.storage import ObjectStore as ZarrObjectStore

# --- Constants ---
GCS_BUCKET = "vci-datacube-bucket-1513742"
ZARR_STORE_PATH = "vci_fpar.zarr"
DATASET_ID = "tilebox.modis_fpar"
MODIS_COLLECTION = "MODIS"
VIIRS_COLLECTION = "VIIRS"

# --- Configuration ---
WIDTH = 80640
HEIGHT = 29346
TIME_CHUNK = 1
HEIGHT_CHUNK = 4819
WIDTH_CHUNK = 2016
COMPRESSOR = BloscCodec(cname="lz4hc", clevel=5, shuffle="shuffle")


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

        # MODIS timeframe
        modis_start_time = max(start_time, datetime(2000, 1, 1))
        modis_end_time = min(end_time, datetime(2023, 1, 1))

        # VIIRS timeframe
        viirs_start_time = max(start_time, datetime(2023, 1, 1))
        viirs_end_time = min(end_time, datetime(datetime.now().year + 1, 1, 1))

        num_modis_dekads = 0
        if modis_start_time < modis_end_time:
            modis_dekads = dataset.collection(MODIS_COLLECTION).query(
                temporal_extent=(modis_start_time, modis_end_time),
                skip_data=True, show_progress=False,
            )
            num_modis_dekads = len(modis_dekads.time.values)

        num_viirs_dekads = 0
        if viirs_start_time < viirs_end_time:
            viirs_dekads = dataset.collection(VIIRS_COLLECTION).query(
                temporal_extent=(viirs_start_time, viirs_end_time),
                skip_data=True, show_progress=False,
            )
            num_viirs_dekads = len(viirs_dekads.time.values)

        num_dekads = num_modis_dekads + num_viirs_dekads
        logger.info(f"Found {num_dekads} assets for the given time range.")

        shape = (num_dekads, HEIGHT, WIDTH)
        chunks = (TIME_CHUNK, HEIGHT_CHUNK, WIDTH_CHUNK)

        object_store = GCSStore(bucket=GCS_BUCKET, prefix=ZARR_STORE_PATH, credential_provider=GoogleCredentialProvider())
        zarr_store = ZarrObjectStore(object_store)

        coords = {"time": np.arange(num_dekads), "y": np.arange(HEIGHT), "x": np.arange(WIDTH)}
        dims = ("time", "y", "x")

        dummy_data = np.empty(shape, dtype=np.float32)
        ds = xr.Dataset({"fpar": (dims, dummy_data)}, coords=coords)

        encoding = {"fpar": {"compressors": (COMPRESSOR,), "chunks": chunks}}
        ds.to_zarr(zarr_store, mode='w', compute=False, encoding=encoding, zarr_format=3)  # type: ignore[arg-type]

        logger.info(f"Successfully initialized Zarr store at: gs://{GCS_BUCKET}/{ZARR_STORE_PATH}")
        context.submit_subtask(OrchestrateDataLoading(time_range=self.time_range))


class OrchestrateDataLoading(Task):
    """
    Submits a subtask for each year in the time range.
    """
    time_range: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating data loading by year...")
        start_str, end_str = self.time_range.split("/")
        start_year = datetime.fromisoformat(start_str).year
        end_year = datetime.fromisoformat(end_str).year

        for year in range(start_year, end_year + 1):
            context.submit_subtask(LoadYearData(year=year, time_range=self.time_range))
        logger.info(f"Submitted tasks for years {start_year} to {end_year}.")


class LoadYearData(Task):
    """
    Queries for a single year and submits subtasks for each dekad in that year.
    """
    year: int
    time_range: str

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
            # Convert numpy.datetime64 to python datetime, then to string
            dt64 = dp.time.values
            ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            timestamp = datetime.fromtimestamp(ts, tz=datetime.timezone.utc)

            context.submit_subtask(LoadDekadIntoZarr(
                asset_url=str(dp["asset_url"].values),
                timestamp_str=timestamp.isoformat(),
                time_range=self.time_range
            ))
        logger.info(f"Submitted {len(datapoints.time)} subtasks for year {self.year}.")


class LoadDekadIntoZarr(Task):
    """
    Loads a single dekad's GeoTIFF into the correct time slice of the Zarr datacube.
    """
    asset_url: str
    timestamp_str: str
    time_range: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        timestamp = datetime.fromisoformat(self.timestamp_str)

        datasets_client = DatasetsClient()
        dataset = datasets_client.dataset(DATASET_ID)
        start_str, end_str = self.time_range.split("/")
        start_time = datetime.fromisoformat(start_str)
        end_time = datetime.fromisoformat(end_str)

        # This logic must exactly mirror InitializeZarrStore
        modis_start_time = max(start_time, datetime(2000, 1, 1))
        modis_end_time = min(end_time, datetime(2023, 1, 1))
        viirs_start_time = max(start_time, datetime(2023, 1, 1))
        viirs_end_time = min(end_time, datetime(datetime.now().year + 1, 1, 1))

        all_times = []
        if modis_start_time < modis_end_time:
            modis_dekads = dataset.collection(MODIS_COLLECTION).query(
                temporal_extent=(modis_start_time, modis_end_time),
                skip_data=True, show_progress=False,
            )
            all_times.extend(modis_dekads.time.values)

        if viirs_start_time < viirs_end_time:
            viirs_dekads = dataset.collection(VIIRS_COLLECTION).query(
                temporal_extent=(viirs_start_time, viirs_end_time),
                skip_data=True, show_progress=False,
            )
            all_times.extend(viirs_dekads.time.values)
        
        all_times.sort()

        time_index = -1
        timestamp_np = np.datetime64(timestamp)
        for i, dp_time in enumerate(all_times):
            if dp_time == timestamp_np:
                time_index = i
                break
        
        if time_index == -1:
            logger.error(f"Could not find time index for asset {self.asset_url}")
            return

        logger.info(f"Loading asset {self.asset_url} into time index {time_index}...")

        object_store = GCSStore(bucket=GCS_BUCKET, prefix=ZARR_STORE_PATH, credential_provider=GoogleCredentialProvider())
        zarr_store = ZarrObjectStore(object_store)

        data = rioxarray.open_rasterio(self.asset_url, masked=True)
        if isinstance(data, list):
            data = data[0]
        
        # The GeoTIFFs use specific values for no data, land, and water.
        # We want to treat all of these as NaN.
        # rioxarray.open_rasterio with masked=True should handle the _FillValue.
        # We will manually mask the other values.
        data = data.where(data != 251) # other land
        data = data.where(data != 254) # water
        
        # Remove conflicting attributes
        if "_FillValue" in data.attrs:
            del data.attrs["_FillValue"]

        data = data.squeeze('band', drop=True)
        ds = data.to_dataset(name="fpar").expand_dims("time")
        
        # Drop non-time coordinates before writing to a region
        ds = ds.drop_vars(['x', 'y', 'spatial_ref'])

        ds.to_zarr(
            zarr_store,  # type: ignore[arg-type]
            region={"time": slice(time_index, time_index + 1)},
            mode="r+",
            zarr_format=3,
            consolidated=False,
        )
        logger.info(f"Successfully loaded asset {self.asset_url} into time index {time_index}.")


if __name__ == "__main__":
    configure_console_logging()
    client = WorkflowsClient()
    runner = client.runner(tasks=[InitializeZarrStore, OrchestrateDataLoading, LoadYearData, LoadDekadIntoZarr])
    runner.run_forever()
