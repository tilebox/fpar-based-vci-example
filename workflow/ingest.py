import os
import pickle
from datetime import datetime, timedelta
from urllib.parse import urlparse
from memory_logger import MemoryLogger

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
from zarr.storage import ObjectStore as ZarrObjectStore

from config import (
    COMPRESSOR,
    DATASET_ID,
    DATA_SEARCH_POST_DAYS,
    DATA_SEARCH_PRE_DAYS,
    FILL_VALUE,
    FPAR_NO_DATA_VALUES,
    GCS_BUCKET,
    HEIGHT,
    HEIGHT_CHUNK,
    MODIS_COLLECTION,
    TIME_CHUNK,
    VIIRS_COLLECTION,
    VIIRS_START_DATE,
    WIDTH,
    WIDTH_CHUNK,
    _calc_time_index,
)
# from minmax import InitializeMinMaxArrays
from vci import InitializeVciArray
from zarr_helper import get_job_zarr_prefix
from utils import _query_fpar_metadata, from_param_or_cache


class WriteFparToZarr(Task):
    """
    Writes a single FPAR asset to the Zarr store.
    """

    time_range: str
    zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        init_zarr_store = context.submit_subtask(
            InitializeZarrStore(time_range=self.time_range, zarr_path=self.zarr_path)
        )
        context.submit_subtask(
            WriteFparDataIntoEmptyZarr(zarr_path=self.zarr_path, slice=None),
            depends_on=[init_zarr_store],
        )


class InitializeZarrStore(Task):
    """
    Initializes a Zarr store on GCS to hold the consolidated FPAR datacube.
    """

    time_range: str
    zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing Zarr store...")

        datasets_client = DatasetsClient()
        dataset = datasets_client.dataset(DATASET_ID)
        start_str, end_str = self.time_range.split("/")
        start_time = datetime.fromisoformat(start_str)
        end_time = datetime.fromisoformat(end_str)
        fpar = _query_fpar_metadata(dataset, (start_time, end_time))
        if not fpar:
            logger.info(
                f"No fpar metadata found in the given time range: {self.time_range}"
            )
            return

        num_dekads = fpar.sizes["time"]
        shape = (num_dekads, HEIGHT, WIDTH)
        chunks = (TIME_CHUNK, HEIGHT_CHUNK, WIDTH_CHUNK)
        logger.info(f"Num dekads: {num_dekads}")
        logger.info(f"Shape: {shape}")
        logger.info(f"Chunks: {chunks}")

        # Use a job-specific path to prevent concurrent jobs from interfering.
        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)

        # Initialize the Zarr group and arrays directly. This is more memory-efficient
        # than creating a large in-memory xarray/dask object and writing it.

        # gcp-bucekt/myzarr/

        zarr.create_array(
            store=zarr_store,
            name="fpar",
            shape=shape,
            chunks=chunks,
            dimension_names=["time", "y", "x"],
            dtype="u1",
            compressors=COMPRESSOR,
            fill_value=FILL_VALUE,
        )

        dekads = zarr.create_array(
            store=zarr_store,
            name="dekad",
            shape=(num_dekads),
            dimension_names=["time"],
            dtype=np.uint8,
            compressors=COMPRESSOR,
            fill_value=FILL_VALUE,
        )
        years = zarr.create_array(
            store=zarr_store,
            name="year",
            shape=(num_dekads),
            dimension_names=["time"],
            dtype=np.uint32,
            compressors=COMPRESSOR,
            fill_value=FILL_VALUE,
        )
        urls = zarr.create_array(
            store=zarr_store,
            name="asset_urls",
            shape=(num_dekads),
            chunks=(TIME_CHUNK,),
            dimension_names=["time"],
            dtype="S120",  # asset urls are no longer than 120 chars
        )

        dekads[:] = fpar.dekad.to_numpy()
        years[:] = fpar.year.to_numpy()
        urls[:] = fpar.asset_url.to_numpy()

        logger.info(
            f"Successfully initialized Zarr store at: {self.zarr_path} for {num_dekads} dekads"
        )


class WriteFparDataIntoEmptyZarr(Task):
    """
    Submits a subtask for each year in the time range.
    """

    zarr_path: str
    slice: tuple[int, int] | None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating data loading by year...")

        slice = self.slice
        if slice is None:  # do the whole thing, fetch number of timestamps from zarr
            object_store = GCSStore(
                bucket=GCS_BUCKET,
                prefix=self.zarr_path,
                credential_provider=GoogleCredentialProvider(),
            )
            zarr_store = ZarrObjectStore(object_store)
            dekad_array = zarr.open_group(zarr_store, mode="r")["dekad"]
            num_dekads = dekad_array.shape[0]  # type: ignore[arg-type], we know this is an array, not a subgroup
            slice = (0, num_dekads)

        n_timestamps = slice[1] - slice[0]
        context.current_task.display = f"WriteFpar[{slice[0]}:{slice[1]}]"
        if n_timestamps > 4:
            middle = slice[0] + n_timestamps // 2
            context.submit_subtask(
                WriteFparDataIntoEmptyZarr(
                    zarr_path=self.zarr_path, slice=(slice[0], middle)
                )
            )
            context.submit_subtask(
                WriteFparDataIntoEmptyZarr(
                    zarr_path=self.zarr_path, slice=(middle, slice[1])
                )
            )
            return

        for time_idx in range(slice[0], slice[1]):
            context.submit_subtask(
                LoadDekadIntoZarr(zarr_path=self.zarr_path, time_index=time_idx)
            )


class LoadDekadIntoZarr(Task):
    """
    Loads a single dekad's GeoTIFF into the correct time slice of the Zarr datacube.
    """

    zarr_path: str
    time_index: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        from memory_logger import MemoryLogger
        memory_logger = MemoryLogger()
        memory_logger.log_snapshot(f"Start LoadDekad {self.time_index}")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        group = zarr.open_group(store=zarr_store, mode="a")
        asset_url = group["asset_urls"][self.time_index].item().decode()  # type: ignore[arg-type]

        memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} after open group")
        logger.info(f"Loading asset {asset_url} into time index {self.time_index}...")

        memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} after parse url")
        # 1. Download the file into an in-memory buffer first.
        # This is faster than streaming directly from the URL with rioxarray
        # as it avoids multiple HTTP range requests
        parsed_url = urlparse(asset_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        store = HTTPStore.from_url(base_url)
        file_name = parsed_url.path.lstrip("/")
        obj = store.get(file_name)
        buffer = memoryview(obj.bytes())
        memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} after download")

        processed_arr = np.empty((HEIGHT, WIDTH), dtype=np.uint8)
        # 2. read the tif file from the memory buffer as numpy array
        with rasterio.MemoryFile(buffer) as memfile:
            with memfile.open() as product:
                memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} after read")
                arr = product.read(1)
                # Manually replace flagged no-data values with the standard fill value.
                # mask = np.isin(arr, FPAR_NO_DATA_VALUES)
                # arr[mask] = FILL_VALUE
                arr = np.where((arr == 251) | (arr == 254), FILL_VALUE, arr)
                # arr = np.where(np.isin(arr, FPAR_NO_DATA_VALUES), FILL_VALUE, arr)
                # Select the first band to get a 2D array for writing.
                processed_arr = arr[:, :]
                memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} after process")
                

        # 3. Write the 2D array to the correct Zarr slice.
        memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} before write")
        fpar_array = group["fpar"]
        fpar_array[self.time_index, :, :] = processed_arr  # type: ignore[index]
        memory_logger.log_snapshot(f"Start LoadDekad {self.time_index} after write")
        logger.info(
            f"Successfully loaded asset {asset_url} into time index {self.time_index}."
        )
