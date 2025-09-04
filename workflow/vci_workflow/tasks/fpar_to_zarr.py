"""
fpar_to_zarr contains Tilebox tasks for downloading FPAR data (as GeoTIFFs) and writing them into a Zarr store.

FPAR data is downloaded from https://agricultural-production-hotspots.ec.europa.eu/data
"""

import tempfile
from datetime import datetime

import httpx
import numpy as np
import rasterio
import xarray as xr
import zarr
from tilebox.datasets import Client as DatasetsClient
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger

from vci_workflow.memory_logger import MemoryLogger
from vci_workflow.zarr import (
    COMPRESSOR,
    FILL_VALUE,
    HEIGHT,
    HEIGHT_CHUNK,
    TIME_CHUNK,
    WIDTH,
    WIDTH_CHUNK,
    open_zarr_group,
    open_zarr_store,
)

logger = get_logger()
memory_logger = MemoryLogger()

_TILEBOX_DATASET_SLUG = "tilebox.modis_fpar"
_MODIS_COLLECTION = "MODIS"
_VIIRS_COLLECTION = "VIIRS"
_VIIRS_START_DATE = datetime(2023, 1, 1)


class WriteFparToZarr(Task):
    """
    Download, convert and write all FPAR data in a given time range into a Zarr store.
    """

    zarr_path: str
    time_range: str

    def execute(self, context: ExecutionContext) -> None:
        n_time = _initialize_zarr_store(self.zarr_path, self.time_range)
        logger.info(f"Successfully initialized FPAR Zarr store at: {self.zarr_path} for {n_time} dekads")

        full_time_range = (0, n_time)
        context.submit_subtask(WriteFparDataIntoEmptyZarr(self.zarr_path, full_time_range))


def _initialize_zarr_store(zarr_path: str, time_range: str) -> int:
    """
    Initializes the Zarr store with the correct shape and metadata.

    Shape and metadata is inferred by querying the Tilebox FPAR dataset for the given time range.

    Args:
        zarr_path: Path to the Zarr store in the GCS bucket.
        time_range: Time range for which to initialize the Zarr store.

    Returns:
        The time dimension of the Zarr store.
    """
    start_str, end_str = time_range.split("/")
    start_time = datetime.fromisoformat(start_str)
    end_time = datetime.fromisoformat(end_str)
    fpar = _query_fpar_metadata((start_time, end_time))
    if not fpar:
        logger.info(f"No fpar metadata found for the given time range: {time_range}")
        return None

    n_time = fpar.sizes["time"]
    shape = (n_time, HEIGHT, WIDTH)
    chunks = (TIME_CHUNK, HEIGHT_CHUNK, WIDTH_CHUNK)

    zarr_store = open_zarr_store(zarr_path)

    # Initialize the Zarr group and arrays directly. This is more memory-efficient
    # than creating a large in-memory xarray/dask object and writing it.

    zarr.create_array(
        store=zarr_store,
        name="fpar",
        shape=shape,
        chunks=chunks,
        dimension_names=["time", "y", "x"],
        dtype=np.uint8,
        compressors=COMPRESSOR,
        fill_value=FILL_VALUE,
    )

    dekads = zarr.create_array(
        store=zarr_store,
        name="dekad",
        shape=(n_time),
        dimension_names=["time"],
        dtype=np.uint8,
        compressors=COMPRESSOR,
        fill_value=FILL_VALUE,
    )
    years = zarr.create_array(
        store=zarr_store,
        name="year",
        shape=(n_time),
        dimension_names=["time"],
        dtype=np.uint32,
        compressors=COMPRESSOR,
        fill_value=FILL_VALUE,
    )
    urls = zarr.create_array(
        store=zarr_store,
        name="asset_urls",
        shape=(n_time),
        chunks=(TIME_CHUNK,),
        dimension_names=["time"],
        dtype="S120",  # asset urls are no longer than 120 chars
    )

    dekads[:] = fpar.dekad.to_numpy()
    years[:] = fpar.year.to_numpy()
    urls[:] = fpar.asset_url.to_numpy()

    return n_time


def _query_fpar_metadata(time_range: tuple[datetime, datetime]) -> xr.Dataset:
    """
    Queries FPAR metadata within a given time range from a Tilebox dataset.
    """
    datasets_client = DatasetsClient()
    dataset = datasets_client.dataset(_TILEBOX_DATASET_SLUG)
    modis = dataset.collection(_MODIS_COLLECTION).query(
        temporal_extent=(time_range[0], min(time_range[1], _VIIRS_START_DATE))
    )
    viirs = dataset.collection(_VIIRS_COLLECTION).query(
        temporal_extent=(max(time_range[0], _VIIRS_START_DATE), time_range[1])
    )

    non_empty = [ds for ds in [modis, viirs] if ds]
    if len(non_empty) == 0:
        return xr.Dataset()
    if len(non_empty) == 1:
        return non_empty[0]
    return xr.concat(non_empty, dim="time")


class WriteFparDataIntoEmptyZarr(Task):
    """
    Recursively splits the time range into smaller chunks, until we reach leaf tasks for writing each dekad.
    """

    zarr_path: str
    time_index_range: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        logger.info("Orchestrating data loading by year...")

        start = self.time_index_range[0]
        end = self.time_index_range[1]
        n_time = end - start
        context.current_task.display = f"WriteFpar[{start}:{end}, :, :]"

        if n_time > 4:  # more than 4 time indices, so recursively split into subtasks
            middle = start + n_time // 2
            context.submit_subtasks(
                [
                    WriteFparDataIntoEmptyZarr(self.zarr_path, half_range)
                    for half_range in [(start, middle), (middle, end)]
                ]
            )
            return

        # otherwise, submit leaf tasks to actually load each dekad
        for time_index in range(start, end):
            context.submit_subtask(LoadDekadIntoZarr(self.zarr_path, time_index))


class LoadDekadIntoZarr(Task):
    """
    Loads a single dekad's GeoTIFF, converts it into Zarr and writes it into the correct time index of the Zarr cube.
    """

    zarr_path: str
    dekad_index: int

    def execute(self, context: ExecutionContext) -> None:
        context.current_task.display = f"LoadDekadIntoZarr[{self.dekad_index}]"

        group = open_zarr_group(self.zarr_path, mode="a")
        asset_url = group["asset_urls"][self.dekad_index].item().decode()  # type: ignore[arg-type]

        logger.info(f"Loading dekad {self.dekad_index} into {self.zarr_path} from URL {asset_url}")
        memory_logger.log_snapshot()

        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001
        # Download to temporary file to avoid double memory usage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            with (
                tracer.start_span("download_asset"),
                httpx.stream("GET", asset_url, timeout=300.0, follow_redirects=True) as response,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=2 << 20):  # 2MB chunks
                    tmp_file.write(chunk)
            tmp_file.flush()

            logger.info(f"Downloaded dekad {asset_url} to temporary file")
            memory_logger.log_snapshot()

            # Load TIFF from temporary file
            with tracer.start_span("read_tiff"), rasterio.open(tmp_file.name) as src:
                # Validate raster geometry
                if src.width != WIDTH or src.height != HEIGHT:
                    raise ValueError(f"Unexpected raster size: {src.width}x{src.height}, expected {WIDTH}x{HEIGHT}")

                # Read with explicit dtype to avoid hidden casts
                arr = src.read(1, out_dtype="uint8", masked=False)
                logger.info(f"Read raster: {src.height}x{src.width}, dtype={src.dtypes[0]}")
                memory_logger.log_snapshot()

        with tracer.start_span("write_to_zarr"):
            # Manually replace flagged no-data values with the standard fill value.
            arr[arr == 251] = FILL_VALUE
            arr[arr == 254] = FILL_VALUE

            # Write to Zarr
            group["fpar"][self.dekad_index, :, :] = arr  # type: ignore[arg-type]

        group.store.close()
        logger.info(f"Successfully wrote dekad {self.dekad_index} to Zarr store. Asset URL: {asset_url}")
        memory_logger.log_snapshot()
