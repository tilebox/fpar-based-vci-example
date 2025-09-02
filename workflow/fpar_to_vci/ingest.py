import tempfile
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import rasterio
import xarray as xr
import zarr

from fpar_to_vci.config import (
    COMPRESSOR,
    FILL_VALUE,
    HEIGHT,
    HEIGHT_CHUNK,
    TIME_CHUNK,
    WIDTH,
    WIDTH_CHUNK,
)
from fpar_to_vci.memory_logger import MemoryLogger
from fpar_to_vci.zarr_helpers import open_zarr_group, open_zarr_store
from tilebox.datasets import Client as DatasetsClient
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger

logger = get_logger()
memory_logger = MemoryLogger()

_TILEBOX_DATASET_SLUG = "tilebox.modis_fpar"
_MODIS_COLLECTION = "MODIS"
_VIIRS_COLLECTION = "VIIRS"
_VIIRS_START_DATE = datetime(2023, 1, 1)


class WriteFparToZarr(Task):
    """
    Write FPAR data in a given time range into a Zarr store.
    """

    time_range: str
    zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        init_store_task = context.submit_subtask(
            InitializeZarrStore(time_range=self.time_range, zarr_path=self.zarr_path)
        )
        context.submit_subtask(
            WriteFparDataIntoEmptyZarr(zarr_path=self.zarr_path, dekad_range=None),
            depends_on=[init_store_task],
        )


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


class InitializeZarrStore(Task):
    """
    Initializes a Zarr store on GCS to hold the consolidated FPAR datacube.
    """

    time_range: str
    zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger.info("Initializing Zarr store...")

        start_str, end_str = self.time_range.split("/")
        start_time = datetime.fromisoformat(start_str)
        end_time = datetime.fromisoformat(end_str)
        fpar = _query_fpar_metadata((start_time, end_time))
        if not fpar:
            logger.info(f"No fpar metadata found for the given time range: {self.time_range}")
            return

        num_dekads = fpar.sizes["time"]
        shape = (num_dekads, HEIGHT, WIDTH)
        chunks = (TIME_CHUNK, HEIGHT_CHUNK, WIDTH_CHUNK)

        zarr_store = open_zarr_store(self.zarr_path)

        # Initialize the Zarr group and arrays directly. This is more memory-efficient
        # than creating a large in-memory xarray/dask object and writing it.

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

        logger.info(f"Successfully initialized Zarr store at: {self.zarr_path} for {num_dekads} dekads")


class WriteFparDataIntoEmptyZarr(Task):
    """
    Submits a subtask for each year in the time range.
    """

    zarr_path: str
    dekad_range: tuple[int, int] | None

    def execute(self, context: ExecutionContext) -> None:
        logger.info("Orchestrating data loading by year...")

        dekad_range = self.dekad_range
        if dekad_range is None:
            group = open_zarr_group(self.zarr_path, mode="r")
            dekad_array = group["dekad"]
            num_dekads = dekad_array.shape[0]  # type: ignore[arg-type], we know this is an array, not a subgroup
            dekad_range = (0, num_dekads)

        n_timestamps = dekad_range[1] - dekad_range[0]
        context.current_task.display = f"WriteFpar[{dekad_range[0]}:{dekad_range[1]}]"
        if n_timestamps > 4:  # more than 4 time indices, so recursively split into subtasks
            middle = dekad_range[0] + n_timestamps // 2
            context.submit_subtask(
                WriteFparDataIntoEmptyZarr(zarr_path=self.zarr_path, dekad_range=(dekad_range[0], middle))
            )
            context.submit_subtask(
                WriteFparDataIntoEmptyZarr(zarr_path=self.zarr_path, dekad_range=(middle, dekad_range[1]))
            )
            return

        # otherwise, submit leaf tasks to actually load each dekad
        for time_idx in range(dekad_range[0], dekad_range[1]):
            context.submit_subtask(LoadDekadIntoZarr(zarr_path=self.zarr_path, dekad_index=time_idx))


class LoadDekadIntoZarr(Task):
    """
    Loads a single dekad's GeoTIFF, converts it into Zarr and writes it into the correct time index of the Zarr cube.
    """

    zarr_path: str
    dekad_index: int

    def execute(self, context: ExecutionContext) -> None:
        context.current_task.display = f"LoadDekadIntoZarr[{self.dekad_index}]"
        memory_logger = MemoryLogger()
        logger.info(f"loading dekad {self.dekad_index} into {self.zarr_path}")
        memory_logger.log_snapshot()

        group = open_zarr_group(self.zarr_path, mode="a")
        asset_url = group["asset_urls"][self.dekad_index].item().decode()  # type: ignore[arg-type]

        logger.info(f"Loading dekad {self.dekad_index}: {asset_url}")
        memory_logger.log_snapshot()

        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001
        # Download to temporary file to avoid double memory usage
        with tracer.start_span("download_asset"), tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            with httpx.stream("GET", asset_url, timeout=300.0, follow_redirects=True) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=2 << 20):  # 2MB chunks
                    tmp_file.write(chunk)
            tmp_file.flush()
            local_path = tmp_file.name

        logger.info(f"Downloaded dekad {asset_url} to temporary file")
        memory_logger.log_snapshot()

        # Load TIFF from temporary file
        with tracer.start_span("read_tiff"), rasterio.open(local_path) as src:
            # Validate raster geometry
            if src.width != WIDTH or src.height != HEIGHT:
                raise ValueError(f"Unexpected raster size: {src.width}x{src.height}, expected {WIDTH}x{HEIGHT}")

            # Read with explicit dtype to avoid hidden casts
            arr = src.read(1, out_dtype="uint8", masked=False)
            logger.info(f"Reading raster: {src.width}x{src.height}, dtype={src.dtypes[0]}")
            memory_logger.log_snapshot()

        # Clean up temp file immediately
        Path(local_path).unlink()

        with tracer.start_span("write_to_zarr"):
            # Manually replace flagged no-data values with the standard fill value.
            arr[arr == 251] = FILL_VALUE
            arr[arr == 254] = FILL_VALUE

            # Write to Zarr
            group["fpar"][self.dekad_index, :, :] = arr  # type: ignore[arg-type]

        group.store.close()
        logger.info(f"Successfully wrote dekad {self.dekad_index} to Zarr store. Asset URL: {asset_url}")
        memory_logger.log_snapshot()
