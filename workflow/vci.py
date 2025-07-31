import pickle
import numpy as np
import xarray as xr
import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from tilebox.workflows import ExecutionContext, Task  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import get_logger  # type: ignore[import-untyped]
from zarr.storage import ObjectStore as ZarrObjectStore

from config import (
    COMPRESSOR,
    FILL_VALUE,
    GCS_BUCKET,
    HEIGHT,
    HEIGHT_CHUNK,
    WIDTH,
    WIDTH_CHUNK,
    ZARR_STORE_PATH,
    _calc_time_index,
)


class InitializeVciArray(Task):
    """
    Creates the `vci` array in the Zarr store and kicks off the VCI calculation.
    This task is chained to run after the min/max calculation is complete.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing VCI array...")

        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)
        root = zarr.open_group(store=zarr_store, mode="r+")

        if not isinstance(root, zarr.Group):
            raise TypeError(f"Expected a Zarr Group, but got {type(root)}")

        # Read shape and chunks from the cache
        shape = pickle.loads(context.job_cache["shape"])
        chunks = pickle.loads(context.job_cache["chunks"])

        root.create_array(
            "vci",
            shape=shape,
            chunks=chunks,
            dtype="f4",  # VCI is a float
            compressors=COMPRESSOR,
            fill_value=np.nan,
            overwrite=True,
            dimension_names=["time", "y", "x"],
        )

        logger.info("Successfully initialized VCI array.")
        context.submit_subtask(OrchestrateVciByYear())


class OrchestrateVciByYear(Task):
    """
    Retrieves the overall time range from the cache and submits a subtask for each year.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating VCI calculation by year...")

        dekad_range = pickle.loads(context.job_cache["dekad_range"])

        start_year = dekad_range["start"][0]
        end_year = dekad_range["end"][0]

        for year in range(start_year, end_year + 1):
            context.submit_subtask(CalculateVciForYear(year=year))

        logger.info(f"Submitted VCI calculation tasks for years {start_year} to {end_year}.")


class CalculateVciForYear(Task):
    """
    Submits a subtask for each dekad within a given year.
    """

    year: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Calculating VCI for year {self.year}...")

        dekad_range = pickle.loads(context.job_cache["dekad_range"])

        start_year_dekad = dekad_range["start"]
        end_year_dekad = dekad_range["end"]

        # Determine the dekad range for the current year
        start_dekad = 1
        if self.year == start_year_dekad[0]:
            start_dekad = start_year_dekad[1]

        end_dekad = 36
        if self.year == end_year_dekad[0]:
            end_dekad = end_year_dekad[1]

        for dekad in range(start_dekad, end_dekad + 1):
            time_index = _calc_time_index(self.year, dekad, *start_year_dekad)
            if time_index >= 0:
                context.submit_subtask(CalculateVciDekad(time_index=time_index))

        logger.info(f"Submitted VCI tasks for dekads {start_dekad}-{end_dekad} in year {self.year}.")


class CalculateVciDekad(Task):
    """
    Orchestrates the VCI calculation for a single time slice (dekad) by submitting
    spatial chunk tasks for parallel processing.
    """

    time_index: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Orchestrating VCI calculation for time index {self.time_index}...")

        num_y_chunks = (HEIGHT + HEIGHT_CHUNK - 1) // HEIGHT_CHUNK
        num_x_chunks = (WIDTH + WIDTH_CHUNK - 1) // WIDTH_CHUNK

        for y_idx in range(num_y_chunks):
            for x_idx in range(num_x_chunks):
                context.submit_subtask(CalculateVciChunk(time_index=self.time_index, y_idx=y_idx, x_idx=x_idx))

        logger.info(f"Submitted {num_y_chunks * num_x_chunks} VCI chunk tasks for time index {self.time_index}.")


class CalculateVciChunk(Task):
    """
    Calculates the VCI for a single spatial chunk at a specific time slice.
    VCI = (FPAR - FPAR_min) / (FPAR_max - FPAR_min)
    """

    time_index: int
    y_idx: int
    x_idx: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Calculating VCI for time index {self.time_index}, chunk ({self.y_idx}, {self.x_idx})...")

        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)

        ds = xr.open_zarr(zarr_store, consolidated=False)

        # Calculate spatial chunk boundaries
        y_start = self.y_idx * HEIGHT_CHUNK
        y_end = min((self.y_idx + 1) * HEIGHT_CHUNK, HEIGHT)
        x_start = self.x_idx * WIDTH_CHUNK
        x_end = min((self.x_idx + 1) * WIDTH_CHUNK, WIDTH)

        # Select the spatial chunk for the specific time slice
        fpar = ds["fpar"].isel(time=self.time_index)[y_start:y_end, x_start:x_end]
        fpar_min = ds["min_fpar"][y_start:y_end, x_start:x_end]
        fpar_max = ds["max_fpar"][y_start:y_end, x_start:x_end]

        # Ensure we are working with floats for the calculation
        fpar = fpar.astype("f4")
        fpar_min = fpar_min.astype("f4")
        fpar_max = fpar_max.astype("f4")

        # Replace the fill value with NaN to ensure correct calculations
        fpar = fpar.where(fpar != FILL_VALUE)
        fpar_min = fpar_min.where(fpar_min != FILL_VALUE)
        fpar_max = fpar_max.where(fpar_max != FILL_VALUE)

        fpar_range = fpar_max - fpar_min
        # Use .where to avoid division by zero, placing NaN where the range is zero.
        vci = ((fpar - fpar_min) / fpar_range).where(fpar_range > 0)
        vci = vci.astype("f4")

        # Open the Zarr group directly for writing.
        root = zarr.open_group(store=zarr_store, mode="r+", use_consolidated=False)
        vci_array = root["vci"]

        # Write the computed VCI values to the corresponding spatial chunk in the Zarr array.
        vci_array[self.time_index, y_start:y_end, x_start:x_end] = vci.values  # type: ignore[index]

        logger.info(f"Successfully calculated and wrote VCI for time index {self.time_index}, chunk ({self.y_idx}, {self.x_idx}).")
