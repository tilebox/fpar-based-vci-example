import numpy as np
import xarray as xr
import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from tilebox.workflows import ExecutionContext, Task  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import get_logger  # type: ignore[import-untyped]
from zarr.storage import ObjectStore as ZarrObjectStore

from config import (
    FILL_VALUE,
    GCS_BUCKET,
    HEIGHT,
    HEIGHT_CHUNK,
    WIDTH,
    WIDTH_CHUNK,
    ZARR_STORE_PATH,
)


class InitializeMinMaxArrays(Task):
    """
    Creates and initializes the min_fpar and max_fpar arrays in the Zarr store.
    This task is chained to run after the main data ingestion is complete.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing min/max FPAR arrays...")

        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"  # type: ignore[attr-defined]
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)
        root = zarr.group(store=zarr_store, overwrite=False)

        shape = (HEIGHT, WIDTH)
        chunks = (HEIGHT_CHUNK, WIDTH_CHUNK)

        # Create min_fpar array, initialized to the max possible uint8 value.
        # The fill_value argument is the most efficient way to initialize an array.
        min_fpar = root.create_array(
            "min_fpar", shape=shape, chunks=chunks, dtype="u1", overwrite=True, fill_value=FILL_VALUE, dimension_names=["y", "x"],
        )
        min_fpar.attrs["_FillValue"] = FILL_VALUE

        # Create max_fpar array, initialized to zero.
        max_fpar = root.create_array(
            "max_fpar", shape=shape, chunks=chunks, dtype="u1", overwrite=True, fill_value=0, dimension_names=["y", "x"],
        )
        max_fpar.attrs["_FillValue"] = FILL_VALUE

        logger.info("Successfully initialized min/max arrays.")
        context.submit_subtask(OrchestrateMinMaxCalculation())


class OrchestrateMinMaxCalculation(Task):
    """
    Submits a subtask for each spatial chunk in the datacube.
    This creates a parallel map-reduce job across the spatial domain.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating min/max calculation by chunk...")

        num_y_chunks = (HEIGHT + HEIGHT_CHUNK - 1) // HEIGHT_CHUNK
        num_x_chunks = (WIDTH + WIDTH_CHUNK - 1) // WIDTH_CHUNK

        for y_idx in range(num_y_chunks):
            for x_idx in range(num_x_chunks):
                context.submit_subtask(CalculateChunkMinMax(y_idx=y_idx, x_idx=x_idx))

        logger.info(f"Submitted {num_y_chunks * num_x_chunks} chunk processing tasks.")


class CalculateChunkMinMax(Task):
    """
    Calculates the min and max for a single spatial chunk across the time dimension.
    """

    y_idx: int
    x_idx: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Calculating min/max for chunk ({self.y_idx}, {self.x_idx})...")

        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"  # type: ignore[attr-defined]
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)

        # Open the entire dataset with xarray. Dask lazy-loads the data, so this is efficient.
        ds = xr.open_zarr(zarr_store, consolidated=False)
        fpar_array = ds["fpar"]

        y_start = self.y_idx * HEIGHT_CHUNK
        y_end = min((self.y_idx + 1) * HEIGHT_CHUNK, HEIGHT)
        x_start = self.x_idx * WIDTH_CHUNK
        x_end = min((self.x_idx + 1) * WIDTH_CHUNK, WIDTH)

        # Select the spatial chunk for all time points.
        chunk_selection = fpar_array[:, y_start:y_end, x_start:x_end]

        # Use xarray's built-in, optimized methods to calculate min/max along the time dimension.
        # skipna=True correctly handles the _FillValue. The result is a 2D DataArray.
        min_values = chunk_selection.min(dim="time", skipna=True).astype("u1")
        max_values = chunk_selection.max(dim="time", skipna=True).astype("u1")

        root = zarr.open_group(store=zarr_store, mode="r+", use_consolidated=False)
        min_array = root["min_fpar"]    
        max_array = root["max_fpar"]

        # Write the resulting 2D chunks directly to the corresponding regions in the Zarr arrays.
        min_array[y_start:y_end, x_start:x_end] = min_values.values  # type: ignore[index]
        max_array[y_start:y_end, x_start:x_end] = max_values.values  # type: ignore[index]

        logger.info(f"Successfully calculated and wrote min/max for chunk ({self.y_idx}, {self.x_idx}).")
