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

        if not isinstance(root, zarr.Group):
            raise TypeError(f"Expected a Zarr Group, but got {type(root)}")


        num_dekads = pickle.loads(context.job_cache["num_dekads"])
        shape = pickle.loads(context.job_cache["shape"])
        chunks = pickle.loads(context.job_cache["chunks"])

        vci_array = root.create_array(
            "vci",
            shape=shape,
            chunks=chunks,
            dtype="f4",  # VCI is a float
            compressors=COMPRESSOR,
            fill_value=np.nan,
            overwrite=True,
            dimension_names=["time", "y", "x"],
        )
        # vci_array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
        # vci_array.attrs["_FillValue"] = np.nan

        logger.info("Successfully initialized VCI array.")
        context.submit_subtask(OrchestrateVciCalculation())


class OrchestrateVciCalculation(Task):
    """
    Submits a subtask for each spatial chunk in the datacube.
    This creates a parallel map-reduce job across the spatial domain for the VCI calculation.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating VCI calculation by chunk...")

        num_y_chunks = (HEIGHT + HEIGHT_CHUNK - 1) // HEIGHT_CHUNK
        num_x_chunks = (WIDTH + WIDTH_CHUNK - 1) // WIDTH_CHUNK

        for y_idx in range(num_y_chunks):
            for x_idx in range(num_x_chunks):
                context.submit_subtask(CalculateVciChunk(y_idx=y_idx, x_idx=x_idx))

        logger.info(f"Submitted {num_y_chunks * num_x_chunks} VCI chunk processing tasks.")


class CalculateVciChunk(Task):
    """
    Calculates the VCI for a single spatial chunk across all time slices.
    VCI = (FPAR - FPAR_min) / (FPAR_max - FPAR_min)
    """

    y_idx: int
    x_idx: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Calculating VCI for chunk ({self.y_idx}, {self.x_idx})...")

        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)

        ds = xr.open_zarr(zarr_store)

        y_start = self.y_idx * HEIGHT_CHUNK
        y_end = min((self.y_idx + 1) * HEIGHT_CHUNK, HEIGHT)
        x_start = self.x_idx * WIDTH_CHUNK
        x_end = min((self.x_idx + 1) * WIDTH_CHUNK, WIDTH)

        fpar_chunk = ds["fpar"][:, y_start:y_end, x_start:x_end]
        fpar_min_chunk = ds["min_fpar"][y_start:y_end, x_start:x_end]
        fpar_max_chunk = ds["max_fpar"][y_start:y_end, x_start:x_end]

        # Ensure we are working with floats for the calculation
        fpar_chunk = fpar_chunk.astype("f4")
        fpar_min_chunk = fpar_min_chunk.astype("f4")
        fpar_max_chunk = fpar_max_chunk.astype("f4")

        # Replace the fill value with NaN to ensure correct calculations
        fpar_chunk = fpar_chunk.where(fpar_chunk != FILL_VALUE)
        fpar_min_chunk = fpar_min_chunk.where(fpar_min_chunk != FILL_VALUE)
        fpar_max_chunk = fpar_max_chunk.where(fpar_max_chunk != FILL_VALUE)

        fpar_range = fpar_max_chunk - fpar_min_chunk
        # Use .where to avoid division by zero, placing NaN where the range is zero.
        vci_chunk = ((fpar_chunk - fpar_min_chunk) / fpar_range).where(fpar_range > 0)
        vci_chunk = vci_chunk.astype("f4")

        # Open the Zarr group directly for writing.
        root = zarr.open_group(store=zarr_store, mode="r+")
        vci_array = root["vci"]

        # Define the region to write to using NumPy-style slicing.
        region = (slice(None), slice(y_start, y_end), slice(x_start, x_end))

        # Assign the computed VCI values to the corresponding chunk in the Zarr array.
        # mypy struggles with this dynamic assignment, so we ignore the type error.
        vci_array[region] = vci_chunk.values  # type: ignore[index]

        logger.info(f"Successfully calculated and wrote VCI for chunk ({self.y_idx}, {self.x_idx}).")
