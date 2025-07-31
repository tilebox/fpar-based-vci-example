import config
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
    ZARR_STORE_PATH,
    _calc_time_index,
)


class InitializeVciArray(Task):
    """
    Creates the `vci` array in the Zarr store and kicks off the VCI calculation.
    This task is chained to run after the min/max calculation is complete.
    """

    def execute(self, context: ExecutionContext) -> None:
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
        context.submit_subtask(OrchestrateVciByYear())



class OrchestrateVciByYear(Task):
    """
    Retrieves the overall time range from the cache and submits a subtask for each year.
    """

    def execute(self, context: ExecutionContext) -> None:
        dekad_range = pickle.loads(context.job_cache["dekad_range"])
        if not dekad_range:
            return

        start_year = dekad_range["start"][0]
        end_year = dekad_range["end"][0]

        for year in range(start_year, end_year + 1):
            context.submit_subtask(CalculateVciForYear(year=year))


class CalculateVciForYear(Task):
    """
    Submits a subtask for each dekad within a given year.
    """

    year: int

    def execute(self, context: ExecutionContext) -> None:
        dekad_range = pickle.loads(context.job_cache["dekad_range"])
        if not dekad_range:
            return

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


class CalculateVciDekad(Task):
    """
    Calculates the VCI for a single time slice (dekad).
    VCI = (FPAR - FPAR_min) / (FPAR_max - FPAR_min)
    """

    time_index: int

    def execute(self, context: ExecutionContext) -> None:
        zarr_prefix = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET, prefix=zarr_prefix, credential_provider=GoogleCredentialProvider()
        )
        zarr_store = ZarrObjectStore(object_store)  # type: ignore[arg-type]

        ds = xr.open_zarr(zarr_store)

        fpar = ds["fpar"].isel(time=self.time_index)
        fpar_min = ds["min_fpar"]
        fpar_max = ds["max_fpar"]

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
        vci.name = "vci"

        root = zarr.open_group(store=zarr_store, mode="r+", use_consolidated=False)
        vci_array = root["vci"]    

        vci_array[self.time_index, :, :] = vci.values  # type: ignore[index]