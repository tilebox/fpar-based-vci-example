from functools import lru_cache
import pickle
from typing import Any, Dict, Literal
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
    _calc_year_dekad_from_time_index,
)
from minmax import SpatialChunk
from utils import from_param_or_cache


class ComputeVci(Task):
    """
    Computes the VCI for each time slice and stores it in the Zarr store.
    """

    fpar_zarr_path: str
    min_max_zarr_path: str
    vci_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Computing VCI...")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.fpar_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        fpar_zarr_store = ZarrObjectStore(object_store)
        fpar_group = zarr.open_group(store=fpar_zarr_store, mode="r")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.min_max_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        min_max_zarr_store = ZarrObjectStore(object_store)
        min_max_group = zarr.open_group(store=min_max_zarr_store, mode="r")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.vci_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )

        vci_init_task = context.submit_subtask(
            InitializeVciArray(
                vci_zarr_path=self.vci_zarr_path,
                fpar_zarr_path=self.fpar_zarr_path,
            )
        )

        context.submit_subtask(
            ComputeVciSlice(
                vci_zarr_path=self.vci_zarr_path,
                fpar_zarr_path=self.fpar_zarr_path,
                min_max_zarr_path=self.min_max_zarr_path,
            ),
            depends_on=[vci_init_task],
        )

        logger.info("Successfully submitted VCI computation tasks.")


class InitializeVciArray(Task):
    """
    Creates the `vci` array in the Zarr store and kicks off the VCI calculation.
    This task is chained to run after the min/max calculation is complete.
    """

    vci_zarr_path: str
    fpar_zarr_path: str
    # job_id: str | None = None
    # shape: tuple[int, int, int] | None = None
    # chunks: tuple[int, int, int] | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing VCI array...")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.fpar_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        fpar_group = zarr.open_group(store=zarr_store, mode="r")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.vci_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)

        zarr.create_array(
            store=zarr_store,
            name="vci",
            shape=fpar_group["fpar"].shape,
            chunks=fpar_group["fpar"].chunks,
            dtype=np.float32,  # VCI is a float
            compressors=COMPRESSOR,
            fill_value=np.nan,
            dimension_names=["time", "y", "x"],
        )

        # zarr.create_array(
        #     store=zarr_store,
        #     name="year",
        #     shape=(fpar_group["fpar"].shape[0],),
        #     chunks=(1,),
        #     dtype=np.int32,  # VCI is a float
        #     compressors=COMPRESSOR,
        #     fill_value=9999,
        #     dimension_names=["time"],
        # )
        # zarr.create_array(
        #     store=zarr_store,
        #     name="dekad",
        #     shape=(fpar_group["fpar"].shape[0],),
        #     chunks=(1,),
        #     dtype=np.int32,  # VCI is a float
        #     compressors=COMPRESSOR,
        #     fill_value=9999,
        #     dimension_names=["time"],
        # )

        logger.info("Successfully initialized VCI array.")


class ComputeVciSlice(Task):
    """
    Computes the VCI for each time slice and stores it in the Zarr store.
    """

    vci_zarr_path: str
    fpar_zarr_path: str
    min_max_zarr_path: str
    slice: tuple[int, int] | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Computing VCI...")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.fpar_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        fpar_zarr_store = ZarrObjectStore(object_store)
        fpar_group = zarr.open_group(store=fpar_zarr_store, mode="r")
        dekad_array = fpar_group["dekad"]

        slice = self.slice
        if slice is None:  # do the whole thing, fetch number of timestamps from zarr
            slice = (0, dekad_array.shape[0])

        n_timestamps = slice[1] - slice[0]
        context.current_task.display = f"ComputeVciSlice[{slice[0]}:{slice[1]}]"
        if n_timestamps > 4:
            middle = slice[0] + n_timestamps // 2
            context.submit_subtask(
                ComputeVciSlice(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    slice=(slice[0], middle),
                )
            )
            context.submit_subtask(
                ComputeVciSlice(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    slice=(middle, slice[1]),
                )
            )
            return

        for time_idx in range(slice[0], slice[1]):
            context.submit_subtask(
                CalculateVciDekad(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    time_index=time_idx,
                )
            )


@lru_cache
def open_zarr_store(path: str, mode: Literal["r", "r+", "w"] = "r") -> ZarrObjectStore:
    object_store = GCSStore(
        bucket=GCS_BUCKET,
        prefix=path,
        credential_provider=GoogleCredentialProvider(),
    )
    return ZarrObjectStore(object_store)


class CalculateVciDekad(Task):
    """
    Orchestrates the VCI calculation for a single time slice (dekad) by submitting
    spatial chunk tasks for parallel processing.
    """

    vci_zarr_path: str
    fpar_zarr_path: str
    min_max_zarr_path: str
    time_index: int
    spatial_chunk: SpatialChunk | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(
            f"Orchestrating VCI calculation for time index {self.time_index}..."
        )

        if self.spatial_chunk is None:
            spatial_chunk = SpatialChunk(0, HEIGHT, 0, WIDTH)
            logger.info(f"Created root chunk {spatial_chunk}...")
        else:
            logger.info(f"Processing chunk {self.spatial_chunk}...")
            spatial_chunk = SpatialChunk(
                self.spatial_chunk["y_start"],
                self.spatial_chunk["y_end"],
                self.spatial_chunk["x_start"],
                self.spatial_chunk["x_end"],
            )

        sub_chunks = spatial_chunk.immediate_sub_chunks(HEIGHT_CHUNK, WIDTH_CHUNK)
        if len(sub_chunks) > 1:
            for chunk in sub_chunks:
                context.submit_subtask(
                    CalculateVciDekad(
                        vci_zarr_path=self.vci_zarr_path,
                        fpar_zarr_path=self.fpar_zarr_path,
                        min_max_zarr_path=self.min_max_zarr_path,
                        time_index=self.time_index,
                        spatial_chunk=chunk,
                    )
                )
            return

        chunk = sub_chunks[0]  # one chunk, process it
        logger.info(f"Opening FPAR Zarr store...")
        fpar_zarr_store = open_zarr_store(self.fpar_zarr_path)
        fpar_group = zarr.open_group(store=fpar_zarr_store, mode="r")
        dekad_array = fpar_group["dekad"]

        logger.info(f"Opening MinMax Zarr store...")
        min_max_zarr_store = open_zarr_store(self.min_max_zarr_path)

        logger.info(f"Opening VCI Zarr store...")
        vci_zarr_store = open_zarr_store(self.vci_zarr_path, mode="a")
        vci_group = zarr.open_group(store=vci_zarr_store, mode="a")
        vci_array = vci_group["vci"]

        logger.info(
            f"Calculating VCI for time index {self.time_index} and chunk {chunk}..."
        )
        dekad = dekad_array[self.time_index].item()

        min_fpar = xr.open_zarr(min_max_zarr_store, consolidated=False)["min_fpar"][
            dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        ]
        # min_fpar = min_max_group["min_fpar"][
        #     dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        # ]
        # max_fpar = min_max_group["max_fpar"][
        #     dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        # ]
        max_fpar = xr.open_zarr(min_max_zarr_store, consolidated=False)["max_fpar"][
            dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        ]
        # fpar = fpar_group["fpar"][
        #     self.time_index, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        # ]
        fpar = xr.open_zarr(fpar_zarr_store, consolidated=False)["fpar"][
            self.time_index, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        ]

        fpar_range = max_fpar - min_fpar
        vci = ((fpar - min_fpar) / fpar_range).where(fpar_range > 0)


        vci_array[
            self.time_index, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        ] = vci.values  # type: ignore[index]
        logger.info(
            f"Successfully calculated and wrote VCI for time index {self.time_index} and chunk {chunk}."
        )
