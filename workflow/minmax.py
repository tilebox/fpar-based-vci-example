from dataclasses import dataclass
import math
from typing import Iterator
import numpy as np
import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger
from zarr.storage import ObjectStore as ZarrObjectStore

from config import (
    FILL_VALUE,
    GCS_BUCKET,
    HEIGHT,
    HEIGHT_CHUNK,
    WIDTH,
    WIDTH_CHUNK,
)


class ComputeMinMaxPerDekad(Task):
    """
    Computes the min and max FPAR values for each dekad and stores them in the Zarr store.
    """

    fpar_zarr_path: str
    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        init_minmax = context.submit_subtask(
            InitializeMinMaxArrays(min_max_zarr_path=self.min_max_zarr_path)
        )
        context.submit_subtask(
            OrchestrateDekadMinMaxCalculation(
                fpar_zarr_path=self.fpar_zarr_path,
                min_max_zarr_path=self.min_max_zarr_path,
            ),
            depends_on=[init_minmax],
        )


class InitializeMinMaxArrays(Task):
    """
    Creates and initializes the min_fpar and max_fpar arrays in the Zarr store.
    This task is chained to run after the main data ingestion is complete.
    """

    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing min/max FPAR arrays...")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.min_max_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)

        shape = (36, HEIGHT, WIDTH)
        chunks = (36, HEIGHT_CHUNK, WIDTH_CHUNK)

        zarr.create_array(
            store=zarr_store,
            name="min_fpar",
            shape=shape,
            chunks=chunks,
            dtype=np.uint8,
            overwrite=True,
            fill_value=FILL_VALUE,
            dimension_names=["dekad", "y", "x"],
        )

        zarr.create_array(
            store=zarr_store,
            name="max_fpar",
            shape=shape,
            chunks=chunks,
            dtype=np.uint8,
            overwrite=True,
            fill_value=FILL_VALUE,
            dimension_names=["dekad", "y", "x"],
        )

        logger.info("Successfully initialized min/max arrays.")


class OrchestrateDekadMinMaxCalculation(Task):
    """
    Submits a subtask for each of the 36 dekads of the year.
    """

    fpar_zarr_path: str
    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating min/max calculation by dekad...")

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.fpar_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        dekad_array = zarr.open_group(zarr_store, mode="r")["dekad"]

        unique_dekads = np.unique(dekad_array[:])
        for i in unique_dekads:
            logger.info(f"Submitting task for dekad {i}...")
            context.submit_subtask(
                CalculateMinMaxForDekad(
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    dekad=int(i),  # numpy uint8 can't be serialized
                )
            )
        logger.info(f"Submitted {len(unique_dekads)} dekad processing tasks.")


def _split_interval(start: int, end: int, max_size: int) -> Iterator[tuple[int, int]]:
    """
    Split an interval into two sub-intervals in case it is larger than the given maximum size.
    The split is done at the closest power of two.

    Example:
        _split_interval(0, 1000, 512) -> (0, 512), (512, 1000)
        _split_interval(512, 1000, 512) -> (512, 1000)
    """
    n = end - start
    if n > max_size:
        split_at = 2 ** math.floor(math.log2(n - 1)) + start
        yield start, split_at
        yield split_at, end
    else:
        yield start, end


@dataclass(frozen=True, order=True)
class SpatialChunk:
    """
    SpatialChunk represents a 2D chunk within a potentially larger 2D space.

    It is useful for sub-dividing larger 2D spaces into smaller chunks for parallel processing.
    """

    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def __str__(self) -> str:
        """String representation of the chunk in slice notation."""
        return f"{self.y_start}:{self.y_end}, {self.x_start}:{self.x_end}"

    def __repr__(self) -> str:
        return (
            f"SpatialChunk({self.y_start}, {self.y_end}, {self.x_start}, {self.x_end})"
        )

    def immediate_sub_chunks(self, y_size: int, x_size: int) -> list["SpatialChunk"]:
        """
        Subdivide a given chunk into at most four sub-chunks, for dividing it for parallel processing.

        If a chunk is already smaller than the given size in both dimensions, it will no longer be subdivided and
        instead returned as it is as single element list.

        By calling this function recursively, a chunk tree is created, where each node is a chunk and the leaves are
        all chunks that are at most (y_size, x_size)

        Returns:
            list[SpatialChunk]: A list of immediate sub-chunks of this chunk by splitting it into at most
                four sub-chunks.
        """
        sub_chunks = []
        for y_start, y_end in _split_interval(self.y_start, self.y_end, y_size):
            for x_start, x_end in _split_interval(self.x_start, self.x_end, x_size):
                sub_chunks.append(SpatialChunk(y_start, y_end, x_start, x_end))
        return sub_chunks


class CalculateMinMaxForDekad(Task):
    """
    Submits a subtask for each spatial chunk for a given dekad.
    """

    fpar_zarr_path: str
    min_max_zarr_path: str
    dekad: int
    spatial_chunk: SpatialChunk | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Orchestrating min/max for dekad {self.dekad}...")

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
                    CalculateMinMaxForDekad(
                        fpar_zarr_path=self.fpar_zarr_path,
                        min_max_zarr_path=self.min_max_zarr_path,
                        dekad=self.dekad,
                        spatial_chunk=chunk,
                    )
                )
            return

        chunk = sub_chunks[0]  # one chunk, process it

        logger.info(f"Opening FPAR Zarr store...")
        fpar_object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.fpar_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        fpar_zarr_store = ZarrObjectStore(fpar_object_store)

        logger.info(f"Opening MinMax Zarr store...")
        min_max_object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=self.min_max_zarr_path,
            credential_provider=GoogleCredentialProvider(),
        )
        min_max_zarr_store = ZarrObjectStore(min_max_object_store)

        logger.info(f"Reading Dekad array...")
        fpar_zarr_group = zarr.open_group(store=fpar_zarr_store, mode="r")
        dekad_array = fpar_zarr_group["dekad"][:]  # type: ignore
        relevant_dekad_indices = np.where(dekad_array == self.dekad)[0]

        logger.info(f"Computing min/max for dekad {self.dekad}...")

        # Initialize min/max arrays with proper fill values
        chunk_shape = (chunk.y_end - chunk.y_start, chunk.x_end - chunk.x_start)
        min_fpar = np.full(
            chunk_shape, 255, dtype=np.uint8
        )  # Start with max possible value
        max_fpar = np.full(
            chunk_shape, 0, dtype=np.uint8
        )  # Start with min possible value

        # Process each time index individually to save memory
        for idx in relevant_dekad_indices:
            fpar_zarr_group = zarr.open_group(store=fpar_zarr_store, mode="r")
            fpar_slice = fpar_zarr_group["fpar"][
                idx,
                chunk.y_start : chunk.y_end,
                chunk.x_start : chunk.x_end,
            ]

            # Only update where we have valid data (not fill value)
            valid_mask = fpar_slice != 255
            min_fpar = np.where(
                valid_mask & (fpar_slice < min_fpar), fpar_slice, min_fpar
            )
            max_fpar = np.where(
                valid_mask & (fpar_slice > max_fpar), fpar_slice, max_fpar
            )
            del fpar_slice
            del fpar_zarr_group


        logger.info(f"Writing min/max for dekad {self.dekad}...")
        min_max_group = zarr.open_group(store=min_max_zarr_store, mode="a")
        min_max_group["min_fpar"][
            self.dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        ] = min_fpar  # type: ignore[index]
        min_max_group["max_fpar"][
            self.dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end
        ] = max_fpar  # type: ignore[index]
        logger.info(f"Successfully wrote min/max for dekad {self.dekad}.")
