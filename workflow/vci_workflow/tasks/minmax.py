"""
minmax contains Tilebox tasks for calculating min/max FPAR values per dekad and overall for a dataset

It requires fpar data to be available in Zarr format (see fpar_to_zarr.py) and also outputs min/max data as Zarr arrays.
"""

import pickle
from collections import defaultdict
from typing import cast

import numpy as np
import zarr
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger

from vci_workflow.chunks import SpatialChunk
from vci_workflow.zarr import (
    FILL_VALUE,
    HEIGHT,
    HEIGHT_CHUNK,
    NUM_DEKADS,
    WIDTH,
    WIDTH_CHUNK,
    open_zarr_group,
    open_zarr_store,
)

logger = get_logger()


class ComputeMinMaxPerDekad(Task):
    """Computes min and max FPAR values for each dekad."""

    fpar_zarr_path: str
    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        fpar_group = open_zarr_group(self.fpar_zarr_path, "r")
        dekads = cast(np.ndarray, fpar_group["dekad"][:])

        # create a mapping of every dekad to the indices it appears at
        dekads_to_indices = defaultdict(list)
        for index, dekad in enumerate(dekads):
            dekads_to_indices[int(dekad)].append(index)

        dekads_to_indices = dict(dekads_to_indices)
        context.job_cache["dekad_indices"] = pickle.dumps(dekads_to_indices)

        _initialize_min_max_arrays(self.min_max_zarr_path)
        logger.info(f"Successfully initialized min/max Zarr arrays at: {self.min_max_zarr_path}")

        context.submit_subtask(
            ComputeMinMaxForChunk(
                fpar_zarr_path=self.fpar_zarr_path,
                min_max_zarr_path=self.min_max_zarr_path,
                spatial_chunk=SpatialChunk(0, HEIGHT, 0, WIDTH),
            )
        )


def _initialize_min_max_arrays(zarr_path: str) -> None:
    store = open_zarr_store(zarr_path)
    for name in ["min_fpar_dekad", "max_fpar_dekad"]:
        zarr.create_array(
            store=store,
            name=name,
            shape=(NUM_DEKADS, HEIGHT, WIDTH),
            chunks=(1, HEIGHT_CHUNK, WIDTH_CHUNK),
            dtype=np.uint8,
            overwrite=True,
            fill_value=FILL_VALUE,
            dimension_names=["dekad", "y", "x"],
        )


class ComputeMinMaxForChunk(Task):
    """Computes min/max per dekad and overall, with spatial chunking for large datasets."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    spatial_chunk: SpatialChunk

    def execute(self, context: ExecutionContext) -> None:
        spatial_chunk = self.spatial_chunk

        sub_chunks = spatial_chunk.immediate_sub_chunks(HEIGHT_CHUNK, WIDTH_CHUNK)
        if len(sub_chunks) > 1:
            for chunk in sub_chunks:
                context.submit_subtask(
                    ComputeMinMaxForChunk(
                        fpar_zarr_path=self.fpar_zarr_path,
                        min_max_zarr_path=self.min_max_zarr_path,
                        spatial_chunk=chunk,
                    )
                )
            return

        chunk = sub_chunks[0]
        logger.info(f"Calculating min/max for chunk {chunk}...")
        dekad_to_indices: dict[int, list[int]] = pickle.loads(context.job_cache["dekad_indices"])  # noqa: S301
        context.submit_subtasks(
            [
                ComputeMinMaxForDekad(self.fpar_zarr_path, self.min_max_zarr_path, dekad, chunk)
                for dekad in dekad_to_indices
            ]
        )


class ComputeMinMaxForDekad(Task):
    """Computes min/max for a dekad and a single spatial chunk."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    dekad: int
    spatial_chunk: SpatialChunk

    def execute(self, context: ExecutionContext) -> None:
        if len(self.spatial_chunk.immediate_sub_chunks(HEIGHT_CHUNK, WIDTH_CHUNK)) > 1:
            raise ValueError("Compute MinMaxForDekad should be called for a single leaf chunk.")

        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001
        chunk = self.spatial_chunk
        context.current_task.display = f"MinMaxForDekad({self.dekad}, {chunk})"

        fpar_group = open_zarr_group(self.fpar_zarr_path, mode="r")
        fpar_array: np.ndarray = fpar_group["fpar"]

        dekad_to_indices: dict[int, list[int]] = pickle.loads(context.job_cache["dekad_indices"])  # noqa: S301
        relevant_indices = dekad_to_indices[self.dekad]
        logger.info(
            f"Calculating min/max for dekad {self.dekad} for chunk {chunk}. "
            f"{len(relevant_indices)} relevant indices: {relevant_indices}"
        )

        min_max_group = open_zarr_group(self.min_max_zarr_path, mode="a")
        min_fpar_array = min_max_group["min_fpar_dekad"]
        max_fpar_array = min_max_group["max_fpar_dekad"]

        chunk_shape = (chunk.y_end - chunk.y_start, chunk.x_end - chunk.x_start)
        min_fpar = np.full(chunk_shape, 255, dtype=np.uint8)
        max_fpar = np.full(chunk_shape, 0, dtype=np.uint8)
        has_valid_data = np.full(chunk_shape, False, dtype=bool)

        with tracer.start_span("read_fpar"):
            for idx in relevant_indices:
                fpar_slice = fpar_array[idx, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end]
                valid_mask = fpar_slice != 255
                has_valid_data |= valid_mask

                min_fpar = np.where(valid_mask & (fpar_slice < min_fpar), fpar_slice, min_fpar)
                max_fpar = np.where(valid_mask & (fpar_slice > max_fpar), fpar_slice, max_fpar)

                del fpar_slice, valid_mask

        # Set fill value for pixels with no valid data
        min_fpar = np.where(has_valid_data, min_fpar, FILL_VALUE)
        max_fpar = np.where(has_valid_data, max_fpar, FILL_VALUE)

        min_fpar_array[self.dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end] = min_fpar  # type: ignore[index]
        max_fpar_array[self.dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end] = max_fpar  # type: ignore[index]
        logger.info(f"Successfully computed min/max for dekad {self.dekad} for chunk {chunk}.")
