"""Min/max calculation tasks for VCI workflow."""

from dataclasses import field

import numpy as np
import zarr

from fpar_to_vci.chunks import SpatialChunk
from fpar_to_vci.config import FILL_VALUE, HEIGHT, HEIGHT_CHUNK, WIDTH, WIDTH_CHUNK
from fpar_to_vci.zarr_helpers import open_zarr_group
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger


class ComputeMinMaxPerDekad(Task):
    """Computes min and max FPAR values for each dekad."""

    fpar_zarr_path: str
    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        init_task = context.submit_subtask(InitializeMinMaxArrays(min_max_zarr_path=self.min_max_zarr_path))
        context.submit_subtask(
            OrchestrateDekadMinMaxCalculation(
                fpar_zarr_path=self.fpar_zarr_path,
                min_max_zarr_path=self.min_max_zarr_path,
            ),
            depends_on=[init_task],
        )


class InitializeMinMaxArrays(Task):
    """Creates min_fpar and max_fpar arrays in the Zarr store."""

    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing min/max arrays...")

        group = open_zarr_group(self.min_max_zarr_path, mode="w")
        shape = (36, HEIGHT, WIDTH)
        chunks = (36, HEIGHT_CHUNK, WIDTH_CHUNK)

        for name in ["min_fpar", "max_fpar"]:
            zarr.create_array(
                store=group.store,
                name=name,
                shape=shape,
                chunks=chunks,
                dtype=np.uint8,
                overwrite=True,
                fill_value=FILL_VALUE,
                dimension_names=["dekad", "y", "x"],
            )


class OrchestrateDekadMinMaxCalculation(Task):
    """Submits a subtask for each of the 36 dekads."""

    fpar_zarr_path: str
    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()

        fpar_group = open_zarr_group(self.fpar_zarr_path)
        dekad_data = fpar_group["dekad"][:]  # type: ignore
        unique_dekads = np.unique(dekad_data)

        for dekad in unique_dekads:
            context.submit_subtask(
                CalculateMinMaxForDekad(
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    dekad=int(dekad),
                )
            )
        logger.info(f"Submitted {len(unique_dekads)} dekad processing tasks.")


class CalculateMinMaxForDekad(Task):
    """Calculates min/max for a dekad, with spatial chunking for large datasets."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    dekad: int
    spatial_chunk: SpatialChunk = field(default_factory=lambda: SpatialChunk(0, HEIGHT, 0, WIDTH))

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()

        spatial_chunk = self.spatial_chunk

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

        chunk = sub_chunks[0]

        fpar_group = open_zarr_group(self.fpar_zarr_path)
        fpar_array = fpar_group["fpar"]
        dekad_array = fpar_group["dekad"][:]  # type: ignore
        relevant_indices = np.where(dekad_array == self.dekad)[0]

        min_max_group = open_zarr_group(self.min_max_zarr_path, mode="a")
        min_fpar_array = min_max_group["min_fpar"]
        max_fpar_array = min_max_group["max_fpar"]

        chunk_shape = (chunk.y_end - chunk.y_start, chunk.x_end - chunk.x_start)
        min_fpar = np.full(chunk_shape, 255, dtype=np.uint8)
        max_fpar = np.full(chunk_shape, 0, dtype=np.uint8)
        has_valid_data = np.full(chunk_shape, False, dtype=bool)

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
