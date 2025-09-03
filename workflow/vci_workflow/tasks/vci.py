from typing import cast

import numpy as np
import zarr
from tilebox.workflows import ExecutionContext, Task  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import get_logger  # type: ignore[import-untyped]

from vci_workflow.chunks import SpatialChunk
from vci_workflow.memory_logger import MemoryLogger
from vci_workflow.zarr import (
    COMPRESSOR,
    FILL_VALUE,
    HEIGHT,
    HEIGHT_CHUNK,
    WIDTH,
    WIDTH_CHUNK,
    open_zarr_group,
    open_zarr_store,
)

logger = get_logger()
memory_logger = MemoryLogger()


class ComputeVCI(Task):
    """Computes the VCI for each time slice and stores it in the Zarr store."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    vci_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        n_time = _initialize_vci_arrays(self.vci_zarr_path, self.fpar_zarr_path)
        logger.info(f"Successfully initialized VCI Zarr arrays at: {self.vci_zarr_path} for {n_time} dekads")

        full_time_range = (0, n_time)

        context.submit_subtask(
            ComputeVCIRecursively(self.fpar_zarr_path, self.min_max_zarr_path, self.vci_zarr_path, full_time_range)
        )

        logger.info("Successfully submitted VCI computation tasks.")


def _initialize_vci_arrays(vci_zarr_path: str, fpar_zarr_path: str) -> int:
    logger.info(f"Initializing VCI arrays..., vci path: {vci_zarr_path}, fpar path: {fpar_zarr_path}")
    vci_store = open_zarr_store(vci_zarr_path)

    fpar_group = open_zarr_group(fpar_zarr_path, mode="r")
    fpar_array = cast(zarr.Array, fpar_group["fpar"])

    zarr.create_array(
        store=vci_store,
        name="vci",
        shape=fpar_array.shape,
        chunks=fpar_array.chunks,
        dtype=np.uint8,
        compressors=COMPRESSOR,
        fill_value=FILL_VALUE,
        dimension_names=["time", "y", "x"],
    )
    return fpar_array.shape[0]


class ComputeVCIRecursively(Task):
    """Computes the VCI for each time slice and stores it in the Zarr store."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    vci_zarr_path: str
    time_index_range: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        start = self.time_index_range[0]
        end = self.time_index_range[1]
        n_time = self.time_index_range[1] - self.time_index_range[0]
        context.current_task.display = f"ComputeVCI[{start}:{end}, :, :]"

        if n_time > 4:
            middle = start + n_time // 2
            context.submit_subtasks(
                [
                    ComputeVCIRecursively(self.fpar_zarr_path, self.min_max_zarr_path, self.vci_zarr_path, half_range)
                    for half_range in [(start, middle), (middle, end)]
                ]
            )
            return

        # less than 4 indices, submit leaf tasks for individual indices
        for time_index in range(start, end):
            context.submit_subtask(
                ComputeVCIForDekad(
                    self.fpar_zarr_path,
                    self.min_max_zarr_path,
                    self.vci_zarr_path,
                    time_index,
                    SpatialChunk(0, HEIGHT, 0, WIDTH),
                )
            )


class ComputeVCIForDekad(Task):
    """Orchestrates the VCI calculation for a single time slice (dekad) by submitting
    spatial chunk tasks for parallel processing."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    vci_zarr_path: str
    time_index: int
    spatial_chunk: SpatialChunk

    def execute(self, context: ExecutionContext) -> None:
        sub_chunks = self.spatial_chunk.immediate_sub_chunks(HEIGHT_CHUNK, WIDTH_CHUNK)
        if len(sub_chunks) > 1:
            context.submit_subtasks(
                [
                    ComputeVCIForDekad(
                        self.fpar_zarr_path, self.min_max_zarr_path, self.vci_zarr_path, self.time_index, chunk
                    )
                    for chunk in sub_chunks
                ]
            )
            return

        chunk = sub_chunks[0]
        context.current_task.display = f"ComputeVCIForDekad[{self.time_index}, {chunk}]"
        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001

        with tracer.start_span("load_fpar"):
            # Read the FPAR array for the time index
            fpar_group = open_zarr_group(self.fpar_zarr_path, mode="r")
            fpar_array = cast(zarr.Array, fpar_group["fpar"])
            fpar = fpar_array[self.time_index, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end]

            # Find the dekad for the given time index
            dekad = int(fpar_group["dekad"][self.time_index])  # type: ignore[arg-type]

        logger.info(f"Computing VCI for time index {self.time_index} (dekad={dekad}) and chunk {chunk}...")

        with tracer.start_span("load_min_max"):
            # Read min/max arrays for the dekad
            min_max_group = open_zarr_group(self.min_max_zarr_path, mode="r")
            min_array = cast(zarr.Array, min_max_group["min_fpar_dekad"])
            max_array = cast(zarr.Array, min_max_group["max_fpar_dekad"])
            min_fpar = min_array[dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end]
            max_fpar = max_array[dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end]

        # Close stores and delete groups to make sure our memory usage is low
        fpar_group.store.close()
        min_max_group.store.close()
        del fpar_group, min_max_group
        logger.info(f"Successfully loaded fpar, min and max for time index {self.time_index} and chunk {chunk}.")
        memory_logger.log_snapshot()

        with tracer.start_span("compute_vci"):
            # Compute VCI with proper handling of fill values
            fpar = np.ma.masked_array(fpar, mask=(fpar == FILL_VALUE))
            min_fpar = np.ma.masked_array(min_fpar, mask=(min_fpar == FILL_VALUE))
            max_fpar = np.ma.masked_array(max_fpar, mask=(max_fpar == FILL_VALUE))

            # cast as float32, otherwise numpy automatically casts to float64, but 32bit precision is enough
            fpar = fpar.astype(np.float32)
            vci = (fpar - min_fpar) / (max_fpar - min_fpar) * 100
            vci = vci.astype(np.uint8).filled(FILL_VALUE)

        with tracer.start_span("write_vci_to_zarr"):
            # Open VCI output zarr group in write mode
            vci_group = open_zarr_group(self.vci_zarr_path, mode="a")
            vci_array = cast(zarr.Array, vci_group["vci"])

            # Write result
            vci_array[self.time_index, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end] = vci
            vci_group.store.close()
            del vci_group

        logger.info(f"Successfully computed VCI for time index {self.time_index} and chunk {chunk}.")
        memory_logger.log_snapshot()
