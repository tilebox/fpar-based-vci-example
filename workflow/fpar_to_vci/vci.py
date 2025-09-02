import gc
from typing import cast

import numpy as np
import zarr

from fpar_to_vci.chunks import SpatialChunk
from fpar_to_vci.config import COMPRESSOR, HEIGHT, HEIGHT_CHUNK, WIDTH, WIDTH_CHUNK
from fpar_to_vci.zarr_helpers import open_zarr_group
from tilebox.workflows import ExecutionContext, Task  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import get_logger  # type: ignore[import-untyped]


class ComputeVci(Task):
    """Computes the VCI for each time slice and stores it in the Zarr store."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    vci_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Computing VCI...")

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
    """Creates the `vci` array in the Zarr store."""

    vci_zarr_path: str
    fpar_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Initializing VCI array...")

        fpar_group = open_zarr_group(self.fpar_zarr_path, mode="r")
        vci_group = open_zarr_group(self.vci_zarr_path, mode="w")

        fpar_array = cast(zarr.Array, fpar_group["fpar"])
        zarr.create_array(
            store=vci_group.store,
            name="vci",
            shape=fpar_array.shape,
            chunks=fpar_array.chunks,
            dtype=np.float32,
            compressors=COMPRESSOR,
            fill_value=np.nan,
            dimension_names=["time", "y", "x"],
        )

        logger.info("Successfully initialized VCI array.")


class ComputeVciSlice(Task):
    """Computes the VCI for each time slice and stores it in the Zarr store."""

    vci_zarr_path: str
    fpar_zarr_path: str
    min_max_zarr_path: str
    slice: tuple[int, int] | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Computing VCI...")

        fpar_group = open_zarr_group(self.fpar_zarr_path, mode="r")
        dekad_array = cast(zarr.Array, fpar_group["dekad"])

        slice_range = self.slice
        if slice_range is None:
            dekad_arr = cast(np.ndarray, dekad_array[:])
            slice_range = (0, len(dekad_arr))

        n_timestamps = slice_range[1] - slice_range[0]

        if n_timestamps > 4:
            middle = slice_range[0] + n_timestamps // 2
            context.submit_subtask(
                ComputeVciSlice(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    slice=(slice_range[0], middle),
                )
            )
            context.submit_subtask(
                ComputeVciSlice(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    slice=(middle, slice_range[1]),
                )
            )
            return

        for time_idx in range(slice_range[0], slice_range[1]):
            context.submit_subtask(
                CalculateVciDekad(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    min_max_zarr_path=self.min_max_zarr_path,
                    time_index=time_idx,
                )
            )


class CalculateVciDekad(Task):
    """Orchestrates the VCI calculation for a single time slice (dekad) by submitting
    spatial chunk tasks for parallel processing."""

    vci_zarr_path: str
    fpar_zarr_path: str
    min_max_zarr_path: str
    time_index: int
    spatial_chunk: SpatialChunk | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Orchestrating VCI calculation for time index {self.time_index}...")

        if self.spatial_chunk is None:
            spatial_chunk = SpatialChunk(0, HEIGHT, 0, WIDTH)
            logger.info(f"Created root chunk {spatial_chunk}...")
        else:
            logger.info(f"Processing chunk {self.spatial_chunk}...")
            spatial_chunk = self.spatial_chunk

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

        chunk = sub_chunks[0]

        logger.info(f"Calculating VCI for time index {self.time_index} and chunk {chunk}...")

        # Open all zarr groups
        fpar_group = open_zarr_group(self.fpar_zarr_path, mode="r")
        min_max_group = open_zarr_group(self.min_max_zarr_path, mode="r")
        vci_group = open_zarr_group(self.vci_zarr_path, mode="a")

        # Get arrays
        fpar_array = cast(zarr.Array, fpar_group["fpar"])
        dekad_array = cast(zarr.Array, fpar_group["dekad"])
        min_fpar_array = cast(zarr.Array, min_max_group["min_fpar"])
        max_fpar_array = cast(zarr.Array, min_max_group["max_fpar"])
        vci_array = cast(zarr.Array, vci_group["vci"])

        dekad_value = cast(np.ndarray, dekad_array[self.time_index])
        dekad = int(dekad_value.item()) if hasattr(dekad_value, "item") else int(dekad_value)

        # Load data chunks
        min_fpar = cast(
            np.ndarray,
            min_fpar_array[dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end],
        )
        max_fpar = cast(
            np.ndarray,
            max_fpar_array[dekad - 1, chunk.y_start : chunk.y_end, chunk.x_start : chunk.x_end],
        )
        fpar = cast(
            np.ndarray,
            fpar_array[
                self.time_index,
                chunk.y_start : chunk.y_end,
                chunk.x_start : chunk.x_end,
            ],
        )

        # Calculate VCI with proper handling of division by zero
        fpar_range = max_fpar.astype(np.float32) - min_fpar.astype(np.float32)
        vci = np.where(
            fpar_range > 0,
            (fpar.astype(np.float32) - min_fpar.astype(np.float32)) / fpar_range,
            np.nan,
        )

        # Write result
        vci_array[
            self.time_index,
            chunk.y_start : chunk.y_end,
            chunk.x_start : chunk.x_end,
        ] = vci

        # Clean up memory
        del min_fpar, max_fpar, fpar, fpar_range, vci
        gc.collect()

        logger.info(f"Successfully calculated VCI for time index {self.time_index} and chunk {chunk}.")
