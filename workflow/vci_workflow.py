from tilebox.workflows import Task, ExecutionContext
from config import ZARR_STORE_PATH
from ingest import WriteFparToZarr
from minmax import ComputeMinMaxPerDekad
from vci import ComputeVci
from vci_visualization import CreateVciMp4


class VciWorkflow(Task):
    """
    A high-level workflow that ties together the entire VCI calculation process.
    """
    time_range: str
    video_output_cluster: str | None = None
    video_downsample_factor: int = 20
    fpar_zarr_path: str = "job_id"

    def execute(self, context: ExecutionContext) -> None:
        # Alternative default zarr path, based on initially loaded fpar data cube
        zarr_path = f"{ZARR_STORE_PATH}/0198771f-8498-abce-2857-2c573373fb37/cube.zarr"
        if self.fpar_zarr_path == "job_id":
            zarr_path = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"

        ingestion_task = context.submit_subtask(
            WriteFparToZarr(zarr_path=zarr_path, time_range=self.time_range)
        )
        min_max_task = context.submit_subtask(
            ComputeMinMaxPerDekad(
                fpar_zarr_path=zarr_path, min_max_zarr_path=zarr_path
            ),
            depends_on=[ingestion_task],
        )
        vci_task = context.submit_subtask(
            ComputeVci(
                fpar_zarr_path=zarr_path,
                min_max_zarr_path=zarr_path,
                vci_zarr_path=zarr_path,
            ),
            depends_on=[min_max_task],
        )
        context.submit_subtask(
            CreateVciMp4(
                vci_zarr_path=zarr_path,
                fpar_zarr_path=zarr_path,
                downsample_factor=self.video_downsample_factor,
                output_cluster=self.video_output_cluster,
            ),
            depends_on=[vci_task],
        )
