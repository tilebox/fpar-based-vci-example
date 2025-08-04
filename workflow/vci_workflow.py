from tilebox.workflows import Task, ExecutionContext
from config import ZARR_STORE_PATH
from ingest import WriteFparToZarr
from minmax import ComputeMinMaxPerDekad, InitializeMinMaxArrays
from vci import InitializeVciArray
from vci_visualization import CreateVciVideo


class VciWorkflow(Task):
    """
    A high-level workflow that ties together the entire VCI calculation process.
    """

    time_range: str
    # video_output_cluster: str | None = None
    # video_downsample_factor: int = 20

    def execute(self, context: ExecutionContext) -> None:
        zarr_path = f"{ZARR_STORE_PATH}/{context.current_task.job.id}/cube.zarr"
        ingestion_task = context.submit_subtask(
            WriteFparToZarr(zarr_path=zarr_path, time_range=self.time_range)
        )
        min_max_task = context.submit_subtask(
            ComputeMinMaxPerDekad(fpar_zarr_path=zarr_path, min_max_zarr_path=zarr_path), depends_on=[ingestion_task]
        )
        vci_task = context.submit_subtask(
            InitializeVciArray(), depends_on=[min_max_task]
        )
        # context.submit_subtask(
        #     CreateVciVideo(
        #         job_id=self.fpar_datacube_location_uuid,
        #         time_range=self.time_range,
        #         downsample_factor=self.video_downsample_factor,
        #         output_cluster=self.video_output_cluster,
        #     ),
        #     depends_on=[vci_task],
        # )
