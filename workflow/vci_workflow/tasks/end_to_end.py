from tilebox.workflows import ExecutionContext, Task

from vci_workflow.tasks.fpar_to_zarr import WriteFparToZarr
from vci_workflow.tasks.minmax import ComputeMinMaxPerDekad
from vci_workflow.tasks.vci import ComputeVCI
from vci_workflow.tasks.video import ZarrArrayToVideo


class EndToEndFparToVideos(Task):
    """
    Umbrella task for combining the individual workflow steps of generating FPAR/VCI videos into a single end-to-end
    workflow.
    """

    time_range: str

    def execute(self, context: ExecutionContext) -> None:
        job_id = context.current_task.job.id

        zarr_store = f"fpar_vci_workflow/{job_id}/store.zarr"

        # we can write all our zarr arrays into the same store:
        fpar_store = zarr_store
        minmax_store = zarr_store
        vci_store = zarr_store

        ingest_fpar = context.submit_subtask(WriteFparToZarr(fpar_store, self.time_range))

        compute_minmax = context.submit_subtask(
            ComputeMinMaxPerDekad(fpar_store, minmax_store), depends_on=[ingest_fpar]
        )

        compute_vci = context.submit_subtask(
            ComputeVCI(fpar_store, minmax_store, vci_store), depends_on=[compute_minmax]
        )

        video_downsample_factor = 5
        video_downsize_factor = 4

        # create the FPAR video
        context.submit_subtask(
            ZarrArrayToVideo(
                fpar_store,
                "fpar",
                fpar_store,
                (video_downsample_factor, video_downsize_factor),
                "FPAR (%)",
                "Fraction of absorbed photosynthetically active radiation",
            ),
            depends_on=[ingest_fpar],
        )

        # create the VCI video
        context.submit_subtask(
            ZarrArrayToVideo(
                vci_store,
                "vci",
                fpar_store,
                (video_downsample_factor, video_downsize_factor),
                "VCI",
                "Vegetation Condition Index (Derived from: MODIS/VIIRS FPAR)",
            ),
            depends_on=[compute_vci],
        )
