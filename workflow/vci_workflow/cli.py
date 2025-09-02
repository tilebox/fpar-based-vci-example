#!/usr/bin/env python3

"""
Modular VCI workflow definitions following Tilebox best practices.
Each workflow can be run independently or as part of a larger pipeline.
"""

import argparse
import sys
from datetime import datetime

from tilebox.workflows import Client as WorkflowsClient
from tilebox.workflows import ExecutionContext, Task

from vci_workflow.ingest import WriteFparToZarr
from vci_workflow.minmax import CalculateMinMaxPerDekad
from vci_workflow.vci import ComputeVci
from vci_workflow.vci_visualization import CreateVciMp4
from vci_workflow.zarr import ZARR_STORE_PATH


class FparIngestionWorkflow(Task):
    """Standalone workflow for FPAR data ingestion."""

    time_range: str
    fpar_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        context.submit_subtask(WriteFparToZarr(zarr_path=self.fpar_zarr_path, time_range=self.time_range))


class MinMaxWorkflow(Task):
    """Standalone workflow for min/max calculation."""

    fpar_zarr_path: str
    min_max_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        context.submit_subtask(
            CalculateMinMaxPerDekad(
                fpar_zarr_path=self.fpar_zarr_path,
                min_max_zarr_path=self.min_max_zarr_path,
            )
        )


class VciCalculationWorkflow(Task):
    """Standalone workflow for VCI calculation."""

    fpar_zarr_path: str
    min_max_zarr_path: str
    vci_zarr_path: str

    def execute(self, context: ExecutionContext) -> None:
        context.submit_subtask(
            ComputeVci(
                fpar_zarr_path=self.fpar_zarr_path,
                min_max_zarr_path=self.min_max_zarr_path,
                vci_zarr_path=self.vci_zarr_path,
            )
        )


class VciVideoWorkflow(Task):
    """Standalone workflow for VCI video generation."""

    vci_zarr_path: str
    fpar_zarr_path: str
    downsample_factor: int = 20
    output_cluster: str | None = None

    def execute(self, context: ExecutionContext) -> None:
        context.submit_subtask(
            CreateVciMp4(
                vci_zarr_path=self.vci_zarr_path,
                fpar_zarr_path=self.fpar_zarr_path,
                downsample_factor=self.downsample_factor,
                output_cluster=self.output_cluster,
            )
        )


class EndToEndVciWorkflow(Task):
    """
    Umbrella orchestrator that runs all VCI workflow steps in sequence.
    Each step can also be run independently using the individual workflow classes.
    """

    time_range: str
    video_output_cluster: str | None = None
    video_downsample_factor: int = 20
    fpar_zarr_path: str | None = None
    min_max_zarr_path: str | None = None
    vci_zarr_path: str | None = None
    ingest_fpar: bool = True
    calculate_minmax: bool = True
    calculate_vci: bool = True
    create_video: bool = True

    def execute(self, context: ExecutionContext) -> None:
        job_id = context.current_task.job.id

        # Generate deterministic paths
        fpar_path = self.fpar_zarr_path or f"{ZARR_STORE_PATH}/{job_id}/cube.zarr"
        minmax_path = self.min_max_zarr_path or f"{ZARR_STORE_PATH}/{job_id}/minmax.zarr"
        vci_path = self.vci_zarr_path or f"{ZARR_STORE_PATH}/{job_id}/vci.zarr"

        # Sequential workflow execution with dependency management
        tasks = []

        if self.ingest_fpar:
            ingest_task = context.submit_subtask(
                FparIngestionWorkflow(time_range=self.time_range, fpar_zarr_path=fpar_path)
            )
            tasks.append(ingest_task)

        if self.calculate_minmax:
            minmax_task = context.submit_subtask(
                MinMaxWorkflow(fpar_zarr_path=fpar_path, min_max_zarr_path=minmax_path),
                depends_on=tasks if self.ingest_fpar else [],
            )
            tasks.append(minmax_task)

        if self.calculate_vci:
            vci_task = context.submit_subtask(
                VciCalculationWorkflow(
                    fpar_zarr_path=fpar_path,
                    min_max_zarr_path=minmax_path,
                    vci_zarr_path=vci_path,
                ),
                depends_on=tasks if self.calculate_minmax else [],
            )
            tasks.append(vci_task)

        if self.create_video:
            context.submit_subtask(
                VciVideoWorkflow(
                    vci_zarr_path=vci_path,
                    fpar_zarr_path=fpar_path,
                    downsample_factor=self.video_downsample_factor,
                    output_cluster=self.video_output_cluster,
                ),
                depends_on=tasks if self.calculate_vci else [],
            )


def main():
    """Command line interface for submitting modular VCI workflow jobs."""
    parser = argparse.ArgumentParser(
        description="Submit VCI workflow jobs (individual steps or end-to-end)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full end-to-end workflow
  %(prog)s end-to-end --time-range "2022-01-01/2022-12-31"
  
  # Individual steps (manual chaining)
  %(prog)s ingest --time-range "2022-01-01/2022-12-31" --fpar-zarr-path "path/to/fpar.zarr"
  %(prog)s minmax --fpar-zarr-path "path/to/fpar.zarr" --min-max-zarr-path "path/to/minmax.zarr"
  %(prog)s vci --fpar-zarr-path "path/to/fpar.zarr" --min-max-zarr-path "path/to/minmax.zarr" --vci-zarr-path "path/to/vci.zarr"
  %(prog)s video --vci-zarr-path "path/to/vci.zarr" --fpar-zarr-path "path/to/fpar.zarr"
        """,
    )

    subparsers = parser.add_subparsers(dest="workflow", help="Workflow to run")

    # End-to-end workflow
    end_to_end = subparsers.add_parser("end-to-end", help="Run complete VCI pipeline")
    end_to_end.add_argument("--time-range", required=True, help="Time range (ISO format)")
    end_to_end.add_argument("--video-downsample-factor", type=int, default=20)
    end_to_end.add_argument("--video-output-cluster", type=str)
    end_to_end.add_argument("--no-ingest-fpar", action="store_true")
    end_to_end.add_argument("--no-calculate-minmax", action="store_true")
    end_to_end.add_argument("--no-calculate-vci", action="store_true")
    end_to_end.add_argument("--no-create-video", action="store_true")

    # Individual workflows
    ingest = subparsers.add_parser("ingest", help="FPAR data ingestion only")
    ingest.add_argument("--time-range", required=True, help="Time range (ISO format)")
    ingest.add_argument("--fpar-zarr-path", required=True, help="Output FPAR zarr path")

    minmax = subparsers.add_parser("minmax", help="Min/max calculation only")
    minmax.add_argument("--fpar-zarr-path", required=True, help="Input FPAR zarr path")
    minmax.add_argument("--min-max-zarr-path", required=True, help="Output min/max zarr path")

    vci = subparsers.add_parser("vci", help="VCI calculation only")
    vci.add_argument("--fpar-zarr-path", required=True, help="Input FPAR zarr path")
    vci.add_argument("--min-max-zarr-path", required=True, help="Input min/max zarr path")
    vci.add_argument("--vci-zarr-path", required=True, help="Output VCI zarr path")

    video = subparsers.add_parser("video", help="Video generation only")
    video.add_argument("--vci-zarr-path", required=True, help="Input VCI zarr path")
    video.add_argument("--fpar-zarr-path", required=True, help="Input FPAR zarr path")
    video.add_argument("--downsample-factor", type=int, default=20)
    video.add_argument("--output-cluster", type=str)

    # Common arguments
    for subparser in [end_to_end, ingest, minmax, vci, video]:
        subparser.add_argument("--job-name", default="")
        subparser.add_argument("--cluster", help="Execution cluster")

    args = parser.parse_args()

    if not args.workflow:
        parser.print_help()
        sys.exit(1)

    # Create appropriate workflow task
    if args.workflow == "end-to-end":
        task = EndToEndVciWorkflow(
            time_range=args.time_range,
            video_output_cluster=args.video_output_cluster,
            video_downsample_factor=args.video_downsample_factor,
            ingest_fpar=not args.no_ingest_fpar,
            calculate_minmax=not args.no_calculate_minmax,
            calculate_vci=not args.no_calculate_vci,
            create_video=not args.no_create_video,
        )
    elif args.workflow == "ingest":
        task = FparIngestionWorkflow(time_range=args.time_range, fpar_zarr_path=args.fpar_zarr_path)
    elif args.workflow == "minmax":
        task = MinMaxWorkflow(
            fpar_zarr_path=args.fpar_zarr_path,
            min_max_zarr_path=args.min_max_zarr_path,
        )
    elif args.workflow == "vci":
        task = VciCalculationWorkflow(
            fpar_zarr_path=args.fpar_zarr_path,
            min_max_zarr_path=args.min_max_zarr_path,
            vci_zarr_path=args.vci_zarr_path,
        )
    elif args.workflow == "video":
        task = VciVideoWorkflow(
            vci_zarr_path=args.vci_zarr_path,
            fpar_zarr_path=args.fpar_zarr_path,
            downsample_factor=args.downsample_factor,
            output_cluster=args.output_cluster,
        )
    else:
        parser.print_help()
        sys.exit(1)

    # Generate descriptive job name with timestamp and key parameters
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")

    job_name = args.job_name
    if not job_name:
        if args.workflow in {"end-to-end", "ingest"}:
            years = args.time_range.split("/")[0][:4] + "-" + args.time_range.split("/")[1][:4]
            job_name = f"vci-{args.workflow}-{timestamp}--{years}"
        else:
            job_name = f"vci-{args.workflow}-{timestamp}"

    client = WorkflowsClient().jobs()
    job = client.submit(job_name, task, cluster=args.cluster, max_retries=3)

    print(f"Successfully submitted {args.workflow} job: {job.id}")  # noqa: T201
    print(f"Monitor at: https://console.tilebox.com/workflows/jobs/{job.id}")  # noqa: T201


if __name__ == "__main__":
    main()
