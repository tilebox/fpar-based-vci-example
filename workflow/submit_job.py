import argparse
from datetime import datetime, timedelta

from main import InitializeZarrStore
from tilebox.workflows import Client as WorkflowsClient


def submit_vci_workflow(start_date: str, num_dekads: int):
    """
    Submits the VCI workflow for a given time range.

    Args:
        start_date: The start date in YYYY-MM-DD format.
        num_dekads: The number of 10-day periods to process.
    """
    start_time = datetime.fromisoformat(start_date)
    end_time = start_time + timedelta(days=num_dekads * 10)

    time_range = f"{start_time.isoformat()}/{end_time.isoformat()}"

    print(f"Submitting job for time range: {time_range}")

    client = WorkflowsClient()
    job = client.jobs().submit(
        "vci-datacube-creation",
        InitializeZarrStore(time_range=time_range)
    )

    print(f"Successfully submitted job with ID: {job.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit the VCI workflow.")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="The start date for the workflow in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--num-dekads",
        type=int,
        required=True,
        help="The number of 10-day periods (dekads) to process.",
    )
    args = parser.parse_args()

    submit_vci_workflow(args.start_date, args.num_dekads)
