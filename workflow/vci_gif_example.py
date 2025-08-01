#!/usr/bin/env python3
"""
Example of how to use the VCI MP4 video visualization workflow with datetime ranges.
"""

from tilebox.workflows import Client as WorkflowsClient
from vci_visualization import CreateVciVideo


def create_vci_video_example():
    """Example of creating a VCI MP4 video with datetime range."""

    # Your VCI datacube job ID
    job_id = "01986054-7c60-3d95-7ca6-e996b68b577b"

    # Example 1: Create MP4 for a specific year
    client = WorkflowsClient()
    job_client = client.jobs()

    # task = CreateVciVideo(
    #     job_id=job_id,
    #     time_range="2020-01-01/2020-12-31",  # All of 2020
    #     downsample_factor=20  # 20x downsampling for good balance
    # )

    # job = job_client.submit("vci-visualization", task)
    # print(f"Submitted VCI MP4 job for 2020: {job.id}")
    # print(f"Monitor at: https://app.tilebox.com/workflows/jobs/{job.id}")

    # Example 2: Create MP4 for a specific season
    task2 = CreateVciVideo(
        job_id=job_id,
        time_range="2021-01-01/2022-01-01",
        downsample_factor=20,
        output_cluster="stefan-dev-cluster-CVscQ2mm81Q1UP"
    )

    job2 = job_client.submit("vci-visualization", task2)
    print(f"Submitted VCI MP4 job for June 2021: {job2.id}")

    # Example 3: Create MP4 for entire dataset (no time range)
    # task3 = CreateVciVideo(
    #     job_id=job_id,
    #     time_range=None,  # Process all available data
    #     downsample_factor=32  # Higher downsampling for faster processing
    # )

    # job3 = job_client.submit("vci-visualization", task3)
    # print(f"Submitted VCI MP4 job for entire dataset: {job3.id}")


def create_vci_video_drought_analysis():
    """Example focused on drought analysis periods."""

    job_id = "01986039-99a7-dc8c-aa11-52d327e17a44"
    client = WorkflowsClient()
    job_client = client.jobs()

    # Analyze specific drought periods
    drought_periods = [
        ("2012-06-01/2012-09-30", "2012 Drought"),
        ("2018-07-01/2018-10-31", "2018 Drought"),
        ("2021-06-01/2021-08-31", "2021 Drought"),
    ]

    jobs = []
    for time_range, description in drought_periods:
        task = CreateVciVideo(
            job_id=job_id,
            time_range=time_range,
            downsample_factor=20
        )

        job = job_client.submit("vci-visualization", task)
        jobs.append((job.id, description))
        print(f"Submitted {description} MP4 job: {job.id}")

    print("\nAll drought analysis jobs submitted!")
    for job_id, description in jobs:
        print(f"{description}: https://app.tilebox.com/workflows/jobs/{job_id}")


if __name__ == "__main__":
    print("VCI MP4 Video Visualization Examples")
    print("=" * 40)

    print("\n1. Basic Examples:")
    create_vci_video_example()

    print("\n2. Drought Analysis Examples:")
    # create_vci_video_drought_analysis()

    print("\nTime Range Format:")
    print("- Use ISO format: 'YYYY-MM-DD/YYYY-MM-DD'")
    print("- Examples:")
    print("  - Full year: '2020-01-01/2020-12-31'")
    print("  - Growing season: '2021-04-01/2021-10-31'")
    print("  - Summer: '2022-06-01/2022-08-31'")
    print("  - Specific months: '2023-07-01/2023-09-30'")

    print("\nDownsample Factor Guidelines:")
    print("- 10x: High detail, slower processing")
    print("- 20x: Good balance (recommended)")
    print("- 32x: Lower detail, faster processing")
    print("- 50x: Quick preview, very fast")
