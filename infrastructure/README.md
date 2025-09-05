# IaC for a auto-scaling Tilebox cluster utilizing GCP Spot instances.

This [Pulumi](https://www.pulumi.com/) project provisions the following resources:

- A Google Cloud Storage bucket to store the FPAR Zarr datacube
- A Google Artifact Registry repository to store the runner Docker image
- A Google Cloud Build command to build the runner Docker image on code changes
- A Google Compute Engine instance template for the runner instances
- A Google Compute Engine managed instance group (MIG) to run the runner instances
- An autoscaler to scale the MIG based on CPU utilization

## Setup

```bash
uv sync
pulumi login
pulumi up
```
