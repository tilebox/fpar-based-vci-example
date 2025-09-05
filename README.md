<h1 align="center">
  <img src="https://storage.googleapis.com/tbx-web-assets-2bad228/banners/tilebox-banner.svg" alt="Tilebox Logo">
  <br>
</h1>

# Generate VCI and FPAR visualizations using Tilebox

This repository contains a workflow for calculating the Vegetation Condition Index (VCI) from FPAR data. The workflow is built using [Tilebox Workflows](https://docs.tilebox.com/workflows/) and can be run on one or more local machines or on
a cloud cluster.

<p align="center">
  <img src="VCI.png"></a>
</p>

## Workflow

The workflow is implemented in `python` and is located in the [workflow](workflow/) directory.

## IaC for a auto-scaling Tilebox cluster utilizing GCP Spot instances.

The repository also contains a [Pulumi](https://www.pulumi.com/) project for provisioning the necessary GCP resources to run the workflow at scale using cheap GCP Spot instances. The Pulumi Infrastructure as Code (IaC) project is located in the [infrastructure](infrastructure/) directory.
