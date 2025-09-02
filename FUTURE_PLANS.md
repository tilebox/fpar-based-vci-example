# Future Infrastructure Improvements Plan

This document outlines potential improvements for the VCI workflow infrastructure that can be implemented at a later date.

## Flexible Spot Instances with Multiple Machine Types

### Goal

Increase the resilience and potentially lower the cost of the Spot-based compute cluster by allowing the Managed Instance Group (MIG) to provision VMs from a list of several different machine types. This makes the cluster less susceptible to stockouts of a single machine type.

### Implementation Plan

1.  **Identify Suitable Machine Types:**
    _ Use the `gcloud` CLI to query for available machine types in the `europe-west4` region that are cost-effective and meet the workload's requirements (e.g., 1-2 vCPUs, ~8GB RAM).
    _ Example command:
    `bash
gcloud compute machine-types list \
  --filter="zone:( europe-west4-a ) AND guestCpus <= 2 AND memoryMb >= 7168 AND memoryMb <= 9216" \
  --format="table(name, guestCpus, memoryMb)"
        `

2.  **Refactor Pulumi Infrastructure (`infrastructure/__main__.py`):**

    - Remove the `gcp.compute.InstanceTemplate` resource.
    - Modify the `gcp.compute.RegionInstanceGroupManager` resource. Instead of referencing a single instance template, define the instance properties directly within the MIG's `version` block.
    - Provide the list of identified machine types to the `instance_template` property within the `version` block.
    - Ensure the MIG's `allocation_policy` is set to `BALANCED` to allow GCP to optimize for both availability and cost across the provided machine types.

3.  **Update Autoscaler (If Necessary):**
    - Ensure the autoscaler is still configured correctly after the MIG refactoring. The logic for memory-based scaling should remain the same.
