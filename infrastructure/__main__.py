import pulumi
import pulumi_gcp as gcp

# Get the GCP project and region from the Pulumi config
config = pulumi.Config("gcp")
gcp_project = config.require("project")
gcp_region = config.require("region")

# Create a GCS bucket to store the Zarr datacube
bucket = gcp.storage.Bucket("vci-datacube-bucket",
    location=gcp_region,
    project=gcp_project,
    uniform_bucket_level_access=True,
    storage_class="STANDARD",
    versioning=gcp.storage.BucketVersioningArgs(
        enabled=False,
    ),
    lifecycle_rules=[gcp.storage.BucketLifecycleRuleArgs(
        action=gcp.storage.BucketLifecycleRuleActionArgs(
            type="Delete",
        ),
        condition=gcp.storage.BucketLifecycleRuleConditionArgs(
            age=30, # Automatically delete objects older than 30 days
        ),
    )],
)

# Export the bucket name
pulumi.export("bucket_name", bucket.name)
pulumi.export("bucket_url", bucket.url)
