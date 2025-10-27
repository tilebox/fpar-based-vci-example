from pathlib import Path

from pulumi import Config, ResourceOptions, export
from pulumi_gcp.artifactregistry import Repository
from pulumi_gcp.projects import Service
from pulumi_gcp.storage import Bucket
from tilebox_infrastructure import AutoScalingGCPCluster, LocalBuildTrigger, Secret

# Get the GCP project and region from the Pulumi config
gcp_config = Config("gcp")
gcp_project = gcp_config.require("project")
gcp_region = gcp_config.require("region")

# Get other configuration from the vci-infrastructure namespace
infra_config = Config("vci-infrastructure")
cluster_enabled = infra_config.require_bool("cluster_enabled")
min_replicas = infra_config.require_int("min_replicas")
max_replicas = infra_config.get_int("max_replicas") or 10
machine_type = infra_config.get("machine_type") or "e2-standard-4"
cpu_target = infra_config.get_float("cpu_target") or 0.1
tilebox_cluster = infra_config.get("tilebox_cluster")
if tilebox_cluster is None:
    raise ValueError("Missing tilebox cluster")

# Get the Tilebox API key from Pulumi secrets
tilebox_config = Config("tilebox")
tilebox_api_key = tilebox_config.require_secret("api_key")

# Get the Axiom credentials from Pulumi secrets
axiom_config = Config("axiom")
axiom_api_key = axiom_config.require_secret("api_key")
axiom_logs_dataset = axiom_config.require("logs_dataset")
axiom_traces_dataset = axiom_config.require("traces_dataset")

workflow_dir = Path(__file__).parent.parent / "workflow"

# Enable necessary GCP services declaratively. This makes the Pulumi program
# self-contained and ensures that it can be run on a fresh GCP project.
artifact_registry_api = Service("artifact-registry-api", service="artifactregistry.googleapis.com")
storage_api = Service("storage-api", service="storage.googleapis.com")
secret_manager = Service("secret-manager-api", service="secretmanager.googleapis.com")
compute_api = Service("compute-api", service="compute.googleapis.com")
iam_api = Service("iam-api", service="iam.googleapis.com")
cloud_build_api = Service("cloud-build-api", service="cloudbuild.googleapis.com")

# Create an Artifact Registry repository to store our Docker images
repository = Repository(
    "vci-repository",
    location=gcp_region,
    repository_id="vci-runners",
    format="DOCKER",
    description="Repository for VCI workflow runner images.",
    opts=ResourceOptions(depends_on=[artifact_registry_api]),
)

build = LocalBuildTrigger(
    "vci-runner-image",
    gcp_region=gcp_region,
    gcp_project=gcp_project,
    repository_id=repository.repository_id,
    workflow_dir=workflow_dir,
    opts=ResourceOptions(depends_on=[repository]),
)

# Create a GCS bucket to store the Zarr datacube
bucket = Bucket(
    "vci-runner-bucket",
    location=gcp_region,
    project=gcp_project,
    uniform_bucket_level_access=True,
    storage_class="STANDARD",
    versioning={
        "enabled": False,
    },
    lifecycle_rules=[
        {
            "action": {
                "type": "Delete",
            },
            "condition": {
                "age": 30,  # Automatically delete objects older than 30 days
            },
        },
    ],
    opts=ResourceOptions(depends_on=[storage_api]),
)

secret_tilebox_api_key = Secret("tilebox-api-key", secret_data=tilebox_api_key)
secret_axiom_api_key = Secret("axiom-api-key", secret_data=axiom_api_key)

cluster = AutoScalingGCPCluster(
    "vci-runner",
    container={
        "image": build.container_image,
        "tag": build.tag,
    },
    environment_variables={
        "TILEBOX_API_KEY": secret_tilebox_api_key,
        "AXIOM_API_KEY": secret_axiom_api_key,
        "AXIOM_LOGS_DATASET": axiom_logs_dataset,
        "AXIOM_TRACES_DATASET": axiom_traces_dataset,
        "TILEBOX_CLUSTER": tilebox_cluster,
        "GCS_BUCKET": bucket.name,
    },
    roles={
        "roles": [
            "roles/logging.logWriter",  # write logs to the logging console
        ],
        "repository_roles": [
            {
                "repository_slug": "docker-container-images",
                "repository": repository,
                "role": "roles/artifactregistry.reader",  # pull the container image
            }
        ],
        "bucket_roles": [
            {
                "bucket_slug": "vci-runner-bucket",
                "bucket": bucket,
                "role": "roles/storage.objectUser",  # create, read, update and delete objects in the bucket
            },
        ],
    },
    gcp_project=gcp_project,
    gcp_region=gcp_region,
    machine_type=machine_type,
    cpu_target=cpu_target,
    cluster_enabled=cluster_enabled,
    min_replicas_config=min_replicas,
    max_replicas_config=max_replicas,
    opts=ResourceOptions(depends_on=[build, secret_tilebox_api_key, secret_axiom_api_key]),
)

export("bucket_name", bucket.name)
export("bucket_url", bucket.url)
export("container_image", build.container_image)
export("container_tag", build.tag)
