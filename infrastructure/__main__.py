import hashlib
import os
from pathlib import Path

import pulumi
import pulumi_command as command
import pulumi_gcp as gcp

# --- Helper Function ---


def hash_directory(directory: Path) -> str:
    """
    Computes a stable SHA256 hash of a directory's contents.
    This is used to generate a unique, content-based tag for the Docker image.
    The build is only triggered if this hash changes.
    """
    directory_hash = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            file_path = Path(root) / name

            # Add file names to the hash sum, to account for file moves
            directory_hash.update(str(file_path.relative_to(directory)).encode())

            with file_path.open("rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    directory_hash.update(chunk)

    return directory_hash.hexdigest()


# --- Configuration ---

# Get the GCP project and region from the Pulumi config
gcp_config = pulumi.Config("gcp")
gcp_project = gcp_config.require("project")
gcp_region = gcp_config.require("region")

# Get other configuration from the vci-infrastructure namespace
infra_config = pulumi.Config("vci-infrastructure")
cluster_enabled = infra_config.require_bool("cluster_enabled")
min_replicas_config = infra_config.require_int("min_replicas")
max_replicas_config = infra_config.get_int("max_replicas") or 10
machine_type = infra_config.get("machine_type") or "e2-standard-4"
cpu_target = infra_config.get_float("cpu_target") or 0.1
tilebox_cluster = infra_config.get("tilebox_cluster")
if tilebox_cluster is None:
    raise ValueError("Missing tilebox cluster")

# Get the Tilebox API key from Pulumi secrets
tilebox_config = pulumi.Config("tilebox")
tilebox_api_key = tilebox_config.require_secret("api_key")

# Get the Axiom credentials from Pulumi secrets
axiom_config = pulumi.Config("axiom")
axiom_api_key = axiom_config.require_secret("api_key")
axiom_logs_dataset = axiom_config.require("logs_dataset")
axiom_traces_dataset = axiom_config.require("traces_dataset")


# --- API Enabler ---

# Enable necessary GCP services declaratively. This makes the Pulumi program
# self-contained and ensures that it can be run on a fresh GCP project.
compute_api = gcp.projects.Service("compute-api", service="compute.googleapis.com")
artifact_registry_api = gcp.projects.Service("artifact-registry-api", service="artifactregistry.googleapis.com")
iam_api = gcp.projects.Service("iam-api", service="iam.googleapis.com")
storage_api = gcp.projects.Service("storage-api", service="storage.googleapis.com")
cloud_build_api = gcp.projects.Service("cloud-build-api", service="cloudbuild.googleapis.com")


# --- Resources ---

# Create an Artifact Registry repository to store our Docker images
repository = gcp.artifactregistry.Repository(
    "vci-repository",
    location=gcp_region,
    repository_id="vci-runners",
    format="DOCKER",
    description="Repository for VCI workflow runner images.",
    opts=pulumi.ResourceOptions(depends_on=[artifact_registry_api]),
)

# Create a dedicated service account for the runner instances to follow the
# principle of least privilege.
runner_sa = gcp.serviceaccount.Account(
    "vci-runner-sa",
    account_id="vci-runner-sa",
    display_name="VCI Workflow Runner Service Account",
    opts=pulumi.ResourceOptions(depends_on=[iam_api]),
)

# Grant the service account only the specific roles it needs.
gcp.projects.IAMMember(
    "vci-runner-sa-storage-admin",
    project=gcp_project,
    role="roles/storage.objectAdmin",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)
gcp.projects.IAMMember(
    "vci-runner-sa-artifact-reader",
    project=gcp_project,
    role="roles/artifactregistry.reader",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)
gcp.projects.IAMMember(
    "vci-runner-sa-log-writer",
    project=gcp_project,
    role="roles/logging.logWriter",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)
gcp.projects.IAMMember(
    "vci-runner-sa-metric-writer",
    project=gcp_project,
    role="roles/monitoring.metricWriter",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)

# --- Cloud Build ---

# Calculate the hash of the workflow code to use as an immutable image tag.

workflow_dir = Path(__file__).parent.parent.absolute() / "workflow"
code_hash = hash_directory(workflow_dir)

# Use proper Artifact Registry domain for the repository
base_image_name = pulumi.Output.concat(gcp_region, "-docker.pkg.dev/", gcp_project, "/vci-runners/workflow-runner")
image_name_with_tag = pulumi.Output.concat(base_image_name, ":", code_hash)

# Use a Command resource to trigger Google Cloud Build.
cloud_build = command.local.Command(
    "cloud-build-image",
    create=pulumi.Output.concat(
        "gcloud builds submit ",
        str(workflow_dir),
        " --config=",
        str(workflow_dir / "cloudbuild.yaml"),
        " --substitutions=_CODE_HASH=",
        code_hash,
        " --project=",
        gcp_project,
    ),
    # The 'triggers' property ensures this command re-runs when the code changes.
    triggers=[code_hash],
    opts=pulumi.ResourceOptions(depends_on=[cloud_build_api, repository]),
)

# --- Networking ---

# Create a Cloud Router, which is a prerequisite for Cloud NAT.
router = gcp.compute.Router(
    "vci-router",
    name="vci-router",
    network="default",
    region=gcp_region,
    opts=pulumi.ResourceOptions(depends_on=[compute_api]),
)

# Create a Cloud NAT gateway. This allows the VMs to make outbound connections
# (e.g., to pull Docker images) without having public IP addresses, which is a
# critical security best practice.
nat = gcp.compute.RouterNat(
    "vci-nat",
    name="vci-nat",
    router=router.name,
    region=gcp_region,
    source_subnetwork_ip_ranges_to_nat="ALL_SUBNETWORKS_ALL_IP_RANGES",
    nat_ip_allocate_option="AUTO_ONLY",
    opts=pulumi.ResourceOptions(depends_on=[router]),
)


# --- Storage ---

# Create a GCS bucket to store the Zarr datacube
bucket = gcp.storage.Bucket(
    "vci-datacube-bucket",
    location=gcp_region,
    project=gcp_project,
    uniform_bucket_level_access=True,
    storage_class="STANDARD",
    versioning=gcp.storage.BucketVersioningArgs(
        enabled=False,
    ),
    lifecycle_rules=[
        gcp.storage.BucketLifecycleRuleArgs(
            action=gcp.storage.BucketLifecycleRuleActionArgs(
                type="Delete",
            ),
            condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                age=30,  # Automatically delete objects older than 30 days
            ),
        )
    ],
    opts=pulumi.ResourceOptions(depends_on=[storage_api]),
)


# --- Compute ---

# Define the Instance Template for the MIG
instance_template = gcp.compute.InstanceTemplate(
    "vci-runner-template",
    machine_type=machine_type,
    tags=["vci-runner"],
    metadata={
        "gce-container-declaration": pulumi.Output.concat(
            "spec:\n",
            "  containers:\n",
            "  - name: vci-workflow-runner\n",
            "    image: '",
            image_name_with_tag,
            "'\n",
            "    env:\n",
            "    - name: TILEBOX_API_KEY\n",
            "      value: '",
            tilebox_api_key,
            "'\n",
            "    - name: AXIOM_API_KEY\n",
            "      value: '",
            axiom_api_key,
            "'\n",
            "    - name: AXIOM_LOGS_DATASET\n",
            "      value: '",
            axiom_logs_dataset,
            "'\n",
            "    - name: AXIOM_TRACES_DATASET\n",
            "      value: '",
            axiom_traces_dataset,
            "'\n",
            "    - name: TILEBOX_CLUSTER\n",
            "      value: '",
            tilebox_cluster,
            "'\n",
            "    - name: GCS_BUCKET\n",
            "      value: '",
            bucket.name,
            "'\n",
            "    - name: ZARR_STORE_PATH\n",
            "      value: 'gs://",
            bucket.name,
            "'\n",
            "    stdin: false\n",
            "    tty: false\n",
            "  restartPolicy: Always\n",
        ),
        # This is the correct metadata key to enable the Ops Agent for monitoring
        # (including memory) on Container-Optimized OS.
        "google-monitoring-enabled": "true",
    },
    disks=[
        gcp.compute.InstanceTemplateDiskArgs(
            source_image="cos-cloud/cos-stable",
            auto_delete=True,
            boot=True,
        )
    ],
    network_interfaces=[gcp.compute.InstanceTemplateNetworkInterfaceArgs(network="default")],
    service_account=gcp.compute.InstanceTemplateServiceAccountArgs(
        email=runner_sa.email,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
    # Use Spot VMs for cost savings. The API requires these specific scheduling options.
    scheduling=gcp.compute.InstanceTemplateSchedulingArgs(
        provisioning_model="SPOT",
        preemptible=True,
        automatic_restart=False,
        on_host_maintenance="TERMINATE",
    ),
    opts=pulumi.ResourceOptions(depends_on=[compute_api, runner_sa, cloud_build, bucket]),
)


# Define the Managed Instance Group
mig = gcp.compute.RegionInstanceGroupManager(
    "vci-runner-mig",
    base_instance_name="vci-runner",
    region=gcp_region,
    versions=[
        gcp.compute.RegionInstanceGroupManagerVersionArgs(
            instance_template=instance_template.self_link,
            name="primary",
        )
    ],
    update_policy=gcp.compute.RegionInstanceGroupManagerUpdatePolicyArgs(
        type="PROACTIVE",
        minimal_action="REPLACE",
        # Increase surge for faster rollouts
        max_surge_fixed=10,
        max_unavailable_fixed=0,
    ),
    opts=pulumi.ResourceOptions(depends_on=[instance_template]),
)

# --- Cluster On/Off Logic ---

if cluster_enabled:
    # If the cluster is enabled, the autoscaler is ON and controls the size.
    # The MIG's target_size is not set, ceding control to the autoscaler,
    # which will scale up to min_replicas immediately.
    min_replicas = min_replicas_config
    max_replicas = max_replicas_config
else:
    # If the cluster is disabled, the autoscaler is turned OFF.
    # The MIG's target_size is explicitly set to 0 to shut down all instances.
    min_replicas = 0
    max_replicas = 0


# Define the Autoscaler for the MIG
autoscaler = gcp.compute.RegionAutoscaler(
    "vci-runner-autoscaler",
    target=mig.self_link,
    region=gcp_region,
    autoscaling_policy=gcp.compute.RegionAutoscalerAutoscalingPolicyArgs(
        max_replicas=max_replicas,
        min_replicas=min_replicas,
        cooldown_period=60,
        mode="ON",
        cpu_utilization=gcp.compute.RegionAutoscalerAutoscalingPolicyCpuUtilizationArgs(
            target=cpu_target,
        ),
    ),
    opts=pulumi.ResourceOptions(depends_on=[mig]),
)

# --- Exports ---
pulumi.export("bucket_name", bucket.name)
pulumi.export("bucket_url", bucket.url)
pulumi.export("imageName", image_name_with_tag)
pulumi.export("migName", mig.name)
