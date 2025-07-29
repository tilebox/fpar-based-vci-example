import pulumi
import pulumi_gcp as gcp
import pulumi_command as command
import os
import hashlib

# --- Helper Function ---

def hash_directory(path):
    """
    Computes a stable SHA256 hash of a directory's contents.
    """
    hasher = hashlib.sha256()
    for root, _, files in os.walk(path):
        for name in sorted(files):
            file_path = os.path.join(root, name)
            # Add relative path to hash to account for file moves
            relative_path = os.path.relpath(file_path, path)
            hasher.update(relative_path.encode())
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    hasher.update(chunk)
    return hasher.hexdigest()

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
machine_type = infra_config.get("machine_type") or "e2-standard-2"

# Get the Tilebox API key from Pulumi secrets
tilebox_config = pulumi.Config("tilebox")
tilebox_api_key = tilebox_config.require_secret("api_key")

# Get the Axiom credentials from Pulumi secrets
axiom_config = pulumi.Config("axiom")
axiom_api_key = axiom_config.require_secret("api_key")
axiom_logs_dataset = axiom_config.require("logs_dataset")
axiom_traces_dataset = axiom_config.require("traces_dataset")


# --- API Enabler ---

# Enable necessary GCP services
compute_api = gcp.projects.Service("compute-api", service="compute.googleapis.com")
artifact_registry_api = gcp.projects.Service("artifact-registry-api", service="artifactregistry.googleapis.com")
iam_api = gcp.projects.Service("iam-api", service="iam.googleapis.com")
storage_api = gcp.projects.Service("storage-api", service="storage.googleapis.com")
cloud_build_api = gcp.projects.Service("cloud-build-api", service="cloudbuild.googleapis.com")


# --- Resources ---

# Create an Artifact Registry repository to store our Docker images
repository = gcp.artifactregistry.Repository("vci-repository",
    location=gcp_region,
    repository_id="vci-runners",
    format="DOCKER",
    description="Repository for VCI workflow runner images.",
    opts=pulumi.ResourceOptions(depends_on=[artifact_registry_api]),
)

# Create a dedicated service account for the runner instances
runner_sa = gcp.serviceaccount.Account("vci-runner-sa",
    account_id="vci-runner-sa",
    display_name="VCI Workflow Runner Service Account",
    opts=pulumi.ResourceOptions(depends_on=[iam_api]),
)

# Grant the service account the necessary roles
gcp.projects.IAMMember("vci-runner-sa-storage-admin",
    project=gcp_project,
    role="roles/storage.objectAdmin",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)
gcp.projects.IAMMember("vci-runner-sa-artifact-reader",
    project=gcp_project,
    role="roles/artifactregistry.reader",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)
gcp.projects.IAMMember("vci-runner-sa-log-writer",
    project=gcp_project,
    role="roles/logging.logWriter",
    member=pulumi.Output.concat("serviceAccount:", runner_sa.email),
    opts=pulumi.ResourceOptions(depends_on=[runner_sa]),
)

# --- Cloud Build ---

# Calculate the hash of the workflow code to use as an immutable image tag
workflow_dir = os.path.join(os.path.dirname(__file__), "../workflow")
code_hash = hash_directory(workflow_dir)

# Define the base image name
base_image_name = pulumi.Output.concat(
    "eu.gcr.io/",
    gcp_project,
    "/vci-runners/workflow-runner"
)
# Define the fully-qualified image name with the code hash as the tag
image_name_with_tag = pulumi.Output.concat(base_image_name, ":", code_hash)

# Use a Command resource to trigger Google Cloud Build.
cloud_build = command.local.Command("cloud-build-image",
    create=pulumi.Output.concat(
        "gcloud builds submit ",
        workflow_dir,
        " --tag=", image_name_with_tag,
        " --project=", gcp_project
    ),
    triggers=[code_hash],
    opts=pulumi.ResourceOptions(depends_on=[cloud_build_api, repository]),
)

# --- Networking ---

# Create a Cloud Router
router = gcp.compute.Router("vci-router",
    name="vci-router",
    network="default",
    region=gcp_region,
    opts=pulumi.ResourceOptions(depends_on=[compute_api]),
)

# Create a Cloud NAT gateway to allow outbound internet access
nat = gcp.compute.RouterNat("vci-nat",
    name="vci-nat",
    router=router.name,
    region=gcp_region,
    source_subnetwork_ip_ranges_to_nat="ALL_SUBNETWORKS_ALL_IP_RANGES",
    nat_ip_allocate_option="AUTO_ONLY",
    opts=pulumi.ResourceOptions(depends_on=[router]),
)


# --- Compute ---

# Define the Instance Template for the MIG
instance_template = gcp.compute.InstanceTemplate("vci-runner-template",
    machine_type=machine_type,
    tags=["vci-runner"],
    metadata={
        "gce-container-declaration": pulumi.Output.concat(
            "spec:\n",
            "  containers:\n",
            "  - name: vci-workflow-runner\n",
            "    image: '", image_name_with_tag, "'\n",
            "    env:\n",
            "    - name: TILEBOX_API_KEY\n",
            "      value: '", tilebox_api_key, "'\n",
            "    - name: AXIOM_API_KEY\n",
            "      value: '", axiom_api_key, "'\n",
            "    - name: AXIOM_LOGS_DATASET\n",
            "      value: '", axiom_logs_dataset, "'\n",
            "    - name: AXIOM_TRACES_DATASET\n",
            "      value: '", axiom_traces_dataset, "'\n",
            "    stdin: false\n",
            "    tty: false\n",
            "  restartPolicy: Always\n"
        )
    },
    disks=[gcp.compute.InstanceTemplateDiskArgs(
        source_image="cos-cloud/cos-stable",
        auto_delete=True,
        boot=True,
    )],
    network_interfaces=[gcp.compute.InstanceTemplateNetworkInterfaceArgs(
        network="default",
    )],
    service_account=gcp.compute.InstanceTemplateServiceAccountArgs(
        email=runner_sa.email,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
    scheduling=gcp.compute.InstanceTemplateSchedulingArgs(
        provisioning_model="SPOT",
        preemptible=True,
        automatic_restart=False,
        on_host_maintenance="TERMINATE",
    ),
    opts=pulumi.ResourceOptions(depends_on=[compute_api, runner_sa, cloud_build]),
)

# --- Cluster On/Off Logic ---

if cluster_enabled:
    # If the cluster is enabled, the autoscaler is ON and controls the size.
    # The MIG's target_size is not set, ceding control to the autoscaler.
    autoscaler_mode = "ON"
    min_replicas = min_replicas_config
    mig_target_size = None # Omit target_size to let autoscaler manage it
else:
    # If the cluster is disabled, the autoscaler is turned OFF.
    # The MIG's target_size is explicitly set to 0.
    autoscaler_mode = "OFF"
    min_replicas = 0
    mig_target_size = 0

# Define the Managed Instance Group
mig = gcp.compute.RegionInstanceGroupManager("vci-runner-mig",
    base_instance_name="vci-runner",
    region=gcp_region,
    versions=[gcp.compute.RegionInstanceGroupManagerVersionArgs(
        instance_template=instance_template.self_link,
        name="primary",
    )],
    target_size=mig_target_size,
    update_policy=gcp.compute.RegionInstanceGroupManagerUpdatePolicyArgs(
        type="PROACTIVE",
        minimal_action="REPLACE",
        max_surge_fixed=3,
        max_unavailable_fixed=0,
    ),
    opts=pulumi.ResourceOptions(depends_on=[instance_template]),
)

# Define the Autoscaler for the MIG
autoscaler = gcp.compute.RegionAutoscaler("vci-runner-autoscaler",
    target=mig.self_link,
    region=gcp_region,
    autoscaling_policy=gcp.compute.RegionAutoscalerAutoscalingPolicyArgs(
        max_replicas=max_replicas_config,
        min_replicas=min_replicas,
        cooldown_period=60,
        mode=autoscaler_mode,
        cpu_utilization=gcp.compute.RegionAutoscalerAutoscalingPolicyCpuUtilizationArgs(
            target=0.2,
        ),
    ),
    opts=pulumi.ResourceOptions(depends_on=[mig]),
)

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
            age=30,
        ),
    )],
    opts=pulumi.ResourceOptions(depends_on=[storage_api]),
)

# --- Exports ---
pulumi.export("bucket_name", bucket.name)
pulumi.export("bucket_url", bucket.url)
pulumi.export("imageName", image_name_with_tag)
pulumi.export("migName", mig.name)
