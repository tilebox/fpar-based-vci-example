from pathlib import Path

from pulumi import Config, ResourceOptions, export
from pulumi_aws import get_caller_identity
from pulumi_aws.ec2 import SecurityGroup, SecurityGroupEgressArgs
from pulumi_aws.ecr import Repository, RepositoryImageScanningConfigurationArgs
from pulumi_aws.s3 import (
    Bucket,
    BucketLifecycleConfiguration,
    BucketLifecycleConfigurationRuleArgs,
    BucketLifecycleConfigurationRuleExpirationArgs,
    BucketLifecycleConfigurationRuleFilterArgs,
)
from tilebox_iac.aws import AutoScalingCluster, LocalBuildTrigger, Network, Secret

# Get the AWS region from the Pulumi config
aws_config = Config("aws")
aws_region = aws_config.require("region")

# Get the AWS account ID from caller identity

aws_account_id = get_caller_identity().account_id

# Get other configuration from the vci-infrastructure namespace
infra_config = Config("vci-infrastructure")
cluster_enabled = infra_config.require_bool("cluster_enabled")
min_replicas = infra_config.require_int("min_replicas")
max_replicas = infra_config.get_int("max_replicas") or 10
instance_type = infra_config.get("instance_type") or "t3.medium"
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

# Create an ECR repository to store our Docker images
repository = Repository(
    "vci-repository",
    name="vci-runners",
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=RepositoryImageScanningConfigurationArgs(scan_on_push=False),
)

build = LocalBuildTrigger(
    "vci-runner-image",
    aws_region=aws_region,
    aws_account_id=aws_account_id,
    repository_name=repository.name,
    source_dir=workflow_dir,
    opts=ResourceOptions(depends_on=[repository]),
)

network = Network(
    "vci-runner-network",
    aws_region=aws_region,
    enable_s3_endpoint=True,
    enable_internet_access=True,
)

# Create a security group for the instances
security_group = SecurityGroup(
    "vci-runner-sg",
    vpc_id=network.vpc_id,
    description="Security group for VCI runner instances",
    egress=[
        SecurityGroupEgressArgs(
            from_port=0,
            to_port=0,
            protocol="-1",
            cidr_blocks=["0.0.0.0/0"],
            description="Allow all outbound traffic",
        ),
    ],
    opts=ResourceOptions(depends_on=[network]),
)

# Create an S3 bucket to store the Zarr datacube
bucket = Bucket(
    "vci-runner-bucket",
    bucket_prefix="vci-runner-",
)

BucketLifecycleConfiguration(
    "vci-runner-bucket-lifecycle",
    bucket=bucket.id,
    rules=[
        BucketLifecycleConfigurationRuleArgs(
            id="delete-old-objects",
            status="Enabled",
            filter=BucketLifecycleConfigurationRuleFilterArgs(),
            expiration=BucketLifecycleConfigurationRuleExpirationArgs(
                days=30,
            ),
        ),
    ],
    opts=ResourceOptions(depends_on=[bucket]),
)

secret_tilebox_api_key = Secret("tilebox-api-key", tilebox_api_key)
secret_axiom_api_key = Secret("axiom-api-key", axiom_api_key)

cluster = AutoScalingCluster(
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
        "S3_BUCKET": bucket.bucket,
    },
    iam_config={
        "managed_policies": [
            "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        ],
        "bucket_access": [
            {
                "bucket_slug": "vci-runner-bucket",
                "bucket_arn": bucket.arn,
                "access_level": "readwrite",
            },
        ],
    },
    instance_type=instance_type,
    cpu_target=cpu_target,
    cluster_enabled=cluster_enabled,
    min_replicas_config=min_replicas,
    max_replicas_config=max_replicas,
    subnet_ids=[network.private_subnet_id],
    security_group_ids=[security_group.id],
    opts=ResourceOptions(depends_on=[build, secret_tilebox_api_key, secret_axiom_api_key, network, security_group]),
)

export("bucket_name", bucket.bucket)
export("bucket_arn", bucket.arn)
export("container_image", build.container_image)
export("container_tag", build.tag)
