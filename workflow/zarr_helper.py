from typing import Literal
import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from zarr.storage import ObjectStore as ZarrObjectStore

from config import GCS_BUCKET, ZARR_STORE_PATH


def get_job_zarr_prefix(job_id: str) -> str:
    """Returns the job-specific Zarr path prefix."""
    return f"{ZARR_STORE_PATH}/{job_id}/cube.zarr"


def get_zarr_root(
    job_id: str, mode: Literal["r", "r+", "w"] = "r+", create_group: bool = False
) -> zarr.Group:
    """
    Opens and returns the root of the Zarr group for a given job.

    Args:
        job_id: The ID of the job.
        mode: The mode to open the Zarr group in (e.g., "r", "r+", "w").

    Returns:
        The root Zarr group object.
    """
    zarr_prefix = get_job_zarr_prefix(job_id)
    object_store = GCSStore(
        bucket=GCS_BUCKET,
        prefix=zarr_prefix,
        credential_provider=GoogleCredentialProvider(),
    )
    zarr_store = ZarrObjectStore(object_store)
    if create_group:
        return zarr.group(store=zarr_store, overwrite=True)
    else:
        # use_consolidated=False is important for concurrent writes in a parallel workflow
        return zarr.open_group(store=zarr_store, mode=mode, use_consolidated=False)
