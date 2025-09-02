"""Zarr store utilities for VCI workflow."""

import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from zarr.core.common import AccessModeLiteral
from zarr.storage import ObjectStore as ZarrObjectStore

from fpar_to_vci.config import GCS_BUCKET, ZARR_STORE_PATH


def open_zarr_store(path: str) -> ZarrObjectStore:
    """Open a Zarr group from GCS."""
    object_store = GCSStore(
        bucket=GCS_BUCKET,
        prefix=path,
        credential_provider=GoogleCredentialProvider(),
    )
    return ZarrObjectStore(object_store)


def open_zarr_group(path: str, mode: AccessModeLiteral = "r") -> zarr.Group:
    return zarr.open_group(store=open_zarr_store(path), mode=mode)


def job_path(job_id: str, name: str) -> str:
    """Generate deterministic zarr path for job artifacts."""

    return f"{ZARR_STORE_PATH}/{job_id}/{name}.zarr"
