import os

import zarr
from zarr.codecs import BloscCodec
from zarr.core.common import AccessModeLiteral
from zarr.storage import ObjectStore as ZarrObjectStore

# --- Constants ---
GCS_BUCKET = os.environ.get("GCS_BUCKET")
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")
FILL_VALUE = 255
START_YEAR_DEKAD = (2000, 15)

# --- Configuration ---
WIDTH = 80640
HEIGHT = 29346
TIME_CHUNK = 1
HEIGHT_CHUNK = 4096
WIDTH_CHUNK = 4096
NUM_DEKADS = 36


COMPRESSOR = BloscCodec(cname="lz4hc", clevel=5, shuffle="shuffle")


def open_zarr_store(path: str) -> ZarrObjectStore:
    """Open a Zarr store from GCS or S3 based on environment."""
    if S3_BUCKET:
        from obstore.store import S3Store

        object_store = S3Store(
            bucket=S3_BUCKET,
            prefix=path,
            region=AWS_REGION,
        )
    elif GCS_BUCKET:
        from obstore.auth.google import GoogleCredentialProvider
        from obstore.store import GCSStore

        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=path,
            credential_provider=GoogleCredentialProvider(),
        )
    else:
        raise ValueError("Either GCS_BUCKET or S3_BUCKET environment variable must be set")

    return ZarrObjectStore(object_store)


def open_zarr_group(path: str, mode: AccessModeLiteral = "r") -> zarr.Group:
    return zarr.open_group(store=open_zarr_store(path), mode=mode)
