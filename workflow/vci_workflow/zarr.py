import zarr
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from zarr.codecs import BloscCodec
from zarr.core.common import AccessModeLiteral
from zarr.storage import ObjectStore as ZarrObjectStore

# --- Constants ---
GCS_BUCKET = "vci-datacube-bucket-1513742"
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
    """Open a Zarr group from GCS."""
    object_store = GCSStore(
        bucket=GCS_BUCKET,
        prefix=path,
        credential_provider=GoogleCredentialProvider(),
    )
    return ZarrObjectStore(object_store)


def open_zarr_group(path: str, mode: AccessModeLiteral = "r") -> zarr.Group:
    return zarr.open_group(store=open_zarr_store(path), mode=mode)
