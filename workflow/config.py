# --- Constants ---
GCS_BUCKET = "vci-datacube-bucket-1513742"
ZARR_STORE_PATH = "vci_fpar.zarr"
DATASET_ID = "tilebox.modis_fpar"
MODIS_COLLECTION = "MODIS"
VIIRS_COLLECTION = "VIIRS"
FILL_VALUE = 255

# --- Configuration ---
WIDTH = 80640
HEIGHT = 29346
TIME_CHUNK = 1
HEIGHT_CHUNK = 8192
WIDTH_CHUNK = 8192

from zarr.codecs import BloscCodec
COMPRESSOR = BloscCodec(cname="lz4hc", clevel=5, shuffle="shuffle")


def _calc_time_index(year: int, dekad: int, start_year: int, start_dekad: int) -> int:
    """
    Calculates a global, zero-based time index for a given year and dekad
    relative to a starting year and dekad. Assumes 36 dekads per year.
    """
    return (year - start_year) * 36 + (dekad - start_dekad)
