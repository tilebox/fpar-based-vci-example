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


def _calc_year_dekad_from_time_index(time_index: int, start_year: int, start_dekad: int) -> tuple[int, int]:
    """
    Calculates year and dekad from a time index.
    Inverse of _calc_time_index function.

    Args:
        time_index: Zero-based time index
        start_year: Starting year
        start_dekad: Starting dekad

    Returns:
        Tuple of (year, dekad)
    """
    # Calculate total dekads from start
    total_dekads = time_index + start_dekad

    # Calculate year offset (how many full years)
    year_offset = (total_dekads - 1) // 36  # -1 because dekads are 1-based

    # Calculate the dekad within the year
    dekad = total_dekads - (year_offset * 36)

    # If dekad > 36, we need to adjust
    if dekad > 36:
        year_offset += 1
        dekad = dekad - 36

    year = start_year + year_offset

    return year, dekad
