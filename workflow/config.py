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
