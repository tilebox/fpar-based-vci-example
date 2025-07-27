import rioxarray
import sys
import xarray as xr

def inspect_geotiff(file_path):
    """Opens a GeoTIFF file and prints its metadata."""
    try:
        with rioxarray.open_rasterio(file_path) as data:
            print("--- GeoTIFF Inspection ---")
            print(f"File: {file_path}")
            print("\n--- Dimensions ---")
            for dim, size in data.sizes.items():
                print(f"- {dim}: {size}")

            print("\n--- Coordinates ---")
            for coord_name, coord_val in data.coords.items():
                print(f"- {coord_name}:")
                print(f"  - Values: {coord_val.values.min()} to {coord_val.values.max()}")
                print(f"  - Dtype: {coord_val.dtype}")

            print("\n--- Data ---")
            print(f"  - Dtype: {data.dtype}")
            print(f"  - Shape: {data.shape}")
            print(f"  - Min value: {data.values.min()}")
            print(f"  - Max value: {data.values.max()}")

            print("\n--- Attributes ---")
            for attr_name, attr_val in data.attrs.items():
                print(f"- {attr_name}: {attr_val}")
            
            print("\n--- CRS ---")
            print(data.rio.crs)

    except Exception as e:
        print(f"Error inspecting GeoTIFF {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_geotiff.py <path_to_geotiff>", file=sys.stderr)
        sys.exit(1)
    
    geotiff_path = sys.argv[1]
    inspect_geotiff(geotiff_path)
