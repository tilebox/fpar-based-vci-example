### Workflow Plan: VCI Calculation

**Objective:** Calculate the Vegetation Condition Index (VCI) from MODIS and VIIRS FPAR data. This involves creating a unified Zarr datacube of FPAR values over a 25-year period, which will then be used to compute the per-pixel minimum and maximum FPAR values required for the VCI formula.

---

#### **Phase 1: Data Consolidation into Zarr Datacube**

This phase focuses on ingesting the raw `.tif` files into a single, analysis-ready Zarr datacube stored in Google Cloud Storage.

**1. Infrastructure Setup (Pulumi):**
-   **Location:** `infrastructure/`
-   **Project Type:** `uv` Python project.
-   **Task:** Create a Pulumi script to provision a Google Cloud Storage (GCS) bucket.
-   **Output:** The script will export the name of the created GCS bucket, which will be used by the workflow.
-   **Dependencies:** `pulumi-gcp`

**2. Workflow & Task Definition:**
-   **Location:** `workflow/`
-   **Project Type:** `uv` Python project.
-   **Dependencies:** `tilebox-sdk`, `zarr`, `gcsfs`, `xarray`, `rioxarray`, `numpy`.

**3. Workflow Steps:**

*   **Task 1: `InitializeZarrStore` (The First Task)**
    *   **Trigger:** Runs once at the beginning of the workflow.
    *   **Inputs:**
        *   GCS Bucket Name (from Pulumi output).
        *   Time range (configurable, e.g., "2020-01-01/2020-02-28").
        *   Static dimensions: `width: 80640`, `height: 29346`.
    *   **Action:**
        1.  Query the `tilebox.modis_fpar` dataset (both `MODIS` and `VIIRS` collections) for the given time range to get the exact number of assets. This count will be `num_dekads`.
        2.  Define the Zarr array shape: `(num_dekads, height, width)`.
        3.  Define chunks: `(1, 4819, 2016)`.
        4.  Define compressor: `Blosc(cname="lz4hc", clevel=5, shuffle="shuffle")`.
        5.  Initialize an empty Zarr array with this configuration in the specified GCS bucket.

*   **Task 2: `OrchestrateDataLoading` (The Main Workflow Task)**
    *   **Trigger:** Runs after `InitializeZarrStore` is complete.
    *   **Inputs:**
        *   Time range.
    *   **Action:**
        1.  Query the `tilebox.modis_fpar` dataset (both `MODIS` and `VIIRS` collections) to get a list of all available assets within the specified time range.
        2.  For each asset returned, submit a `LoadDekadIntoZarr` subtask.
        3.  Pass the asset's URL, timestamp, and dekad number to the subtask.

*   **Task 3: `LoadDekadIntoZarr` (The Parallel Workhorse Task)**
    *   **Trigger:** Submitted by `OrchestrateDataLoading`. Runs in parallel for each dekad.
    *   **Inputs:**
        *   URL of the `.tif` file.
        *   Timestamp of the data.
        *   The time index (slice number) in the Zarr array where this dekad's data should be written.
    *   **Action:**
        1.  Open the GeoTIFF file from the URL using `rioxarray`.
        2.  Access the initialized Zarr store on GCS.
        3.  Write the pixel data from the GeoTIFF into the appropriate time slice of the Zarr array.

---

#### **Phase 2: VCI Calculation (Future Work)**

This phase will be implemented after the Zarr datacube is successfully created and populated.

1.  **New Workflow:** A separate workflow will be created.
2.  **Input:** The path to the consolidated Zarr datacube in GCS.
3.  **Action:**
    *   Load the Zarr array using `xarray`.
    *   For each pixel, calculate the minimum and maximum FPAR value across the time dimension.
    *   Apply the VCI formula: `VCI = 100 * (FPAR - FPAR_min) / (FPAR_max - FPAR_min)`.
    *   Save the resulting VCI map (e.g., as a new Zarr array or GeoTIFF) back to GCS.
