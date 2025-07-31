import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib import colormaps
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from PIL import Image, ImageDraw, ImageFont
from tilebox.workflows import ExecutionContext, Task  # type: ignore[import-untyped]
from tilebox.workflows.observability.logging import get_logger  # type: ignore[import-untyped]
from zarr.storage import ObjectStore as ZarrObjectStore

from config import (
    GCS_BUCKET,
    ZARR_STORE_PATH,
    _calc_time_index,
    _calc_year_dekad_from_time_index,
)


def get_font(size: int = 24) -> Any:
    """Get a font for text rendering."""
    try:
        # Try to use a system font
        return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size)
    except OSError:
        try:
            # Fallback for Linux
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
            )
        except OSError:
            # Use default font as last resort
            return ImageFont.load_default()


class CreateVciGif(Task):
    """
    Main task to orchestrate VCI GIF creation.
    """

    job_id: str
    time_range: str | None = (
        None  # Optional datetime range like "2020-01-01/2021-12-31"
    )
    downsample_factor: int = 20  # Downsample factor for faster processing
    output_cluster: str | None = None  # Optional output cluster to download the gif on

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Starting VCI GIF creation...")

        # Store parameters in cache for subtasks
        context.job_cache["job_id"] = self.job_id.encode()
        context.job_cache["downsample_factor"] = str(self.downsample_factor).encode()

        # Convert datetime range to time indices if provided
        if self.time_range:
            start_str, end_str = self.time_range.split("/")
            start_datetime = datetime.fromisoformat(start_str)
            end_datetime = datetime.fromisoformat(end_str)

            # Convert datetime to year/dekad, then to time indices
            # Use known constants: Index 0 = year 2000, dekad 15
            start_year_dekad = (2000, 15)

            # Convert start datetime to year/dekad (simplified - assumes dekad 1 = Jan 1-10, etc.)
            start_year = start_datetime.year
            start_dekad = (
                ((start_datetime.month - 1) * 3) + ((start_datetime.day - 1) // 10) + 1
            )
            start_dekad = max(1, min(36, start_dekad))  # Clamp to valid dekad range

            end_year = end_datetime.year
            end_dekad = (
                ((end_datetime.month - 1) * 3) + ((end_datetime.day - 1) // 10) + 1
            )
            end_dekad = max(1, min(36, end_dekad))

            # Convert to time indices
            start_idx = _calc_time_index(start_year, start_dekad, *start_year_dekad)
            end_idx = (
                _calc_time_index(end_year, end_dekad, *start_year_dekad) + 1
            )  # +1 for inclusive end

            context.job_cache["time_range"] = pickle.dumps((start_idx, end_idx))
            logger.info(
                f"Converted datetime range {self.time_range} to time indices {start_idx}-{end_idx - 1}"
            )
        else:
            logger.info("No time range specified, will process all available data")

        # Submit subtask to create individual frames (organized by year)
        create_frames_task = context.submit_subtask(CreateVciFramesByYear())

        # Submit subtask to create GIF (depends on frames being created)
        create_gif_task = context.submit_subtask(
            CreateGifFromFrames(), depends_on=[create_frames_task]
        )

        # Submit subtask to download GIF (depends on GIF being created)
        context.submit_subtask(
            DownloadGifFromCache(),
            depends_on=[create_gif_task],
            cluster=self.output_cluster,
        )


class CreateVciFramesByYear(Task):
    """
    Orchestrates frame creation by year to respect the 64 subtask limit.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating VCI frame creation by year...")

        job_id = context.job_cache["job_id"].decode()

        # Load VCI data to get time range
        zarr_prefix = f"{ZARR_STORE_PATH}/{job_id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=zarr_prefix,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        ds = xr.open_zarr(zarr_store, consolidated=False)
        vci_array = ds["vci"]

        # Get time range
        if "time_range" in context.job_cache:
            start_idx, end_idx = pickle.loads(context.job_cache["time_range"])
        else:
            start_idx, end_idx = 0, vci_array.shape[0]

        # Use known constants for VCI datacube
        # Index 0 = year 2000, dekad 15 (first available in tilebox.fpar_modis dataset)
        start_year_dekad = (2000, 15)

        # Calculate which years we need to process
        start_year, start_dekad = _calc_year_dekad_from_time_index(
            start_idx, *start_year_dekad
        )
        end_year, end_dekad = _calc_year_dekad_from_time_index(
            end_idx - 1, *start_year_dekad
        )

        logger.info(f"Processing years {start_year} to {end_year}")

        # Submit one task per year, ensuring we stay within the desired time range
        for year in range(start_year, end_year + 1):
            # Calculate time index range for this year
            year_start_idx = _calc_time_index(year, 1, *start_year_dekad)
            year_end_idx = _calc_time_index(year + 1, 1, *start_year_dekad)

            # Clamp to the desired time range
            actual_start = max(start_idx, year_start_idx)
            actual_end = min(end_idx, year_end_idx)

            # Only submit if there are frames to create in this year
            if actual_start < actual_end:
                context.submit_subtask(
                    CreateVciFramesForYear(
                        year=year, time_start=actual_start, time_end=actual_end
                    )
                )
                logger.info(
                    f"Submitted year {year}: time indices {actual_start} to {actual_end - 1}"
                )

        logger.info(
            f"Submitted frame creation tasks for years {start_year} to {end_year}"
        )


class CreateVciFramesForYear(Task):
    """
    Creates frames for a specific year (up to 36 dekads = well under 64 subtask limit).
    """

    year: int
    time_start: int
    time_end: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(
            f"Creating VCI frames for year {self.year} (time indices {self.time_start} to {self.time_end})"
        )

        # Submit one subtask per time index (dekad) in this year
        for time_idx in range(self.time_start, self.time_end):
            context.submit_subtask(CreateSingleVciFrame(time_index=time_idx))

        logger.info(
            f"Submitted {self.time_end - self.time_start} frame creation tasks for year {self.year}"
        )


class CreateSingleVciFrame(Task):
    """
    Creates a single VCI frame for a specific time index.
    """

    time_index: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Creating VCI frame for time index {self.time_index}")

        job_id = context.job_cache["job_id"].decode()
        downsample_factor = int(context.job_cache["downsample_factor"].decode())

        # Load VCI data from Zarr
        zarr_prefix = f"{ZARR_STORE_PATH}/{job_id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=zarr_prefix,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        ds = xr.open_zarr(zarr_store, consolidated=False)

        vci_array = ds["vci"]

        # Validate time index is within bounds
        if self.time_index >= vci_array.shape[0]:
            logger.warning(
                f"Time index {self.time_index} is out of bounds (max: {vci_array.shape[0] - 1})"
            )
            return

        # Load and downsample VCI data for this specific time step
        vci_slice = vci_array[self.time_index, ::downsample_factor, ::downsample_factor]
        vci_data = vci_slice.compute()

        # Use known constants for VCI datacube
        # Index 0 = year 2000, dekad 15 (first available in tilebox.fpar_modis dataset)
        start_year_dekad = (2000, 15)

        # Create image from VCI data (stores directly in cache)
        self._create_frame_image(vci_data, self.time_index, start_year_dekad, context)

        logger.info(f"Created frame {self.time_index} in cache")

    def _create_frame_image(
        self,
        vci_data: xr.DataArray,
        time_idx: int,
        start_year_dekad: tuple[int, int],
        context: ExecutionContext,
    ) -> str:
        """Create a single frame image with VCI data, logo, and text."""

        # Convert VCI to colors using RdYlGn colormap
        # Handle NaN values by masking them
        vci_normalized = np.where(np.isnan(vci_data), 0, vci_data)
        vci_colors = colormaps["RdYlGn"](vci_normalized)

        # Convert to uint8 image
        img_array = (vci_colors * 255).astype(np.uint8)

        # Set transparent/NaN pixels to white
        alpha = img_array[:, :, 3]
        img_array[alpha == 0, :3] = 255

        # Create PIL image (RGB only) - this is the final image, no header
        final_image = Image.fromarray(img_array[:, :, :3])
        width, height = final_image.size

        # Load and add Tilebox logo to top right corner (overlaid on VCI data)
        logo_path = Path(__file__).parent / "tilebox-symbol-color@2x.png"
        if logo_path.exists():
            try:
                with Image.open(logo_path) as logo_img:
                    # Resize logo to reasonable size (10% of image width)
                    logo_width = int(width * 0.1)
                    logo_height = int(logo_img.height * logo_width / logo_img.width)
                    logo_resized = logo_img.resize(
                        (logo_width, logo_height), Image.Resampling.LANCZOS
                    )

                    # Paste logo in top right corner
                    logo_x = width - logo_width - 10
                    logo_y = 10
                    final_image.paste(
                        logo_resized,
                        (logo_x, logo_y),
                        logo_resized if logo_resized.mode == "RGBA" else None,
                    )
            except Exception as e:
                logger = get_logger()
                logger.warning(f"Could not add logo: {e}")

        # Calculate year/dekad from time index and add text in lower right corner
        year, dekad = _calc_year_dekad_from_time_index(time_idx, *start_year_dekad)
        text = f"{year} D{dekad:02d}"

        # Add larger text to lower right corner
        draw = ImageDraw.Draw(final_image)
        font = get_font(32)  # Larger font size

        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position text in lower right corner with padding
        padding = 8
        text_x = width - text_width - padding
        text_y = height - text_height - padding

        # Add semi-transparent white background for better readability
        bg_padding = 5
        bg_bbox = [
            text_x - bg_padding,
            text_y - bg_padding,
            text_x + text_width + bg_padding,
            text_y + text_height + bg_padding,
        ]
        draw.rectangle(bg_bbox, fill=(255, 255, 255, 200))  # Semi-transparent white

        # Draw the text in black for good contrast
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        # Save frame to cache instead of local file system
        frame_filename = f"vci_frame_{time_idx:04d}.png"

        # Save image to bytes buffer
        from io import BytesIO

        img_buffer = BytesIO()
        final_image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        # Store image in cache group for frames
        frames_group = context.job_cache.group("vci_frames")
        frame_key = f"frame_{time_idx:04d}"
        frames_group[frame_key] = img_bytes

        # Also store the filename for reference
        frames_group[f"filename_{time_idx:04d}"] = frame_filename.encode()

        return frame_key


class CreateGifFromFrames(Task):
    """
    Creates a GIF from the individual frame images using ffmpeg.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Creating GIF from frames...")

        # Collect frame paths from cache (stored by individual CreateSingleVciFrame tasks)
        frame_paths = []

        # Get the time range to know which frames to look for
        if "time_range" in context.job_cache:
            start_idx, end_idx = pickle.loads(context.job_cache["time_range"])
        else:
            # If no time range specified, scan all cache entries for frames
            start_idx, end_idx = 0, 10000  # Large number to scan all

        # Collect frame images from cache group and create temporary files
        import tempfile

        temp_dir = tempfile.mkdtemp()

        frames_group = context.job_cache.group("vci_frames")

        # Collect all frame keys and sort them
        frame_keys = [key for key in frames_group if key.startswith("frame_")]
        frame_keys.sort()  # Ensure chronological order

        logger.info(f"Found {len(frame_keys)} frames in cache")

        for frame_key in frame_keys:
            # Extract time index from key (e.g., "frame_0019" -> 19)
            time_idx = int(frame_key.split("_")[1])

            # Check if this frame is within our desired range
            if start_idx <= time_idx < end_idx:
                # Retrieve image bytes from cache group
                img_bytes = frames_group[frame_key]

                # Create temporary file for ffmpeg
                temp_filename = f"vci_frame_{time_idx:04d}.png"
                temp_path = os.path.join(temp_dir, temp_filename)

                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

                frame_paths.append(temp_path)

        if not frame_paths:
            logger.error("No frame paths found in cache")
            return

        logger.info(f"Found {len(frame_paths)} frames to combine into GIF")

        # Create GIF using ffmpeg
        output_path = os.path.join(temp_dir, "vci_animation.gif")
        palette_path = os.path.join(temp_dir, "palette.png")
        frame_pattern = os.path.join(temp_dir, "vci_frame_*.png")

        try:
            # Use ffmpeg to create GIF with good quality
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate",
                "2",  # 2 frames per second
                "-pattern_type",
                "glob",
                "-i",
                frame_pattern,
                "-vf",
                "palettegen=reserve_transparent=0",  # Generate palette
                palette_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Create GIF with the generated palette
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "2",
                "-pattern_type",
                "glob",
                "-i",
                frame_pattern,
                "-i",
                palette_path,
                "-lavfi",
                "paletteuse",
                output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            logger.info(f"Successfully created GIF: {output_path}")

            # Store GIF in cache for download
            with open(output_path, "rb") as f:
                gif_bytes = f.read()

            gif_cache_key = "vci_animation_gif"
            context.job_cache[gif_cache_key] = gif_bytes

            logger.info(f"GIF stored in cache with key: {gif_cache_key}")
            logger.info(f"GIF size: {len(gif_bytes) / 1024 / 1024:.1f} MB")

            # Clean up temporary directory
            import shutil

            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")
            except OSError as e:
                logger.warning(f"Could not clean up temp directory: {e}")

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e}")
            logger.error(f"stderr: {e.stderr.decode() if e.stderr else 'No stderr'}")
            raise
        except Exception as e:
            logger.error(f"Failed to create GIF: {e}")
            raise


class DownloadGifFromCache(Task):
    """
    Downloads the GIF from the cache.
    """

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Downloading GIF from cache...")

        gif_cache_key = "vci_animation_gif"
        if gif_cache_key in context.job_cache:
            gif_bytes = context.job_cache[gif_cache_key]
            with open("vci_animation.gif", "wb") as f:
                f.write(gif_bytes)
            logger.info("GIF downloaded successfully")
        else:
            logger.error("GIF not found in cache")
