from functools import lru_cache
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


@lru_cache
def assets_store() -> GCSStore:
    return GCSStore(
        bucket=GCS_BUCKET,
        prefix="assets",
        credential_provider=GoogleCredentialProvider(),
    )


@lru_cache
def get_font(size: int = 24, style="Regular") -> Any:
    """Get a font for text rendering with system-specific fallbacks."""
    local = Path(f"Poppins-{style}.ttf")
    if not local.exists():
        assets_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix="assets",
            credential_provider=GoogleCredentialProvider(),
        )
        with local.open("wb") as f:
            f.write(memoryview(assets_store.get(str(local)).bytes()))

    try:
        return ImageFont.truetype(str(local.absolute()), size)
    except OSError:
        return ImageFont.load_default(size)


@lru_cache
def get_logo(scale: float = 0.7) -> Any:
    """Get a font for text rendering with system-specific fallbacks."""
    local = Path("tilebox-logo.png")
    if not local.exists():
        assets_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix="assets",
            credential_provider=GoogleCredentialProvider(),
        )
        with local.open("wb") as f:
            f.write(memoryview(assets_store.get(str(local)).bytes()))

    logo = Image.open(str(local))
    logo = logo.resize((int(logo.size[0] * scale), int(logo.size[1] * scale)))
    return logo


class CreateVciVideo(Task):
    """Main task to orchestrate VCI MP4 video creation."""

    job_id: str
    time_range: str | None = None
    downsample_factor: int = 20
    output_cluster: str | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Starting VCI MP4 video creation...")

        context.job_cache["job_id"] = self.job_id.encode()
        context.job_cache["downsample_factor"] = str(self.downsample_factor).encode()

        # Convert datetime range to time indices
        if self.time_range:
            start_str, end_str = self.time_range.split("/")
            start_datetime = datetime.fromisoformat(start_str)
            end_datetime = datetime.fromisoformat(end_str)

            start_year_dekad = (2000, 15)

            start_year = start_datetime.year
            start_dekad = (
                ((start_datetime.month - 1) * 3) + ((start_datetime.day - 1) // 10) + 1
            )
            start_dekad = max(1, min(36, start_dekad))

            end_year = end_datetime.year
            end_dekad = (
                ((end_datetime.month - 1) * 3) + ((end_datetime.day - 1) // 10) + 1
            )
            end_dekad = max(1, min(36, end_dekad))

            start_idx = _calc_time_index(start_year, start_dekad, *start_year_dekad)
            end_idx = _calc_time_index(end_year, end_dekad, *start_year_dekad) + 1

            context.job_cache["time_range"] = pickle.dumps((start_idx, end_idx))
            logger.info(
                f"Converted datetime range {self.time_range} to time indices {start_idx}-{end_idx - 1}"
            )
        else:
            logger.info("No time range specified, will process all available data")

        create_frames_task = context.submit_subtask(CreateVciFramesByYear())
        create_video_task = context.submit_subtask(
            CreateVideoFromFrames(), depends_on=[create_frames_task]
        )
        context.submit_subtask(
            DownloadVideoFromCache(),
            depends_on=[create_video_task],
            cluster=self.output_cluster,
        )


class CreateVciFramesByYear(Task):
    """Orchestrates frame creation by year to respect the 64 subtask limit."""

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Orchestrating VCI frame creation by year...")

        job_id = context.job_cache["job_id"].decode()

        zarr_prefix = f"{ZARR_STORE_PATH}/{job_id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=zarr_prefix,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        ds = xr.open_zarr(zarr_store, consolidated=False)
        vci_array = ds["vci"]

        if "time_range" in context.job_cache:
            start_idx, end_idx = pickle.loads(context.job_cache["time_range"])
        else:
            start_idx, end_idx = 0, vci_array.shape[0]

        start_year_dekad = (2000, 15)

        start_year, start_dekad = _calc_year_dekad_from_time_index(
            start_idx, *start_year_dekad
        )
        end_year, end_dekad = _calc_year_dekad_from_time_index(
            end_idx - 1, *start_year_dekad
        )

        logger.info(f"Processing years {start_year} to {end_year}")

        for year in range(start_year, end_year + 1):
            year_start_idx = _calc_time_index(year, 1, *start_year_dekad)
            year_end_idx = _calc_time_index(year + 1, 1, *start_year_dekad)

            actual_start = max(start_idx, year_start_idx)
            actual_end = min(end_idx, year_end_idx)

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
    """Creates frames for a specific year (up to 36 time periods)."""

    year: int
    time_start: int
    time_end: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(
            f"Creating VCI frames for year {self.year} (time indices {self.time_start} to {self.time_end})"
        )

        for time_idx in range(self.time_start, self.time_end):
            context.submit_subtask(CreateSingleVciFrame(time_index=time_idx))

        logger.info(
            f"Submitted {self.time_end - self.time_start} frame creation tasks for year {self.year}"
        )


class CreateSingleVciFrame(Task):
    """Creates a single VCI frame for a specific time index."""

    time_index: int

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Creating VCI frame for time index {self.time_index}")

        job_id = context.job_cache["job_id"].decode()
        downsample_factor = int(context.job_cache["downsample_factor"].decode())

        zarr_prefix = f"{ZARR_STORE_PATH}/{job_id}/cube.zarr"
        object_store = GCSStore(
            bucket=GCS_BUCKET,
            prefix=zarr_prefix,
            credential_provider=GoogleCredentialProvider(),
        )
        zarr_store = ZarrObjectStore(object_store)
        ds = xr.open_zarr(zarr_store, consolidated=False)

        vci_array = ds["vci"]

        if self.time_index >= vci_array.shape[0]:
            logger.warning(
                f"Time index {self.time_index} is out of bounds (max: {vci_array.shape[0] - 1})"
            )
            return

        vci_slice = vci_array[self.time_index, ::downsample_factor, ::downsample_factor]
        vci_data = vci_slice.compute()

        start_year_dekad = (2000, 15)

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

        nan_mask = np.isnan(vci_data)
        vci_for_colormap = np.where(nan_mask, 0.5, vci_data)
        vci_colors = colormaps["RdYlGn"](vci_for_colormap)

        img_array = (vci_colors * 255).astype(np.uint8)
        img_array[nan_mask, :3] = 255
        img_array[nan_mask, 3] = 255

        final_image = Image.fromarray(img_array)
        width, height = final_image.size

        try:
            logo = get_logo(scale=0.6)
            # paste the logo into the bottom right corner
            logo_padding = 80
            final_image.alpha_composite(
                logo, (logo_padding, height - logo.height - logo_padding)
            )
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Could not add logo: {e}")

        year, dekad = _calc_year_dekad_from_time_index(time_idx, *start_year_dekad)

        def _draw_centered_text(text: str, y_from_bottom: int, font: Any):
            draw = ImageDraw.Draw(final_image)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = width // 2 - text_width // 2
            text_y = height - text_height - y_from_bottom

            bg_padding = 5
            bg_bbox = [
                text_x - bg_padding,
                text_y - bg_padding,
                text_x + text_width + bg_padding,
                text_y + text_height + bg_padding,
            ]
            draw.rectangle(bg_bbox, fill=(255, 255, 255, 200))
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        _draw_centered_text(
            f"{year} D{dekad:02d}", y_from_bottom=60, font=get_font(40, "Bold")
        )
        _draw_centered_text(
            "Vegetation Condition Index",
            y_from_bottom=180,
            font=get_font(32, "Regular"),
        )
        _draw_centered_text(
            "derived from MODIS FPAR dataset",
            y_from_bottom=140,
            font=get_font(32, "Regular"),
        )

        frame_filename = f"vci_frame_{time_idx:04d}.png"

        final_image = final_image.convert("RGB")  # remove alpha channel

        from io import BytesIO

        img_buffer = BytesIO()
        final_image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        frames_group = context.job_cache.group("vci_frames")
        frame_key = f"frame_{time_idx:04d}"
        frames_group[frame_key] = img_bytes
        frames_group[f"filename_{time_idx:04d}"] = frame_filename.encode()

        return frame_key


class CreateVideoFromFrames(Task):
    """Creates an MP4 video from the individual frame images using ffmpeg."""

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Creating MP4 video from frames...")

        frame_paths = []

        if "time_range" in context.job_cache:
            start_idx, end_idx = pickle.loads(context.job_cache["time_range"])
        else:
            start_idx, end_idx = 0, 10000

        import tempfile

        temp_dir = tempfile.mkdtemp()
        frames_group = context.job_cache.group("vci_frames")

        frame_keys = [key for key in frames_group if key.startswith("frame_")]
        frame_keys.sort()

        logger.info(f"Found {len(frame_keys)} frames in cache")

        for frame_key in frame_keys:
            time_idx = int(frame_key.split("_")[1])

            if start_idx <= time_idx < end_idx:
                img_bytes = frames_group[frame_key]
                temp_filename = f"vci_frame_{time_idx:04d}.png"
                temp_path = os.path.join(temp_dir, temp_filename)

                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

                frame_paths.append(temp_path)

        if not frame_paths:
            logger.error("No frame paths found in cache")
            return

        logger.info(f"Found {len(frame_paths)} frames to combine into MP4")

        output_path = os.path.join(temp_dir, "vci_animation.mp4")
        frame_pattern = os.path.join(temp_dir, "vci_frame_*.png")

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "12",
                "-pattern_type",
                "glob",
                "-i",
                frame_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "medium",
                output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            logger.info(f"Successfully created MP4: {output_path}")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            video_cache_key = "vci_animation_mp4"
            context.job_cache[video_cache_key] = video_bytes

            logger.info(f"MP4 stored in cache with key: {video_cache_key}")
            logger.info(f"MP4 size: {len(video_bytes) / 1024 / 1024:.1f} MB")

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
            logger.error(f"Failed to create MP4: {e}")
            raise


class DownloadVideoFromCache(Task):
    """Downloads the MP4 video from the cache."""

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Downloading MP4 video from cache...")

        video_cache_key = "vci_animation_mp4"
        if video_cache_key in context.job_cache:
            video_bytes = context.job_cache[video_cache_key]
            with open("vci_animation.mp4", "wb") as f:
                f.write(video_bytes)
            logger.info("MP4 video downloaded successfully")
        else:
            logger.error("MP4 video not found in cache")
