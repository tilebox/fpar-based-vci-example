import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib import colormaps
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as ImageFile
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger

from vci_workflow.zarr import GCS_BUCKET, open_zarr_group


def assets_store() -> GCSStore:
    return GCSStore(
        bucket=GCS_BUCKET,
        prefix="assets",
        credential_provider=GoogleCredentialProvider(),
    )


@lru_cache
def get_font(size: int = 24, style: str = "Regular") -> Any:
    """Get a font for text rendering with system-specific fallbacks."""
    local = Path(f"Poppins-{style}.ttf")
    if not local.exists():
        with local.open("wb") as f:
            f.write(memoryview(assets_store().get(str(local)).bytes()))

    try:
        return ImageFont.truetype(str(local.absolute()), size)
    except OSError:
        return ImageFont.load_default(size)


@lru_cache
def get_logo(scale: float = 0.7) -> Any:
    """Get a font for text rendering with system-specific fallbacks."""
    local = Path("tilebox-logo.png")
    if not local.exists():
        with local.open("wb") as f:
            f.write(memoryview(assets_store().get(str(local)).bytes()))

    logo: ImageFile = Image.open(str(local))
    return logo.resize((int(logo.size[0] * scale), int(logo.size[1] * scale)))


class CreateVciMp4(Task):
    """Main task to orchestrate VCI MP4 video creation."""

    vci_zarr_path: str
    fpar_zarr_path: str
    downsample_factor: int | None = None
    output_cluster: str | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Starting VCI MP4 video creation...")

        logger.info("Start VCI MP4 creation")

        # Open zarr groups using helper
        vci_group = open_zarr_group(self.vci_zarr_path)
        vci_array = vci_group["vci"]  # type: ignore[index]

        fpar_group = open_zarr_group(self.fpar_zarr_path)
        dekad_array = fpar_group["dekad"]  # type: ignore[index]
        year_array = fpar_group["year"]  # type: ignore[index]

        # Validate array shapes
        if year_array.shape[0] != dekad_array.shape[0]:  # type: ignore[union-attr]
            raise ValueError(
                f"Year and dekad arrays have different lengths: {year_array.shape[0]} != {dekad_array.shape[0]}"  # type: ignore[union-attr]
            )
        if year_array.shape[0] != vci_array.shape[0]:  # type: ignore[union-attr]
            raise ValueError(
                f"Year array and VCI array have different lengths: {year_array.shape[0]} != {vci_array.shape[0]}"  # type: ignore[union-attr]
            )

        create_frames_task = context.submit_subtask(
            CreateVciFrames(
                slice=(0, vci_array.shape[0]),  # type: ignore[union-attr]
                vci_zarr_path=self.vci_zarr_path,
                fpar_zarr_path=self.fpar_zarr_path,
            )
        )

        context.submit_subtask(
            CreateVideoFromFrames(slice=(0, vci_array.shape[0])),  # type: ignore[union-attr]
            depends_on=[create_frames_task],
        )


class CreateVciFrames(Task):
    """Creates frames for a specific time range."""

    vci_zarr_path: str
    fpar_zarr_path: str
    slice: tuple[int, int]
    downsample_factor: int | None = None

    def execute(self, context: ExecutionContext) -> None:
        start = self.slice[0]
        end = self.slice[1]
        if end - start > 8:
            middle = start + (end - start) // 2
            context.submit_subtask(
                CreateVciFrames(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    slice=(start, middle),
                )
            )
            context.submit_subtask(
                CreateVciFrames(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    downsample_factor=self.downsample_factor,
                    slice=(middle, end),
                )
            )
            return

        for time_idx in range(start, end):
            context.submit_subtask(
                CreateSingleVciFrame(
                    vci_zarr_path=self.vci_zarr_path,
                    fpar_zarr_path=self.fpar_zarr_path,
                    downsample_factor=self.downsample_factor,
                    time_index=time_idx,
                )
            )


class CreateSingleVciFrame(Task):
    """Creates a single VCI frame for a specific time index."""

    time_index: int
    vci_zarr_path: str
    fpar_zarr_path: str
    downsample_factor: int | None = None

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info(f"Creating VCI frame for time index {self.time_index}")

        logger.info(f"Frame {self.time_index} start")

        # Open VCI zarr group using helper and load data with xarray
        vci_group = open_zarr_group(self.vci_zarr_path)
        ds = xr.open_zarr(vci_group.store, consolidated=False)
        vci_array = ds["vci"]

        vci_slice = vci_array[self.time_index, :: self.downsample_factor, :: self.downsample_factor]
        vci_data = vci_slice.compute()

        # Open FPAR zarr group using helper
        fpar_group = open_zarr_group(self.fpar_zarr_path)
        dekad_array = fpar_group["dekad"]  # type: ignore[index]
        year_array = fpar_group["year"]  # type: ignore[index]

        start_year_dekad = (year_array[0].item(), dekad_array[0].item())  # type: ignore[union-attr]

        self._create_frame_image(vci_data, self.time_index, start_year_dekad, context)

        logger.info(f"Frame {self.time_index} complete")
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
            final_image.alpha_composite(logo, (logo_padding, height - logo.height - logo_padding))
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

        _draw_centered_text(f"{year} D{dekad:02d}", y_from_bottom=60, font=get_font(40, "Bold"))
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
        final_image = final_image.convert("RGB")

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

    slice: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        logger = get_logger()
        logger.info("Creating MP4 video from frames...")

        logger.info("Video creation start")

        start_idx, end_idx = self.slice
        temp_dir = tempfile.mkdtemp()
        frames_group = context.job_cache.group("vci_frames")

        frame_keys = [key for key in frames_group if key.startswith("frame_")]
        frame_keys.sort()

        logger.info(f"Found {len(frame_keys)} frames in cache")

        frame_paths = []
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

            logger.info("Video creation complete")

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
