import pickle
import subprocess
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import zarr
from humanize import naturalsize
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from obstore.auth.google import GoogleCredentialProvider
from obstore.store import GCSStore
from PIL import Image, ImageDraw, ImageFont
from tilebox.workflows import ExecutionContext, Task
from tilebox.workflows.observability.logging import get_logger

from vci_workflow.zarr import FILL_VALUE, GCS_BUCKET, open_zarr_group

logger = get_logger()

_COLOR_MAP = "RdYlGn"
_TEXT_COLOR = (0, 0, 0)  # black


def assets_store() -> GCSStore:
    return GCSStore(
        bucket=GCS_BUCKET,
        prefix="assets",
        credential_provider=GoogleCredentialProvider(),
    )


@lru_cache
def get_font(size: int = 24, style: str = "Regular") -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
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
def get_logo(scale: float = 0.7) -> Image.Image:
    """Get a font for text rendering with system-specific fallbacks."""
    local = Path("tilebox-logo.png")
    if not local.exists():
        with local.open("wb") as f:
            f.write(memoryview(assets_store().get(str(local)).bytes()))

    logo = Image.open(str(local))
    return logo.resize((int(logo.size[0] * scale), int(logo.size[1] * scale)))


class ZarrArrayToVideo(Task):
    """Main task to orchestrate VCI MP4 video creation."""

    "Path to a Zarr store containing the array to be visualized."
    video_zarr_path: str
    "Name of the array to be visualized, must be of shape (time, y, x)"
    video_array_name: str
    "Path to a Zarr store containing year and dekad coordinates, indexed by time, which we use to label the frames."
    coordinates_zarr_path: str
    "Spatial downsample and resize factors to apply to the image."
    downscale_factors: tuple[int, int]
    "Main title to render on every frame."
    title: str
    "Subtitle to render on every frame."
    subtitle: str

    def execute(self, context: ExecutionContext) -> None:
        logger.info(f"Starting {self.video_array_name} MP4 video creation...")

        group = open_zarr_group(self.video_zarr_path, "r")
        video_array = cast(zarr.Array, group[self.video_array_name])

        coords_group = open_zarr_group(self.coordinates_zarr_path)
        dekad_array = cast(zarr.Array, coords_group["dekad"])
        year_array = cast(zarr.Array, coords_group["year"])

        # Validate array shapes
        if year_array.shape[0] != dekad_array.shape[0]:
            raise ValueError(
                f"Year and dekad coordinates have different lengths: {year_array.shape[0]} != {dekad_array.shape[0]}"
            )
        if len(video_array.shape) != 3:
            raise ValueError(f"Expected video array to be of shape (time, y, x), but has shape {video_array.shape}")
        if year_array.shape[0] != video_array.shape[0]:
            raise ValueError(
                f"Coordinates and video array have different lengths: {year_array.shape[0]} != {video_array.shape[0]}"
            )

        # save title and subtitle to cache
        context.job_cache.group("videos").group(self.video_array_name)["title"] = pickle.dumps(
            (self.title, self.subtitle)
        )

        create_frames_task = context.submit_subtask(
            CreateFrames(
                self.video_zarr_path,
                self.video_array_name,
                self.coordinates_zarr_path,
                self.downscale_factors,
                (0, video_array.shape[0]),
            )
        )

        context.submit_subtask(
            CreateVideoFromFrames(self.video_array_name),
            depends_on=[create_frames_task],
        )


class CreateFrames(Task):
    """Creates frames for a specific time range."""

    "Path to a Zarr store containing the array to be visualized."
    video_zarr_path: str
    "Name of the array to be visualized, must be of shape (time, y, x)"
    video_array_name: str
    "Path to a Zarr store containing year and dekad coordinates, indexed by time, which we use to label the frames."
    coordinates_zarr_path: str
    "Spatial downsample factor to apply on the y and x dimensions before exporting image frames."
    downscale_factors: tuple[int, int]
    "Time index range to create frames for."
    frame_range: tuple[int, int]

    def execute(self, context: ExecutionContext) -> None:
        start = self.frame_range[0]
        end = self.frame_range[1]
        n_frames = end - start
        context.current_task.display = f"CreateFrames[{start}:{end}]"

        if n_frames > 8:
            middle = start + n_frames // 2
            context.submit_subtasks(
                [
                    CreateFrames(
                        self.video_zarr_path,
                        self.video_array_name,
                        self.coordinates_zarr_path,
                        self.downscale_factors,
                        half_range,
                    )
                    for half_range in [(start, middle), (middle, end)]
                ]
            )
            return

        for frame_index in range(start, end):
            context.submit_subtask(
                ExportFrame(
                    self.video_zarr_path,
                    self.video_array_name,
                    self.coordinates_zarr_path,
                    self.downscale_factors,
                    frame_index,
                )
            )


class ExportFrame(Task):
    """Creates a single VCI frame for a specific time index."""

    "Path to a Zarr store containing the array to be visualized."
    video_zarr_path: str
    "Name of the array to be visualized, must be of shape (time, y, x)"
    video_array_name: str
    "Path to a Zarr store containing year and dekad coordinates, indexed by time, which we use to label the frames."
    coordinates_zarr_path: str
    "Spatial downsample factor to apply on the y and x dimensions before exporting image frames."
    downscale_factors: tuple[int, int]
    "Time index range to create frames for."
    frame_index: int

    def execute(self, context: ExecutionContext) -> None:
        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001
        context.current_task.display = f"ExportFrame[{self.frame_index}]"
        logger.info(f"Creating {self.video_array_name} video frame (index {self.frame_index})")

        with tracer.start_span("read_frame_data"):
            group = open_zarr_group(self.video_zarr_path)
            video_arr = cast(zarr.Array, group[self.video_array_name])
            downsample, downsize = self.downscale_factors
            frame = video_arr[self.frame_index, ::downsample, ::downsample]
            frame = np.ma.masked_array(frame, mask=(frame == FILL_VALUE))

        # convert our array values to an image by applying a colormap
        map_to_colors = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=colormaps[_COLOR_MAP])
        image = Image.fromarray((map_to_colors.to_rgba(frame) * 255).astype(np.uint8))

        # instead of downsampling immediately to the target resolution, we first sample it a higher resolution and
        # then resize it to the target size using a resampling method. That produces a smoother visual image overall.
        target_width = image.width // downsize
        target_width -= target_width % 2  # our ffmpeg video codec requires even image dimensions
        target_height = image.height // downsize
        target_height -= target_height % 2  # our ffmpeg video codec requires even image dimensions
        image = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)

        # Paste the Tilebox logo into the bottom left corner
        logo = get_logo(scale=0.6)
        logo_padding = 80
        image.alpha_composite(logo, (logo_padding, image.height - logo.height - logo_padding))

        # Render a colormap gradient centered vertically at the left edge
        colorbar = _generate_colormap_image_vertical(_COLOR_MAP, width=60, height=256)
        colorbar_left_padding = 100
        image.paste(colorbar, (colorbar_left_padding, (image.height - colorbar.height) // 2))
        # add labels 0% and 100% to the colorbar
        draw = ImageDraw.Draw(image)
        colorbar_font = get_font(40, style="Medium")
        colorbar_text_padding = 26
        w, h = _text_size(draw, "100%", colorbar_font)
        draw.text(
            (
                colorbar_left_padding + colorbar.width // 2 - w // 2,
                image.height // 2 - colorbar.height // 2 - h - colorbar_text_padding,
            ),
            "100%",
            fill=_TEXT_COLOR,
            font=colorbar_font,
        )
        w, h = _text_size(draw, "0%", colorbar_font)
        draw.text(
            (
                colorbar_left_padding + colorbar.width // 2 - w // 2,
                image.height // 2 + colorbar.height // 2 + colorbar_text_padding // 2,
            ),
            "0%",
            fill=_TEXT_COLOR,
            font=colorbar_font,
        )

        # Render the frame title and subtitle
        title, subtitle = pickle.loads(context.job_cache.group("videos").group(self.video_array_name)["title"])  # noqa: S301
        y_from_bottom = 210
        title_font = get_font(48, style="Bold")
        w, h = _text_size(draw, title, title_font)
        draw.text(
            (image.width // 2 - w // 2, image.height - h - y_from_bottom), title, fill=_TEXT_COLOR, font=title_font
        )
        y_from_bottom -= h
        y_from_bottom -= 32  # padding between title and subtitle
        subtitle_font = get_font(32, style="Regular")
        w, h = _text_size(draw, subtitle, subtitle_font)
        draw.text(
            (image.width // 2 - w // 2, image.height - h - y_from_bottom),
            subtitle,
            fill=_TEXT_COLOR,
            font=subtitle_font,
        )

        # Render the year and dekad
        metadata_group = open_zarr_group(self.coordinates_zarr_path)
        dekad_array = cast(zarr.Array, metadata_group["dekad"])
        year_array = cast(zarr.Array, metadata_group["year"])
        dekad = int(dekad_array[self.frame_index])  # type: ignore[arg-type]
        year = int(year_array[self.frame_index])  # type: ignore[arg-type]

        y_from_bottom -= h
        y_from_bottom -= 32  # padding between subtitle and year/dekad line
        year_dekad_text = f"{year} D{dekad:02d}"
        year_dekad_font = get_font(40, style="Bold")
        w, h = _text_size(draw, year_dekad_text, year_dekad_font)
        draw.text(
            (image.width // 2 - w // 2, image.height - h - y_from_bottom),
            year_dekad_text,
            fill=_TEXT_COLOR,
            font=year_dekad_font,
        )

        image = convert_rgba_to_rgb(image, background_color=(255, 255, 255))  # convert to RGB with white background
        # save image as PNG
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        # and upload it to the cache
        frame_key = f"frame_{self.video_array_name}_{self.frame_index:04d}.png"
        frames_cache = context.job_cache.group("videos").group(self.video_array_name).group("frames")
        frames_cache[frame_key] = img_bytes
        logger.info(f"{self.video_array_name} frame saved to cache: {frame_key}")


class CreateVideoFromFrames(Task):
    """Creates an MP4 video from the individual frame images using ffmpeg."""

    video_array_name: str

    def execute(self, context: ExecutionContext) -> None:
        tracer = context._runner.tracer._tracer  # type: ignore[arg-defined], # noqa: SLF001
        logger.info("Converting frames to MP4...")

        with TemporaryDirectory(prefix="video_frames") as temp_dir_path:
            frames_dir = Path(temp_dir_path)
            frames_cache = context.job_cache.group("videos").group(self.video_array_name).group("frames")

            with tracer.start_span("download_load_frames"):
                n_frames = 0
                for key in frames_cache:
                    if not key.startswith("frame_"):
                        continue

                    # save frame as file to our temporary directory
                    (frames_dir / key).write_bytes(frames_cache[key])
                    n_frames += 1

            logger.info(f"Found {n_frames} frames to combine into MP4")

            output_path = frames_dir / f"{self.video_array_name}.mp4"
            frame_pattern = frames_dir / f"frame_{self.video_array_name}_*.png"

            with tracer.start_span("run_ffmpeg"):
                try:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        "12",
                        "-pattern_type",
                        "glob",
                        "-i",
                        str(frame_pattern),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "18",
                        "-preset",
                        "medium",
                        str(output_path),
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603

                    logger.info(f"Successfully created MP4: {output_path}")

                except subprocess.CalledProcessError as e:
                    logger.error(f"ffmpeg failed: {e}")
                    logger.error(f"stderr: {e.stderr.decode() if e.stderr else 'No stderr'}")
                    raise

            with tracer.start_span("upload_video_to_cache"):
                video_bytes = output_path.read_bytes()
                context.job_cache.group("videos").group(self.video_array_name)[output_path.name] = video_bytes

                logger.info(f"MP4 saved to cache: videos/{output_path.name}. Size {naturalsize(len(video_bytes))}")
                logger.info("Video creation complete")


def _generate_colormap_image_vertical(
    colormap_name: str,
    width: int,
    height: int,
    border_width: int = 1,
    border_color: tuple[int, ...] = (30, 30, 30, 255),  # dark gray
) -> Image.Image:
    """
    Generate a rectangular image with a vertical color gradient based on the given colormap.

    The top of the image corresponds to the maximum value of the colormap,
    and the bottom corresponds to the minimum value, with a gradient in between.

    The image has a border of the specified width and color.

    Returns:
        The colormap rendered as a vertical gradient in the given size.
    """
    gradient_width = width - border_width * 2
    gradient_height = height - border_width * 2

    color_gradient = np.broadcast_to(
        np.linspace(1, 0, num=gradient_height).reshape(gradient_height, 1), shape=(gradient_height, gradient_width)
    )
    gradient = (colormaps[colormap_name](color_gradient) * 255).astype(np.uint8)

    image = np.zeros(shape=(height, width, 4), dtype=np.uint8)
    image[:, :, :] = border_color

    image[border_width:-border_width, border_width:-border_width, :] = gradient
    return Image.fromarray(image)


def _text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont
) -> tuple[int, int]:
    """Compute the width and height (in pixels) of a text when rendered with a given font."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = int(bbox[2] - bbox[0])
    text_height = int(bbox[3] - bbox[1])
    return text_width, text_height


def convert_rgba_to_rgb(rgba_image: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    """
    Converts an RGBA image to RGB with a given background color.
    """
    # Create a new image with the given background color
    rgb_image = Image.new("RGB", rgba_image.size, background_color)
    # Paste the RGBA image onto the white background
    # The RGBA image's alpha channel is used as the mask
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])

    return rgb_image
