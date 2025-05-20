# utils.py

"""
Utility functions for the Discord bot.

This module provides helper functions for file operations, image manipulation,
data conversion, and other common tasks.
"""

import base64
import imghdr  # For determining image type from bytes
import io
import logging
import mimetypes  # For guessing MIME type from file extension
from datetime import datetime
from pathlib import Path

import cv2
import discord
import httpx
import numpy as np
from moviepy.editor import AudioFileClip, ImageSequenceClip
from PIL import Image
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d

logger: logging.Logger = logging.getLogger(__name__)

TEMP_FILE_DIR: Path = Path(".cache/temp_utils_files")
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)


def create_temp_file(content: str, suffix: str = ".txt") -> Path:
    """
    Creates a temporary file with the given content in a dedicated cache directory.

    The filename includes a timestamp to ensure uniqueness.

    Args:
        content: The string content to write to the file.
        suffix: The desired file suffix (e.g., ".txt", ".md").

    Returns:
        A Path object pointing to the created temporary file.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S%f")
    # Ensure suffix starts with a dot
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    # Sanitize a potential prefix if needed, or use a generic one
    # For now, using a generic prefix "response_"
    file_path: Path = TEMP_FILE_DIR / f"response_{timestamp}{suffix}"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Temporary file created: {file_path}")
    except OSError as e:
        logger.error(f"Failed to create temporary file {file_path}: {e}")
        raise  # Re-raise the exception after logging
    return file_path


def image_to_base64_url(image_path: Path) -> str:
    """
    Converts an image file to a base64 data URL.

    Args:
        image_path: The path to the image file.

    Returns:
        A string representing the base64 data URL (e.g., "data:image/png;base64,...").

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the MIME type cannot be determined.
    """
    if not image_path.exists():
        logger.error(f"Image file not found at {image_path} for base64 conversion.")
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    mime_type: str | None
    mime_type, _ = mimetypes.guess_type(image_path.name)  # Use name for mimetypes
    if mime_type is None:
        # Fallback to imghdr if mimetypes fails (e.g. no extension)
        try:
            image_format: str | None = imghdr.what(image_path)
            if image_format:
                mime_type = f"image/{image_format}"
            else:
                logger.warning(
                    f"Could not determine MIME type for file {image_path} using mimetypes or imghdr."
                )
                raise ValueError(f"Could not determine MIME type for file {image_path}")
        except Exception as e:
            logger.error(
                f"Error determining image type with imghdr for {image_path}: {e}"
            )
            raise ValueError(
                f"Could not determine MIME type for file {image_path}: {e}"
            )

    try:
        with image_path.open("rb") as img_file:
            image_data: bytes = img_file.read()
    except OSError as e:
        logger.error(f"Could not read image file {image_path}: {e}")
        raise

    base64_encoded: str = base64.b64encode(image_data).decode("utf-8")
    base64_url: str = f"data:{mime_type};base64,{base64_encoded}"
    logger.info(f"Successfully converted image {image_path} to base64 data URL.")
    return base64_url


def convert_base64_to_discord_file(
    base64_image_data: str, filename: str = "image.png"
) -> discord.File:
    """
    Converts a base64 encoded image string (without data URL prefix) to a discord.File object.

    Args:
        base64_image_data: The base64 encoded image string.
        filename: The desired filename for the discord.File object.

    Returns:
        A discord.File object ready for sending.
    """
    try:
        image_bytes: bytes = base64.b64decode(base64_image_data)
        image_stream: io.BytesIO = io.BytesIO(image_bytes)
        logger.info(f"Converted base64 data to discord.File with filename: {filename}")
        return discord.File(fp=image_stream, filename=filename)
    except base64.binascii.Error as e:
        logger.error(f"Invalid base64 data provided for discord.File conversion: {e}")
        raise ValueError(f"Invalid base64 data: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error converting base64 to discord.File: {e}")
        raise


def convert_pil_image_to_discord_file(
    pil_image: Image.Image, filename: str = "image.png", image_format: str = "PNG"
) -> discord.File:
    """
    Converts a PIL Image object to a discord.File object.

    Args:
        pil_image: The PIL.Image.Image object.
        filename: The desired filename for the discord.File object.
        image_format: The format to save the image in (e.g., "PNG", "JPEG").

    Returns:
        A discord.File object ready for sending.
    """
    try:
        image_stream: io.BytesIO = io.BytesIO()
        pil_image.save(image_stream, format=image_format)
        image_stream.seek(0)  # Reset stream position to the beginning
        logger.info(
            f"Converted PIL Image to discord.File with filename: {filename}, format: {image_format}"
        )
        return discord.File(fp=image_stream, filename=filename)
    except Exception as e:
        logger.exception(f"Unexpected error converting PIL Image to discord.File: {e}")
        raise


# The original `convert_base64_images_to_discord_files` and `...gifs...`
# created temporary files. The new `convert_base64_to_discord_file` is more generic
# and uses BytesIO, which is preferred. If specific handling for lists of base64
# strings is still needed, it can be built using the new helper.
# For now, these are removed to promote the BytesIO approach.


def download_image(image_url: str, save_dir: Path = TEMP_FILE_DIR) -> Path:
    """
    Downloads an image from a URL and saves it to a specified directory.
    The filename is derived from the URL, and the extension is determined by image content.

    Args:
        image_url: The URL of the image to download.
        save_dir: The directory where the image will be saved. Defaults to TEMP_FILE_DIR.

    Returns:
        The Path object of the saved image file.

    Raises:
        httpx.HTTPStatusError: If the download fails.
        ValueError: If the image type cannot be determined or URL is invalid.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate a somewhat unique filename from the URL or use a timestamp
    try:
        url_path = Path(httpx.URL(image_url).path)
        base_filename = (
            url_path.stem
            if url_path.stem
            else f"downloaded_image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )
    except httpx.InvalidURL:
        logger.error(f"Invalid image URL provided for download: {image_url}")
        raise ValueError(f"Invalid image URL: {image_url}")

    logger.info(f"Downloading image from URL: {image_url}")
    try:
        with httpx.stream(
            "GET", image_url, follow_redirects=True, timeout=30.0
        ) as response:
            response.raise_for_status()
            image_content: bytearray = bytearray()
            for chunk in response.iter_bytes():
                image_content.extend(chunk)

            image_bytes: bytes = bytes(image_content)

        image_type: str | None = imghdr.what(None, h=image_bytes)
        if image_type is None:
            # Fallback: Check Content-Type header if imghdr fails
            content_type_header = response.headers.get("Content-Type", "")
            if content_type_header.startswith("image/"):
                image_type = (
                    content_type_header.split("/")[1].split(";")[0].strip()
                )  # e.g. png from image/png; charset=UTF-8
            else:
                logger.warning(
                    f"Unable to determine image file type for URL: {image_url}. Header: {content_type_header}"
                )
                raise ValueError(
                    f"Unable to determine image file type for URL: {image_url}"
                )

        # Sanitize image_type if it contains characters not suitable for extension
        image_type = "".join(c for c in image_type if c.isalnum())
        save_path: Path = save_dir / f"{base_filename}.{image_type}"

        with open(save_path, "wb") as file:
            file.write(image_bytes)
        logger.info(f"Image downloaded from {image_url} and saved as {save_path}")
        return save_path
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error downloading image {image_url}: {e.response.status_code}"
        )
        raise
    except Exception as e:
        logger.exception(f"Error downloading or saving image from {image_url}: {e}")
        raise


async def download_image_as_b64_data_url(image_url: str) -> str:
    """
    Downloads an image from a URL and returns it as a base64 data URL.
    Determines MIME type from image content or HTTP headers.

    Args:
        image_url: The URL of the image to download.

    Returns:
        A base64 data URL string (e.g., "data:image/png;base64,...").

    Raises:
        httpx.HTTPStatusError: If the download fails.
        ValueError: If image type cannot be determined or URL is invalid.
    """
    logger.info(f"Downloading image for base64 data URL conversion from: {image_url}")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes: bytes = await response.aread()

        # Determine MIME type
        mime_type: str | None = None
        image_format: str | None = imghdr.what(None, h=image_bytes)
        if image_format:
            mime_type = f"image/{image_format}"
        else:
            # Fallback to Content-Type header
            content_type_header = response.headers.get("Content-Type", "")
            if content_type_header.startswith("image/"):
                mime_type = content_type_header.split(";")[
                    0
                ].strip()  # Get "image/png" from "image/png; charset=UTF-8"
            else:
                logger.warning(
                    f"Could not determine MIME type for image from {image_url}. Header: {content_type_header}"
                )
                raise ValueError(
                    f"Could not determine MIME type for image from {image_url}"
                )

        image_base64: str = base64.b64encode(image_bytes).decode("utf-8")
        data_url: str = f"data:{mime_type};base64,{image_base64}"
        logger.info(
            f"Successfully converted image from {image_url} to base64 data URL ({mime_type})."
        )
        return data_url
    except httpx.InvalidURL:
        logger.error(f"Invalid image URL provided for b64 download: {image_url}")
        raise ValueError(f"Invalid image URL: {image_url}")
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error downloading image {image_url} for b64: {e.response.status_code}"
        )
        raise
    except Exception as e:
        logger.exception(f"Error downloading image {image_url} for b64 conversion: {e}")
        raise


def markdown_header(title: str, content: str) -> str:
    """
    Formats a title and content into a Markdown header block.

    Args:
        title: The title for the header (will be bolded).
        content: The content to be placed inside the Markdown code block.

    Returns:
        A string formatted as a Markdown header with a code block.
    """
    return f"**{title}**\n```md\n{content}\n```"


def create_mask_with_alpha(mask_path: Path, output_dir: Path = TEMP_FILE_DIR) -> Path:
    """
    Creates a mask with an alpha channel from a black and white image.
    The input mask should have white for areas to keep and black for areas to make transparent.

    Args:
        mask_path: Path to the black & white mask image (e.g., PNG, JPEG).
        output_dir: Directory to save the new mask with alpha.

    Returns:
        Path to the new mask image (PNG format) with an alpha channel.

    Raises:
        FileNotFoundError: If mask_path does not exist.
        IOError: If there's an issue reading or writing image files.
        ValueError: If the input image cannot be processed by PIL.
    """
    if not mask_path.exists():
        logger.error(f"Mask file not found at: {mask_path}")
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_mask_path: Path = output_dir / f"{mask_path.stem}_alpha.png"

    try:
        # Load the black & white mask as a grayscale image
        with Image.open(mask_path) as mask:
            grayscale_mask: Image.Image = mask.convert("L")

            # Convert it to RGBA so it has space for an alpha channel
            mask_rgba: Image.Image = grayscale_mask.convert("RGBA")

            # Use the grayscale mask itself to fill the alpha channel.
            # White (255) in grayscale_mask becomes fully opaque in alpha.
            # Black (0) in grayscale_mask becomes fully transparent in alpha.
            mask_rgba.putalpha(grayscale_mask)

            # Save the resulting file
            mask_rgba.save(alpha_mask_path, format="PNG")
        logger.info(f"Created mask with alpha channel at: {alpha_mask_path}")
        return alpha_mask_path
    except (
        FileNotFoundError
    ):  # Should be caught by the initial check, but good practice
        logger.error(
            f"Error processing mask: Input file {mask_path} not found during PIL operations."
        )
        raise
    except OSError as e:
        logger.error(
            f"IOError processing mask {mask_path} or saving to {alpha_mask_path}: {e}"
        )
        raise
    except Exception as e:  # Catch other PIL errors
        logger.exception(f"Unexpected error creating alpha mask from {mask_path}: {e}")
        raise ValueError(f"Could not process image {mask_path} with PIL: {e}")


def convert_audio_to_waveform_video(
    audio_file: str | Path, video_file: str | Path
) -> Path:
    """
    Converts an audio file (MP3 or WAV) into a waveform video (MP4).

    Args:
        audio_file: Path to the input audio file.
        video_file: Path for the output video file.

    Returns:
        Path to the generated video file.

    Raises:
        ValueError: If audio/video formats are incorrect.
        FileNotFoundError: If the audio file does not exist.
        Exception: For errors during audio processing or video writing.
    """
    audio_file_path: Path = Path(audio_file)
    video_file_path: Path = Path(video_file)

    logger.info(
        f"Starting waveform video conversion: {audio_file_path} -> {video_file_path}"
    )

    if audio_file_path.suffix not in [".mp3", ".wav"]:
        msg = f"Audio file must be in MP3 or WAV format. Got: {audio_file_path}"
        logger.error(msg)
        raise ValueError(msg)
    if video_file_path.suffix != ".mp4":
        msg = f"Output video file must be in MP4 format. Got: {video_file_path}"
        logger.error(msg)
        raise ValueError(msg)
    if not audio_file_path.exists():
        msg = f"Audio file not found: {audio_file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        # Ensure output directory exists
        video_file_path.parent.mkdir(parents=True, exist_ok=True)

        audio: AudioSegment = AudioSegment.from_file(audio_file_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)  # Convert to mono

        samples: np.ndarray = np.array(audio.get_array_of_samples())
        sample_rate: int = audio.frame_rate
        duration_seconds: float = len(samples) / sample_rate

        video_fps: int = 30
        frame_count: int = int(video_fps * duration_seconds)
        if frame_count == 0:  # Handle very short audio clips
            logger.warning(
                f"Audio duration too short for {video_fps} FPS, resulting in 0 frames. Min duration needed."
            )
            # Create a minimal 1-frame video or raise error
            frame_count = 1  # Ensure at least one frame

        height, width = 120, 480
        line_color: tuple[int, int, int] = (118, 168, 91)  # RGB
        background_color: tuple[int, int, int] = (34, 34, 34)  # RGB

        def generate_video_frame(
            current_samples: np.ndarray,
            is_first_or_last: bool,
            volume_threshold_factor: float = 0.01,  # Relative to max possible sample value
        ) -> np.ndarray:
            """Generates a single video frame for the waveform."""
            if (
                is_first_or_last or len(current_samples) == 0
            ):  # Blank frame for start/end or no samples
                return np.full((height, width, 3), background_color, dtype=np.uint8)

            frame: np.ndarray = np.full(
                (height, width, 3), background_color, dtype=np.uint8
            )

            # Normalize samples to fit frame height with padding
            padding_factor: float = 0.1
            y_normalized: np.ndarray

            # Use a consistent max value for normalization to avoid jumpy amplitudes
            # (e.g. max value of a 16-bit sample if that's what pydub provides)
            # For simplicity, normalizing based on current segment's min/max for now.
            # A more stable visualization might normalize against global min/max or a fixed range.
            y_min_val, y_max_val = current_samples.min(), current_samples.max()
            y_range = y_max_val - y_min_val
            if y_range == 0:
                y_normalized = np.full_like(
                    current_samples, height / 2.0
                )  # Centered line if silent
            else:
                y_normalized = ((current_samples - y_min_val) / y_range) * (
                    height * (1 - 2 * padding_factor)
                ) + (height * padding_factor)

            y_smoothed: np.ndarray = gaussian_filter1d(
                y_normalized, sigma=2, mode="nearest"
            )

            # Calculate x coordinates for plotting
            x_coords: np.ndarray = np.linspace(0, width - 1, len(y_smoothed)).astype(
                np.int32
            )

            # Determine volume threshold based on the max possible value of samples (e.g., for 16-bit audio)
            # This threshold helps in not drawing lines for very low noise.
            # Assuming samples are in a typical range (e.g. -32768 to 32767 for 16-bit)
            # A fixed threshold might be better than one relative to current segment's max.
            # For now, let's use a simple heuristic: don't draw if amplitude is too low.
            # This threshold is relative to the *normalized* height.
            effective_volume_threshold = height * padding_factor + (
                height * (1 - 2 * padding_factor) * volume_threshold_factor
            )

            for i in range(len(x_coords) - 1):
                # Only draw if the point is above a certain "silence" threshold
                if (
                    y_smoothed[i] > effective_volume_threshold
                    or y_smoothed[i + 1] > effective_volume_threshold
                ):
                    pt1: tuple[int, int] = (x_coords[i], int(y_smoothed[i]))
                    pt2: tuple[int, int] = (x_coords[i + 1], int(y_smoothed[i + 1]))
                    cv2.line(frame, pt1, pt2, line_color, thickness=2)
            return frame

        frames: list[np.ndarray] = []
        samples_per_frame: int = max(1, len(samples) // frame_count)

        for i in range(frame_count):
            start_idx: int = i * samples_per_frame
            end_idx: int = (i + 1) * samples_per_frame
            current_frame_samples: np.ndarray = samples[start_idx:end_idx]

            is_boundary_frame: bool = i == 0 or i == frame_count - 1
            frame_image: np.ndarray = generate_video_frame(
                current_frame_samples, is_boundary_frame
            )
            frames.append(frame_image)

        if not frames:  # Should not happen if frame_count >= 1
            logger.error(
                "No frames generated for video. Audio might be too short or empty."
            )
            raise RuntimeError("Video frame generation failed.")

        moviepy_audio_clip: AudioFileClip = AudioFileClip(str(audio_file_path))
        # Ensure video clip duration matches audio clip duration
        video_clip: ImageSequenceClip = ImageSequenceClip(
            frames, fps=video_fps
        ).set_duration(moviepy_audio_clip.duration)

        final_clip = video_clip.set_audio(moviepy_audio_clip)

        # Use a specific logger for moviepy to control its verbosity if needed
        moviepy_logger_level = "ERROR"  # Suppress INFO/DEBUG from moviepy
        final_clip.write_videofile(
            str(video_file_path),
            codec="libx264",
            audio_codec="aac",
            fps=video_fps,
            preset="ultrafast",  # Faster encoding
            logger=moviepy_logger_level,
        )
        logger.info(f"Waveform video successfully created: {video_file_path}")
        return video_file_path

    except Exception as e:
        logger.exception(f"Error converting audio {audio_file_path} to video: {e}")
        raise
