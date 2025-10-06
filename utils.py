# utils.py

"""
Utility functions for the Discord bot.

This module provides helper functions for file operations, image manipulation,
data conversion, and other common tasks.
"""

import base64
import binascii  # Needed for binascii.Error in base64 exception handling
import imghdr  # For determining image type from bytes
import io
import logging
import mimetypes  # For guessing MIME type from file extension
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import discord
import httpx
import numpy as np

# Suppress SyntaxWarnings from moviepy and pydub libraries (invalid escape sequences)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="moviepy.*")
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub.*")
    from moviepy.editor import AudioFileClip, ImageSequenceClip
    from pydub import AudioSegment

from PIL import Image
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
        return file_path
    except OSError:
        logger.exception(f"Failed to create temporary file {file_path}")
        raise  # Re-raise the exception after logging


# Custom exception classes for better error handling
class ImageFileNotFoundError(FileNotFoundError):
    """Raised when an image file cannot be found."""


class MimeTypeError(ValueError):
    """Raised when MIME type cannot be determined."""


class Base64ConversionError(ValueError):
    """Raised when base64 data is invalid."""


def image_to_base64_url(image_path: Path) -> str:
    """
    Converts an image file to a base64 data URL.

    Args:
        image_path: The path to the image file.

    Returns:
        A string representing the base64 data URL (e.g., "data:image/png;base64,...").

    Raises:
        ImageFileNotFoundError: If the image_path does not exist.
        MimeTypeError: If the MIME type cannot be determined.
    """
    if not image_path.exists():
        logger.error(f"Image file not found at {image_path} for base64 conversion.")
        raise ImageFileNotFoundError("Image not found")

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
                raise MimeTypeError("Could not determine MIME type")
        except Exception:
            logger.exception(
                f"Error determining image type with imghdr for {image_path}"
            )
            raise MimeTypeError("Failed to process image format")

    try:
        with image_path.open("rb") as img_file:
            image_data: bytes = img_file.read()
    except OSError:
        logger.exception(f"Could not read image file {image_path}")
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

    Raises:
        Base64ConversionError: If the base64 data is invalid.
    """
    try:
        image_bytes: bytes = base64.b64decode(base64_image_data)
        image_stream: io.BytesIO = io.BytesIO(image_bytes)
        logger.info(f"Converted base64 data to discord.File with filename: {filename}")
        return discord.File(fp=image_stream, filename=filename)
    except binascii.Error:
        logger.exception("Invalid base64 data provided for discord.File conversion")
        raise Base64ConversionError("Invalid base64 data")
    except Exception:
        logger.exception("Unexpected error converting base64 to discord.File")
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
    except Exception:
        logger.exception("Unexpected error converting PIL Image to discord.File")
        raise


class DownloadError(Exception):
    """Base exception for image download failures."""


class InvalidURLError(DownloadError):
    """Raised when an image URL is invalid."""


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
        DownloadError: If the download fails for any reason.
        InvalidURLError: If the URL is invalid.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate a somewhat unique filename from the URL or use a timestamp
    try:
        url_path = Path(httpx.URL(image_url).path)
        base_filename = (
            url_path.stem
            if url_path.stem
            else f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    except httpx.InvalidURL:
        logger.warning(f"Invalid URL format for base filename: {image_url}")
        base_filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        with httpx.Client() as client:
            response = client.get(image_url, follow_redirects=True)
            response.raise_for_status()  # Raise HTTPStatusError for non-2xx
            image_bytes = response.content

            # Try to determine the content type from the response headers
            content_type_header = response.headers.get("content-type", "")
            detected_type = ""

            if "image/" in content_type_header:
                detected_type = content_type_header.split("image/")[1].split(";")[0]
            else:
                # Try to determine from the image content
                detected_type_from_content: str | None = imghdr.what(
                    None, h=image_bytes
                )
                detected_type = (
                    detected_type_from_content if detected_type_from_content else ""
                )
                if not detected_type:
                    logger.warning(
                        f"Unable to determine image file type for URL: {image_url}. Header: {content_type_header}"
                    )
                    raise MimeTypeError(
                        f"Unable to determine image file type for URL: {image_url}"
                    )

            # Sanitize detected_type if it contains characters not suitable for extension
            sanitized_type = "".join(c for c in detected_type if c.isalnum())
            save_path: Path = save_dir / f"{base_filename}.{sanitized_type}"

            with open(save_path, "wb") as file:
                file.write(image_bytes)
            logger.info(f"Image downloaded from {image_url} and saved as {save_path}")
            return save_path
    except httpx.HTTPStatusError as e:
        logger.exception(
            f"HTTP error downloading image {image_url}: {e.response.status_code}"
        )
        raise DownloadError(f"HTTP error: {e.response.status_code}")
    except httpx.InvalidURL:
        logger.exception(f"Invalid URL provided: {image_url}")
        raise InvalidURLError(f"Invalid URL: {image_url}")
    except Exception as e:
        logger.exception(f"Error downloading or saving image from {image_url}")
        raise DownloadError(f"Download failed: {e!s}")


async def download_image_async(image_url: str, save_dir: Path = TEMP_FILE_DIR) -> Path:
    """
    Async version of download_image.
    Downloads an image from a URL and saves it to a specified directory.

    Args:
        image_url: The URL of the image to download.
        save_dir: The directory where the image will be saved.

    Returns:
        The Path object of the saved image file.

    Raises:
        DownloadError: If the download fails for any reason.
        InvalidURLError: If the URL is invalid.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate a somewhat unique filename from the URL or use a timestamp
    try:
        url_path = Path(httpx.URL(image_url).path)
        base_filename = (
            url_path.stem
            if url_path.stem
            else f"image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )
    except httpx.InvalidURL:
        logger.warning(f"Invalid URL format for base filename: {image_url}")
        base_filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True)
            response.raise_for_status()
            image_bytes = response.content

            # Try to determine the content type from the response headers
            content_type_header = response.headers.get("content-type", "")
            detected_type = ""

            if "image/" in content_type_header:
                detected_type = content_type_header.split("image/")[1].split(";")[0]
            else:
                # Try to determine from the image content
                detected_type_from_content = imghdr.what(None, h=image_bytes)
                detected_type = detected_type_from_content if detected_type_from_content else ""
                if not detected_type:
                    logger.warning(f"Unable to determine image file type for URL: {image_url}")
                    # Default to jpg if we can't determine type
                    detected_type = "jpg"

            # Sanitize detected_type if it contains characters not suitable for extension
            sanitized_type = "".join(c for c in detected_type if c.isalnum())
            save_path = save_dir / f"{base_filename}.{sanitized_type}"

            with open(save_path, "wb") as file:
                file.write(image_bytes)
            logger.info(f"Image downloaded from {image_url} and saved as {save_path}")
            return save_path
    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error downloading image {image_url}: {e.response.status_code}")
        raise DownloadError(f"HTTP error: {e.response.status_code}")
    except httpx.InvalidURL:
        logger.exception(f"Invalid URL provided: {image_url}")
        raise InvalidURLError(f"Invalid URL: {image_url}")
    except Exception as e:
        logger.exception(f"Error downloading or saving image from {image_url}")
        raise DownloadError(f"Download failed: {e!s}")


async def download_image_as_b64_data_url(image_url: str) -> str:
    """
    Downloads an image from a URL and returns it as a base64 data URL.
    Determines MIME type from image content or HTTP headers.

    Args:
        image_url: The URL of the image to download.

    Returns:
        A base64 data URL string (e.g., "data:image/png;base64,...").

    Raises:
        InvalidURLError: If the URL is invalid.
        DownloadError: If the download fails.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True)
            response.raise_for_status()
            image_bytes = response.content

            # Determine content type from headers or content
            content_type = response.headers.get("content-type", "")
            mime_type = ""

            if "image/" in content_type:
                mime_type = content_type.split(";")[0].strip()
            else:
                # Determine from content
                img_type = imghdr.what(None, h=image_bytes)
                if img_type:
                    mime_type = f"image/{img_type}"
                else:
                    logger.warning(
                        f"Could not determine image type for URL: {image_url}"
                    )
                    raise MimeTypeError("Could not determine image MIME type")

            image_base64: str = base64.b64encode(image_bytes).decode("utf-8")
            data_url: str = f"data:{mime_type};base64,{image_base64}"
            logger.info(
                f"Successfully converted image from {image_url} to base64 data URL ({mime_type})."
            )
            return data_url
    except httpx.InvalidURL:
        logger.exception(f"Invalid image URL provided for b64 download: {image_url}")
        raise InvalidURLError(f"Invalid URL: {image_url}")
    except httpx.HTTPStatusError as e:
        logger.exception(
            f"HTTP error downloading image {image_url} for b64: {e.response.status_code}"
        )
        raise DownloadError(f"HTTP error: {e.response.status_code}")
    except Exception as e:
        logger.exception(f"Error downloading image {image_url} for b64 conversion")
        raise DownloadError(f"Failed to download or convert image: {e!s}")


class MaskProcessingError(Exception):
    """Raised when an error occurs during mask processing."""


def create_alpha_mask_from_mask(
    mask_path: Path, output_dir: Path = Path(".cache/processed_masks")
) -> Path:
    """
    Creates an alpha channel mask from a regular mask image.
    The white areas in the mask become transparent in the output.

    Args:
        mask_path: Path to the input mask image (black areas = masked, white = visible).
        output_dir: Directory to save the processed mask image.

    Returns:
        Path to the processed mask image with alpha channel.

    Raises:
        ImageFileNotFoundError: If the mask file is not found.
        MaskProcessingError: If there was an error during processing.
    """
    if not mask_path.exists():
        logger.error(f"Mask file not found at: {mask_path}")
        raise ImageFileNotFoundError("Mask file not found")

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_mask_path = output_dir / f"{mask_path.stem}_alpha.png"

    try:
        # Open the mask image
        mask_image = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Create a transparent image (RGBA)
        mask_rgba = Image.new("RGBA", mask_image.size, (0, 0, 0, 0))

        # Create data arrays
        mask_data = np.array(mask_image)
        rgba_data = np.zeros((mask_image.height, mask_image.width, 4), dtype=np.uint8)

        # Set alpha based on mask (invert mask: white in mask = transparent in output)
        rgba_data[..., 3] = 255 - mask_data

        # Update the RGBA image and save
        mask_rgba = Image.fromarray(rgba_data, "RGBA")
        # Save the mask directly without opening a file handle separately
        mask_rgba.save(alpha_mask_path, format="PNG")
        logger.info(f"Created mask with alpha channel at: {alpha_mask_path}")
        return alpha_mask_path
    except FileNotFoundError:
        logger.exception(
            f"Error processing mask: Input file {mask_path} not found during PIL operations."
        )
        raise ImageFileNotFoundError("Mask file not found during processing")
    except OSError:
        logger.exception(
            f"IOError processing mask {mask_path} or saving to {alpha_mask_path}"
        )
        raise MaskProcessingError("IO error during mask processing")
    except Exception:  # Catch other PIL errors
        logger.exception(f"Unexpected error creating alpha mask from {mask_path}")
        raise MaskProcessingError("Failed to process mask image")


class AudioProcessingError(Exception):
    """Raised when there's an error processing audio files."""


def convert_audio_to_waveform_video(
    audio_file: str | Path, video_file: str | Path
) -> Path:
    """
    Converts an audio file to a video visualization of the waveform.

    Args:
        audio_file: Path to the input audio file.
        video_file: Path where the output video should be saved.

    Returns:
        Path to the created video file.

    Raises:
        AudioProcessingError: If any part of the conversion fails.
    """
    audio_file_path = Path(audio_file)
    video_file_path = Path(video_file)

    # Create output directory if it doesn't exist
    video_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load audio using pydub for waveform extraction
        audio: AudioSegment = AudioSegment.from_file(audio_file_path)

        # Get the raw audio data and convert to numpy array
        # This works for mono or stereo by averaging channels
        audio_channels = audio.split_to_mono()
        samples = np.array(
            [channel.get_array_of_samples() for channel in audio_channels]
        )
        samples = np.mean(samples, axis=0)

        # Parameters for the video
        fps = 30
        duration = len(audio) / 1000  # Duration in seconds
        total_frames = int(fps * duration)
        frame_width, frame_height = 1280, 720

        # Preprocessing: smooth the waveform slightly to reduce noise in visualization
        smoothed_samples = gaussian_filter1d(samples, sigma=2)

        # Generate frames
        frames = []

        def _generate_frame_for_time(time_point: float) -> np.ndarray:
            """Helper function to generate a single video frame for the waveform."""
            # Create a blank frame
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Set background color (dark gray)
            frame[:, :] = [20, 20, 20]

            # Calculate the start and end samples for this frame
            sample_rate = len(samples) / duration
            start_sample = int(max(0, time_point * sample_rate - sample_rate / fps * 2))
            end_sample = int(
                min(len(samples), time_point * sample_rate + sample_rate / fps * 3)
            )

            if start_sample >= end_sample or end_sample <= 0:
                return frame  # Return empty frame if no samples to display

            # Get the segment of samples for this frame
            current_samples = smoothed_samples[start_sample:end_sample]

            # Normalize sample values to the frame height
            # This scaling is important for visualization clarity
            # (e.g. max value of a 16-bit sample if that's what pydub provides)
            # For simplicity, normalizing based on current segment's min/max for now.
            # A more stable visualization might normalize against global min/max
            # or a fixed range.
            y_min_val, y_max_val = current_samples.min(), current_samples.max()
            y_range = y_max_val - y_min_val

            if y_range < 1e-6:  # Avoid division by near-zero
                y_range = 1.0

            # Scale factor for drawing horizontal points
            x_scale = frame_width / (end_sample - start_sample)
            # Note: vertical scaling is applied directly when calculating y-coordinates

            # Draw lines connecting points in the waveform
            points = []
            for i, sample_val in enumerate(current_samples):
                x = int(i * x_scale)

                # Normalize y to [0, 1] range
                normalized_y = (sample_val - y_min_val) / y_range

                # Map to frame height with proper centering
                # Invert y because image coordinates increase downward
                y = int(frame_height * (0.5 - 0.4 * normalized_y))

                points.append((x, y))

            # Determine volume threshold based on the max possible value of samples
            # (e.g., for 16-bit audio)
            # This threshold helps in not drawing lines for very low noise.
            # Assuming samples are in a typical range (e.g. -32768 to 32767 for 16-bit)
            # A fixed threshold might be better than one relative to current segment's max.
            # For now, let's use a simple heuristic: don't draw if amplitude is too low.
            # This threshold is relative to the *normalized* height.
            amplitude_threshold = 0.05

            # Draw with anti-aliasing for smoother appearance
            for i in range(1, len(points)):
                # Only draw if the amplitude is significant
                if (
                    abs(points[i][1] - frame_height / 2)
                    > amplitude_threshold * frame_height
                ):
                    cv2.line(
                        frame,
                        points[i - 1],
                        points[i],
                        color=(0, 200, 255),  # Orange-yellow color
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

            return frame

        # Generate all frames
        for frame_num in range(total_frames):
            time_point = frame_num / fps
            frame = _generate_frame_for_time(time_point)
            frames.append(frame)

        if not frames:
            logger.error(
                "No frames generated for video. Audio might be too short or empty."
            )
            raise AudioProcessingError("Video frame generation failed")

        moviepy_audio_clip: AudioFileClip = AudioFileClip(str(audio_file_path))

        video_clip = ImageSequenceClip(frames, fps=fps)
        video_clip = video_clip.set_audio(moviepy_audio_clip)

        # Write the video file
        video_clip.write_videofile(
            str(video_file_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=None,
            remove_temp=True,
            preset="medium",
            fps=fps,
            logger=None,  # Suppress moviepy's verbose logging
        )
        logger.info(f"Waveform video successfully created: {video_file_path}")
        return video_file_path
    except Exception:
        logger.exception(f"Error converting audio {audio_file_path} to video")
        raise AudioProcessingError("Failed to convert audio to waveform video")


# --- Obscast-specific Utility Functions ---

def format_duration(seconds: float | None) -> str:
    """Formats a duration in seconds into a human-readable HH:MM:SS string."""
    if seconds is None or seconds < 0:
        return "N/A"
    delta = timedelta(seconds=int(seconds))
    return str(delta)


def format_file_size(size_bytes: int) -> str:
    """Formats a file size in bytes into a human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    size_gb = size_mb / 1024
    return f"{size_gb:.2f} GB"


def truncate_text(text: str, max_length: int) -> str:
    """Truncates text to a maximum length, adding an ellipsis if needed."""
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


# --- Embed Creation Utilities for Obscast ---

BOT_COLOR = 0x5865F2  # Discord blurple
ERROR_COLOR = 0xED4245  # Discord red
SUCCESS_COLOR = 0x57F287  # Discord green


def create_error_embed(message: str, title: str = "❌ An Error Occurred") -> discord.Embed:
    """Creates a standardized error embed."""
    return discord.Embed(title=title, description=message, color=ERROR_COLOR)


def create_success_embed(message: str, title: str = "✅ Success") -> discord.Embed:
    """Creates a standardized success embed."""
    return discord.Embed(title=title, description=message, color=SUCCESS_COLOR)


def create_now_playing_embed(current: dict[str, Any], queue: dict[str, Any]) -> discord.Embed:
    """Creates the rich embed for the '/now' command."""
    media_file = current.get("media_file")
    if not media_file:
        return create_error_embed("Nothing is currently playing.")

    title = media_file.get("display_name") or media_file.get("name", "Unknown Media")
    status = "▶️ Playing" if current.get("is_playing") else "⏸️ Paused"

    embed = discord.Embed(title=title, description=status, color=BOT_COLOR)

    # Progress Bar
    position = current.get("position", 0.0)
    duration = current.get("duration", 0.0)
    progress_bar = "─" * 20
    if duration > 0:
        percent = position / duration
        filled_blocks = int(percent * 20)
        progress_bar = "█" * filled_blocks + "─" * (20 - filled_blocks)

    progress_str = f"`{format_duration(position)}` {progress_bar} `{format_duration(duration)}`"
    embed.add_field(name="Progress", value=progress_str, inline=False)

    # Queue Info
    queue_items = queue.get("items", [])
    if queue_items:
        next_up_file = queue_items[0].get("media_file", {})
        next_up_title = next_up_file.get("display_name") or next_up_file.get("name", "N/A")
        queue_count = len(queue_items)
        embed.add_field(name="Next Up", value=truncate_text(next_up_title, 100), inline=True)
        embed.add_field(name="Queue", value=f"{queue_count} item(s)", inline=True)

    embed.set_footer(text=f"Media ID: {media_file['id']}")
    started_at = current.get("started_at")
    embed.timestamp = datetime.fromisoformat(started_at) if started_at else discord.utils.utcnow()

    return embed


def create_search_results_embed(results: list[dict[str, Any]], query: str, page: int, per_page: int) -> discord.Embed:
    """Creates a paginated embed for media search results."""
    start_index = page * per_page
    end_index = start_index + per_page
    page_results = results[start_index:end_index]

    embed = discord.Embed(
        title=f"🔎 Search Results for '{query}'",
        description=f"Showing results {start_index + 1}-{min(end_index, len(results))} of {len(results)}",
        color=BOT_COLOR,
    )

    if not page_results:
        embed.description = "No results found on this page."
        return embed

    for i, media in enumerate(page_results, start=start_index + 1):
        name = media.get("display_name") or media.get("name", "Unknown")
        duration = format_duration(media.get("metadata", {}).get("duration"))
        size = format_file_size(media.get("size", 0))

        field_value = f"**Duration:** {duration} | **Size:** {size}\n`ID: {media['id']}`"
        embed.add_field(name=f"#{i}. {truncate_text(name, 200)}", value=field_value, inline=False)

    return embed
