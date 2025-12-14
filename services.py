# services.py

"""
Core service integrations for the Discord bot.

This module handles interactions with external APIs such as OpenAI, Anthropic,
Fal AI, Google Search, YouTube, NWS Weather, and ComfyUI for various functionalities
like chat completions, image generation, speech synthesis, and data retrieval.
"""

import asyncio
import base64
import logging
import os
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Literal

import fal_client
import httpx
import openai  # Main openai client
from anthropic import AsyncAnthropic  # Separate client for Anthropic
from anthropic.types import Message as AnthropicMessage  # Alias to avoid confusion
from openai import OpenAI as OpenAIClient  # Explicitly alias for clarity
from PIL import Image

# Assuming comfy_api_simplified is installed and available
# If it has type stubs, they would be beneficial. For now, using Any.
try:
    from comfy_api_simplified import ComfyApiWrapper, ComfyWorkflowWrapper
except ImportError:
    ComfyApiWrapper = Any
    ComfyWorkflowWrapper = Any

# Local application/library specific imports
from tasks import simple_poll_task
from utils import (  # Assuming these are correctly defined in utils.py
    convert_audio_to_waveform_video,
    # image_to_base64_url, # Not directly used in this file after refactor, but kept if utils uses it
    # create_mask_with_alpha, # Not directly used here, gptimg command prepares mask
)

# Initialize logger
logger: logging.Logger = logging.getLogger(__name__)


# --- OpenAI Chat Completion Services ---
async def gpt_chat_completion(
    messages: list[dict[str, Any]],
    model_name: str,
    seed: int | None = None,
) -> str:
    """
    Generates a chat completion using OpenAI's GPT models.

    Args:
        messages: A list of message objects, following OpenAI's API format.
        model_name: The name of the GPT model to use (e.g., "gpt-4o-mini").
        seed: An optional seed for deterministic output.

    Returns:
        The content of the assistant's response message.

    Raises:
        openai.APIError: If the OpenAI API returns an error.
        ValueError: If the response structure is unexpected.
    """
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise ValueError("OpenAI API key is not configured.")

    client: OpenAIClient = OpenAIClient(api_key=openai_api_key)
    api_args: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if seed is not None:
        api_args["seed"] = seed

    try:
        logger.info(f"Requesting OpenAI chat completion: model={model_name}, num_messages={len(messages)}, seed={seed}")
        completion = client.chat.completions.create(**api_args)

        if completion.choices and completion.choices[0].message:
            response_content: str | None = completion.choices[0].message.content
            if response_content is not None:
                logger.info(f"OpenAI chat completion successful for model {model_name}.")
                return response_content
            else:
                logger.error("OpenAI API response content is None.")
                raise ValueError("OpenAI API returned an empty message content.")
        else:
            logger.error(f"Unexpected response structure from OpenAI API: {completion}")
            raise ValueError("Invalid response structure from OpenAI API.")

    except openai.APIConnectionError as e:
        logger.exception(f"OpenAI API connection error: {e}")
        raise
    except openai.RateLimitError as e:
        logger.exception(f"OpenAI API rate limit exceeded: {e}")
        raise
    except openai.AuthenticationError as e:
        logger.exception(f"OpenAI API authentication error: {e}")
        raise
    except openai.APIStatusError as e:
        logger.exception(f"OpenAI API status error (code {e.status_code}): {e.response}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during OpenAI chat completion: {e}")
        raise


# --- OpenAI Speech Generation Service ---
# Define a type alias for voice options to improve readability
# Updated October 2024 - deprecated: fable, onyx, nova (old shimmer)
VoiceType = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]


async def generate_speech(text: str, voice: VoiceType, speed: float) -> Path:
    """
    Generates speech from text using OpenAI's TTS API and converts it to a waveform video.

    Args:
        text: The text to synthesize.
        voice: The voice to use for synthesis (e.g., "onyx").
        speed: The speed of the speech (e.g., 1.0).

    Returns:
        The file path to the generated waveform video (.mp4).

    Raises:
        ValueError: If the OpenAI API key is not configured.
        openai.APIError: If the TTS API call fails.
    """
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables for TTS.")
        raise ValueError("OpenAI API key is not configured for TTS.")

    # Initialize client here to ensure API key is sourced correctly for this specific call
    # This is okay as client instantiation is lightweight.
    # Alternatively, a module-level client could be used if initialized carefully after dotenv load.
    tts_client: OpenAIClient = OpenAIClient(api_key=openai_api_key)

    # Sanitize text for filename, ensuring it's a valid path component
    safe_text_suffix: str = "".join(c if c.isalnum() else "_" for c in text[:50])
    cache_dir: Path = Path(".cache/tts")
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_filename: str = f"{voice}_{speed}_{safe_text_suffix}"
    speech_file_path: Path = cache_dir / f"{base_filename}.mp3"
    video_file_path: Path = cache_dir / f"{base_filename}.mp4"

    logger.info(f"Generating speech for text (first 50 chars): '{text[:50]}...'")
    try:
        response = tts_client.audio.speech.create(
            model="tts-1-hd",  # Or "tts-1"
            voice=voice,  # Using correct literal type that matches API expectations
            input=text,
            speed=speed,
            response_format="mp3",
        )
        response.stream_to_file(speech_file_path)
        logger.info(f"Speech audio saved to: {speech_file_path}")
    except openai.APIError as e:
        logger.exception(f"OpenAI TTS API error for text '{text[:50]}...': {e}")
        raise

    logger.info(f"Converting speech audio at {speech_file_path} to video at {video_file_path}")
    # The convert_audio_to_waveform_video function is from utils.py
    # and is assumed to handle its own errors or let them propagate.
    convert_audio_to_waveform_video(audio_file=str(speech_file_path), video_file=str(video_file_path))
    logger.info(f"Waveform video saved to: {video_file_path}")
    return video_file_path


# --- Anthropic Chat Completion Service ---
def _format_anthropic_message(msg: AnthropicMessage) -> str:
    """
    Renders an Anthropic Message object into a Markdown string for Discord.
    Inlines citations as [title](url).

    Args:
        msg: The Anthropic Message object.

    Returns:
        A string formatted for Discord.
    """
    lines: list[str] = []
    for item in msg.content:
        if item.type == "text":
            text_content: str = item.text.strip()
            # Anthropic's Python SDK currently doesn't directly expose 'citations'
            # in the same way as some examples might suggest for other libraries.
            # If citations are part of the 'text_content' or a different structure,
            # this part would need adjustment based on actual API response.
            # For now, assuming citations are not separately structured in 'item.citations'.
            lines.append(text_content)
    return "\n\n".join(lines)


async def anthropic_chat_completion(
    prompt: str,
    max_tokens: int = 1024,
    model: str = "claude-3-5-sonnet-20240620",
) -> str:
    """
    Generates a chat completion using Anthropic's Claude models.

    Args:
        prompt: The user's prompt.
        max_tokens: The maximum number of tokens to generate.
        model: The Anthropic model to use.

    Returns:
        The formatted assistant's response message.

    Raises:
        ValueError: If the Anthropic API key is not configured.
        anthropic.APIError: If the Anthropic API returns an error.
    """
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables.")
        raise ValueError("Anthropic API key is not configured.")

    # Client automatically picks up ANTHROPIC_API_KEY from env
    anthropic_client: AsyncAnthropic = AsyncAnthropic()

    logger.info(f"Requesting Anthropic completion: model={model}, prompt='{prompt[:50]}...'")
    try:
        message: AnthropicMessage = await anthropic_client.messages.create(
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        formatted_message: str = _format_anthropic_message(message)
        logger.info(f"Anthropic completion successful for model {model}.")
        return formatted_message
    except Exception as e:  # Catch generic Anthropic API errors
        logger.exception(f"Anthropic API error for model {model}, prompt '{prompt[:50]}...': {e}")
        raise


# --- Fal AI Image Generation Services ---
# --- Web Search Services ---
async def google_search(query: str) -> str:
    """
    Performs a Google search using the Custom Search API.

    Args:
        query: The search query.

    Returns:
        A string containing the top 3 search result links, or an empty string if none.

    Raises:
        ValueError: If Google Search API key or CSE ID is not configured.
        httpx.HTTPStatusError: For HTTP errors from the Google API.
    """
    api_key: str | None = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id: str | None = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not cse_id:
        logger.error("Google Search API key or CSE ID not configured.")
        raise ValueError("Google Search API is not configured.")

    url: str = "https://www.googleapis.com/customsearch/v1"
    params: dict[str, str | int] = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": 3,
    }

    logger.info(f"Performing Google search for query: '{query}'")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses
            data: dict[str, Any] = response.json()
            items: list[dict[str, Any]] = data.get("items", [])
            if not items:
                logger.info(f"No Google search results found for '{query}'.")
                return "No results found."

            links: list[str] = [item["link"] for item in items if "link" in item]
            logger.info(f"Google search successful for '{query}', found {len(links)} links.")
            return "\n".join(links)
        except httpx.HTTPStatusError as e:
            logger.exception(
                f"Google Search API HTTP error for query '{query}': {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.exception(f"Error during Google search for query '{query}': {e}")
            raise


# --- Weather Service ---
FAYETTEVILLE_COORDS: tuple[float, float] = (36.0626, -94.1574)


async def temp_callback() -> str:
    """
    Fetches the current temperature in Fayetteville, AR using the NWS API.

    Returns:
        A string describing the current temperature and wind chill (if available).

    Raises:
        httpx.HTTPStatusError: For HTTP errors from the NWS API.
    """
    lat, lon = FAYETTEVILLE_COORDS
    headers: dict[str, str] = {"User-Agent": "MobiusDiscordBot/1.0 (github.com/afiaka87/mobius)"}

    logger.info(f"Fetching NWS gridpoint for coordinates: {lat}, {lon}")
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            # 1. Get gridpoint URL
            points_url: str = f"https://api.weather.gov/points/{lat},{lon}"
            response_points = await client.get(points_url)
            response_points.raise_for_status()
            points_data: dict[str, Any] = response_points.json()
            forecast_hourly_url: str = points_data.get("properties", {}).get("forecastHourly")

            if not forecast_hourly_url:
                logger.error(f"Could not retrieve hourly forecast URL from NWS API for {lat},{lon}")
                raise ValueError("NWS API did not return a valid hourly forecast URL.")

            # 2. Get hourly forecast
            logger.info(f"Fetching hourly forecast from: {forecast_hourly_url}")
            response_forecast = await client.get(forecast_hourly_url)
            response_forecast.raise_for_status()
            forecast_data: dict[str, Any] = response_forecast.json()

            periods: list[dict[str, Any]] = forecast_data.get("properties", {}).get("periods", [])
            if not periods:
                logger.warning(f"NWS API returned no forecast periods for {forecast_hourly_url}")
                return "Could not retrieve current weather data."

            current_period: dict[str, Any] = periods[0]
            temperature: int | None = current_period.get("temperature")
            temp_unit: str | None = current_period.get("temperatureUnit")
            # NWS API windChill is often given as a full phrase like "10 F ( -12 C)"
            # or just a value. We need to parse it carefully or use a specific field if available.
            # For simplicity, let's assume 'windChill' provides a usable value or is None.
            # The API docs should be consulted for the exact structure of windChill.
            # Example: current_period.get("windChill", {}).get("value") if it's structured.
            # For now, assuming it's a simple value or None.
            wind_chill_value: Any | None = current_period.get("windChill")  # This might be a dict or simple value

            if temperature is None or temp_unit is None:
                return "Temperature data is currently unavailable."

            result_str: str = f"Current temperature in Fayetteville, AR: {temperature}°{temp_unit}"
            # Attempt to parse wind chill if it's a simple numeric value
            # This part is speculative based on typical API structures; adjust if NWS is different.
            if isinstance(wind_chill_value, int | float):
                result_str += f" with wind chill of {wind_chill_value}°{temp_unit}"
            elif (
                isinstance(wind_chill_value, dict)
                and "value" in wind_chill_value
                and isinstance(wind_chill_value["value"], int | float)
            ):
                result_str += f" with wind chill of {wind_chill_value['value']}°{temp_unit}"

            logger.info(f"Successfully fetched weather: {result_str}")
            return result_str

        except httpx.HTTPStatusError as e:
            logger.exception(f"NWS API HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.exception(f"Error fetching weather data from NWS: {e}")
            raise


# --- YouTube Search Service ---
async def get_top_youtube_result(search_query: str, api_key: str) -> dict[str, Any]:
    """
    Fetches the top YouTube video search result for a given query.

    Args:
        search_query: The search query string.
        api_key: The YouTube Data API key.

    Returns:
        A dictionary containing video details or an error message.
    """
    base_url: str = "https://www.googleapis.com/youtube/v3/search"
    params: dict[str, Any] = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "maxResults": 1,
        "key": api_key,
    }

    logger.info(f"Searching YouTube for: '{search_query}'")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            items: list[dict[str, Any]] = data.get("items", [])

            if not items:
                logger.info(f"No YouTube results found for '{search_query}'.")
                return {"error": "No results found"}

            top_result: dict[str, Any] = items[0]
            video_info: dict[str, Any] = {
                "videoId": top_result.get("id", {}).get("videoId"),
                "title": top_result.get("snippet", {}).get("title"),
                "description": top_result.get("snippet", {}).get("description"),
                "channelTitle": top_result.get("snippet", {}).get("channelTitle"),
            }
            if not video_info["videoId"]:  # Essential field missing
                logger.error(f"YouTube API response missing videoId for query '{search_query}': {top_result}")
                return {"error": "Malformed response from YouTube API (missing videoId)"}

            logger.info(f"YouTube search successful for '{search_query}', found videoId: {video_info['videoId']}")
            return video_info

        except httpx.HTTPStatusError as e:
            logger.exception(
                f"YouTube API HTTP error for query '{search_query}': {e.response.status_code} - {e.response.text}"
            )
            return {"error": f"YouTube API error: {e.response.status_code}"}
        except Exception as e:
            logger.exception(f"Error during YouTube search for query '{search_query}': {e}")
            return {"error": "An unexpected error occurred during YouTube search."}


# --- ComfyUI Text-to-Video Service ---
async def generate_gpt_image(
    prompt: str,
    model: str = "gpt-image-1",
    quality: Literal["low", "medium", "high", "auto"] = "auto",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    transparent_background: bool = False,
    user: str | None = None,
) -> Path:
    """
    Generates an image using OpenAI's GPT Image model.

    Args:
        prompt: The text prompt for image generation.
        model: The OpenAI image model to use (e.g., "gpt-image-1").
        quality: Image quality ("low", "medium", "high", or "auto").
        size: Image size ("1024x1024", "1536x1024", "1024x1536", or "auto").
        transparent_background: Whether to generate with transparent background.
        user: A unique identifier for the end-user.

    Returns:
        Path to the generated image file (saved as PNG to support transparency).

    Raises:
        ValueError: If OpenAI API key is not configured or for invalid parameters.
        RuntimeError: For API errors or other operational issues.
        openai.APIError: For specific OpenAI API errors.
    """
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found for image generation.")
        raise ValueError("OpenAI API key is not configured.")

    client: OpenAIClient = OpenAIClient(api_key=openai_api_key)
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    # Build API parameters for GPT Image - include transparent_background if supported
    api_params: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "size": size,
    }

    # Add optional parameters
    if transparent_background:
        api_params["transparent_background"] = transparent_background

    if user:
        api_params["user"] = user

    logger.info(f"Requesting GPT Image generation with model {model}: {prompt[:50]}...")

    try:
        result = await loop.run_in_executor(
            None,
            lambda: client.images.generate(**api_params),
        )

        if not result.data or not result.data[0]:
            logger.error("GPT Image generation returned no image data.")
            raise RuntimeError("GPT Image generation failed to return image data.")

        image_b64_json: str | None = result.data[0].b64_json

        if not image_b64_json:
            logger.error("GPT Image generation did not return b64_json data.")
            raise RuntimeError("GPT Image generation did not provide b64_json data.")

        image_bytes: bytes = base64.b64decode(image_b64_json)

        # Save the image as PNG to support transparency
        cache_dir: Path = Path(".cache/gptimg_generated")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt_suffix: str = "".join(c if c.isalnum() else "_" for c in prompt[:30])
        filename: str = f"{model}_{safe_prompt_suffix}_{size}_{quality}.png"
        file_path: Path = cache_dir / filename

        def save_image_file() -> None:
            """Helper to save image as PNG to preserve transparency."""
            image: Image.Image = Image.open(BytesIO(image_bytes))
            # Save as PNG to preserve any transparency
            image.save(file_path, format="PNG", optimize=True)

        await loop.run_in_executor(None, save_image_file)
        logger.info(f"GPT Image generated and saved to: {file_path}")
        return file_path

    except openai.BadRequestError as e:
        logger.exception(f"GPT Image generation bad request: {e}")
        error_detail = str(e)
        if hasattr(e, "body") and e.body and isinstance(e.body, dict):
            err_dict = e.body.get("error", {})
            if isinstance(err_dict, dict) and "message" in err_dict:
                error_detail = err_dict["message"]
        raise ValueError(f"Invalid request for GPT image generation: {error_detail}")
    except openai.APIError as e:
        logger.exception(f"GPT Image generation API error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error generating GPT image: {e}")
        raise RuntimeError(f"An unexpected error occurred during image generation: {e!s}")


async def edit_gpt_image(
    prompt: str,
    images: Sequence[Path | BinaryIO],
    mask: Path | BinaryIO | None = None,
    model: str = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "auto",
    user: str | None = None,
) -> Path:
    """
    Edits images using OpenAI's GPT Image model.

    Args:
        prompt: Text description of the desired edits.
        images: A sequence of input image file paths or binary file objects (max 10).
                If multiple images, the mask applies to the first image.
        mask: Optional mask file path or binary file object (PNG with alpha channel).
              Applied to the first image if multiple images are provided.
        model: The OpenAI model to use (e.g., "gpt-image-1").
        size: The size of the output image.
        user: Optional user ID for tracking.

    Returns:
        Path to the edited image file (saved as JPEG with 95% quality).

    Raises:
        ValueError: If API key not configured, no images provided, or too many images.
        RuntimeError: For API errors or other operational issues.
        openai.APIError: For specific OpenAI API errors.
    """
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found for image editing.")
        raise ValueError("OpenAI API key is not configured.")

    if not images:
        raise ValueError("At least one image is required for editing.")
    if len(images) > 10:
        logger.warning(f"More than 10 images provided ({len(images)}), only first 10 will be used.")
        images = images[:10]

    client: OpenAIClient = OpenAIClient(api_key=openai_api_key)
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    # Prepare file objects - ensure they're in the right format for GPT Image API
    image_files: list[BinaryIO] = []
    files_to_close: list[BinaryIO] = []
    mask_file: BinaryIO | None = None
    mask_file_to_close: BinaryIO | None = None

    try:
        # Prepare image files - make sure they're proper file objects
        for img_src in images:
            if isinstance(img_src, Path):
                file_obj = open(img_src, "rb")
                files_to_close.append(file_obj)
                image_files.append(file_obj)
            else:  # BinaryIO
                img_src.seek(0)
                image_files.append(img_src)

        # Prepare mask file if provided
        if mask:
            if isinstance(mask, Path):
                mask_file = open(mask, "rb")
                mask_file_to_close = mask_file
            else:  # BinaryIO
                mask.seek(0)
                mask_file = mask

        # OpenAI's image edit API only supports a single image, not multiple images
        # Use the first image if multiple are provided
        if len(image_files) > 1:
            logger.warning(f"GPT Image edit API only supports single image, using first of {len(image_files)} provided")

        # Build API parameters with single image
        api_params: dict[str, Any] = {
            "model": model,
            "image": image_files[0],  # Always use the first (and typically only) image
            "prompt": prompt,
            "size": size,
        }

        if mask_file:
            api_params["mask"] = mask_file

        if user:
            api_params["user"] = user

        logger.info(
            f"Requesting GPT Image edit: model={model}, num_images={len(image_files)}, "
            f"mask_present={bool(mask_file)}, prompt='{prompt[:50]}...'"
        )

        result = await loop.run_in_executor(None, lambda: client.images.edit(**api_params))

        if not result.data or not result.data[0].b64_json:
            logger.error("GPT Image editing returned no image data.")
            raise RuntimeError("GPT Image editing failed to return image data.")

        image_base64: str = result.data[0].b64_json
        image_bytes: bytes = base64.b64decode(image_base64)

        # Save the edited image as PNG to support transparency
        cache_dir: Path = Path(".cache/gptimg_edited")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt_suffix: str = "".join(c if c.isalnum() else "_" for c in prompt[:30])
        filename: str = f"{model}_edit_{safe_prompt_suffix}_{size}.png"
        file_path: Path = cache_dir / filename

        def save_edited_image_file() -> None:
            """Helper to save the edited image as PNG to preserve transparency."""
            image: Image.Image = Image.open(BytesIO(image_bytes))
            # Save as PNG to preserve any transparency
            image.save(file_path, format="PNG", optimize=True)

        await loop.run_in_executor(None, save_edited_image_file)
        logger.info(f"GPT Image edited and saved to: {file_path}")
        return file_path

    except openai.BadRequestError as e:
        logger.exception(f"GPT Image editing bad request: {e}")
        error_detail = str(e)
        if hasattr(e, "body") and e.body and isinstance(e.body, dict):
            err_dict = e.body.get("error", {})
            if isinstance(err_dict, dict) and "message" in err_dict:
                error_detail = err_dict["message"]

        if "mask" in error_detail.lower() and "alpha" in error_detail.lower():
            raise ValueError(f"Invalid mask: {error_detail}. Ensure it's a PNG with a proper alpha channel.")
        raise ValueError(f"Invalid request for GPT image editing: {error_detail}")

    except openai.APIError as e:
        logger.exception(f"GPT Image editing API error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error editing GPT image: {e}")
        raise RuntimeError(f"An unexpected error occurred during image editing: {e!s}")
    finally:
        # Clean up opened files
        for f in files_to_close:
            try:
                f.close()
            except Exception as e_close:
                logger.exception(f"Error closing image file: {e_close}")

        if mask_file_to_close:
            try:
                mask_file_to_close.close()
            except Exception as e_close:
                logger.exception(f"Error closing mask file: {e_close}")


# --- Celeste Diffusion API Service ---

# --- Obscast Media API Service ---


class ObscastAPIError(Exception):
    """Custom exception for Obscast API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ObscastAPIClient:
    """A simple async client to communicate with the Obscast backend API."""

    def __init__(self, base_url: str | None, timeout: float = 30.0) -> None:
        if not base_url:
            raise ValueError("OBSCAST_API_URL is not configured.")
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self.base_url = base_url
        logger.info(f"Obscast API Client initialized for base URL: {base_url}")

    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Performs an API request and handles errors."""
        try:
            response = await self._client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            logger.exception(f"Obscast API request failed: {e.response.status_code} - {e.response.text}")
            raise ObscastAPIError(f"API request failed: {e.response.text}", e.response.status_code) from e
        except httpx.RequestError as e:
            logger.exception(f"Obscast API connection error: {e}")
            raise ObscastAPIError(f"Could not connect to Obscast API at {self.base_url}") from e

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("GET", endpoint, params=params or {})

    async def post(self, endpoint: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("POST", endpoint, json=json or {})

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# Initialize the client from environment variables
OBSCAST_API_URL: str | None = os.getenv("OBSCAST_API_URL")
obscast_client: ObscastAPIClient | None = None
if OBSCAST_API_URL:
    obscast_client = ObscastAPIClient(base_url=OBSCAST_API_URL)
else:
    logger.warning("OBSCAST_API_URL not set. Media commands will not be available.")


K5_API_URL = "http://100.105.155.18:8888"
SD_API_URL = "http://100.70.95.57:8889"
ZIMAGE_API_URL = "https://archest.tailfd5df.ts.net"


# --- Kandinsky-5 Video Generation Services ---
async def check_kandinsky5_health() -> bool:
    """
    Check if the Kandinsky-5 API is healthy and available.

    Returns:
        True if the API is healthy, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{K5_API_URL}/health")
            response.raise_for_status()

            # If we got a 200 response, the API is healthy
            logger.info("Kandinsky-5 API health check: OK (HTTP 200)")
            return True

    except httpx.ConnectError:
        logger.warning(f"Kandinsky-5 API health check: Cannot connect to {K5_API_URL}")
        return False
    except httpx.TimeoutException:
        logger.warning(f"Kandinsky-5 API health check: Connection timeout to {K5_API_URL}")
        return False
    except httpx.HTTPStatusError as e:
        logger.warning(f"Kandinsky-5 API health check: HTTP error {e.response.status_code}")
        return False
    except Exception as e:
        logger.exception(f"Kandinsky-5 API health check: Unexpected error - {type(e).__name__}: {e}")
        return False


async def generate_kandinsky5_video(
    prompt: str,
    negative_prompt: str | None = None,
    duration: int = 5,
    width: int = 512,
    height: int = 512,
    num_steps: int = 50,
    guidance_weight: float | None = None,
    scheduler_scale: float = 5.0,
    seed: int | None = None,
    progress_callback: Any = None,
) -> Path:
    """
    Generate a single video using the Kandinsky-5 API with async task polling.

    Args:
        prompt: Text description of the video to generate
        negative_prompt: Optional negative guidance to avoid certain content
        duration: Duration of the video in seconds (any positive integer)
        width: Video width in pixels (must be a multiple of 16, default: 512)
        height: Video height in pixels (must be a multiple of 16, default: 512)
        num_steps: Number of inference steps (default: 50)
        guidance_weight: Optional CFG weight (default: None, uses model default)
        scheduler_scale: Flow matching scheduler scale (default: 5.0)
        seed: Optional random seed for reproducible results
        progress_callback: Optional async callback function(TaskProgress) for progress updates

    Returns:
        Path to the saved video file

    Raises:
        ValueError: If duration is not positive, or if width/height are not multiples of 16
        RuntimeError: If API call fails or returns invalid data
    """

    # Validate duration (any positive integer is allowed)
    if duration <= 0:
        raise ValueError(f"Duration must be a positive integer, got {duration}")

    # Validate resolution (width and height must be multiples of 16)
    if width % 16 != 0:
        raise ValueError(f"Width must be a multiple of 16, got {width}")
    if height % 16 != 0:
        raise ValueError(f"Height must be a multiple of 16, got {height}")

    logger.info(
        f"Generating Kandinsky-5 video: prompt='{prompt[:50]}...', "
        f"duration={duration}s, resolution={width}x{height}, steps={num_steps}, "
        f"guidance_weight={guidance_weight}, scheduler_scale={scheduler_scale}, seed={seed}"
    )

    payload = {
        "prompt": prompt,
        "duration": duration,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "scheduler_scale": scheduler_scale,
    }

    # Add optional parameters if provided
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if guidance_weight is not None:
        payload["guidance_weight"] = guidance_weight
    if seed is not None:
        payload["seed"] = seed

    try:
        # Submit task to API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{K5_API_URL}/generate", json=payload)
            response.raise_for_status()
            submit_data = response.json()

            if "task_id" not in submit_data:
                logger.error("Kandinsky-5 API response missing task_id")
                raise RuntimeError("Kandinsky-5 API did not return task_id")

            task_id = submit_data["task_id"]
            logger.info(f"Kandinsky-5 video task submitted: {task_id}")

        # Poll for completion
        result_data = await simple_poll_task(
            task_id=task_id,
            base_url=K5_API_URL,
            status_endpoint=f"/task/{task_id}",
            poll_interval=15.0,
            max_duration=3600.0,
            on_progress=progress_callback,
        )

        # DEBUG: Log what we got back
        logger.info(f"Task {task_id} completed. Result keys: {list(result_data.keys())}")
        logger.debug(f"Full result_data: {result_data}")

        # Extract video data from result
        if "video_base64" not in result_data:
            logger.error(f"Kandinsky-5 API result missing video data. Got keys: {list(result_data.keys())}")
            logger.error(f"Full response: {result_data}")
            raise RuntimeError("Kandinsky-5 API did not return video data")

        # Create cache directory
        cache_dir: Path = Path(".cache/kandinsky5")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Decode and save video
        video_bytes: bytes = base64.b64decode(result_data["video_base64"])

        # Generate filename
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
        filename = f"kandinsky5_{safe_prompt}_{duration}s.mp4"
        file_path = cache_dir / filename

        with open(file_path, "wb") as f:
            f.write(video_bytes)

        logger.info(f"Kandinsky-5 video saved: {file_path}")
        return file_path

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Kandinsky-5 API at {K5_API_URL}")
        raise RuntimeError(f"Cannot connect to Kandinsky-5 API at {K5_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Kandinsky-5 API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Kandinsky-5 API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error generating Kandinsky-5 video: {e}")
        raise RuntimeError(f"Failed to generate Kandinsky-5 video: {e!s}")


async def generate_kandinsky5_batch(
    prompts: list[str],
    negative_prompt: str | None = None,
    duration: int = 5,
    width: int = 512,
    height: int = 512,
    num_steps: int = 50,
    guidance_weight: float | None = None,
    scheduler_scale: float = 5.0,
    seed: int | None = None,
    progress_callback: Any = None,
) -> list[Path]:
    """
    Generate multiple videos using the Kandinsky-5 batch API with async task polling.

    Args:
        prompts: List of text descriptions for videos to generate
        negative_prompt: Optional negative guidance applied to all videos
        duration: Duration of each video in seconds (any positive integer)
        width: Video width in pixels (must be a multiple of 16, default: 512)
        height: Video height in pixels (must be a multiple of 16, default: 512)
        num_steps: Number of inference steps (default: 50)
        guidance_weight: Optional CFG weight (default: None, uses model default)
        scheduler_scale: Flow matching scheduler scale (default: 5.0)
        seed: Optional starting seed for batch (auto-increments for each video)
        progress_callback: Optional async callback function(TaskProgress) for progress updates

    Returns:
        List of paths to the saved video files

    Raises:
        ValueError: If duration is not positive, or if width/height are not multiples of 16
        RuntimeError: If API call fails or returns invalid data
    """
    # Validate duration (any positive integer is allowed)
    if duration <= 0:
        raise ValueError(f"Duration must be a positive integer, got {duration}")

    # Validate resolution (width and height must be multiples of 16)
    if width % 16 != 0:
        raise ValueError(f"Width must be a multiple of 16, got {width}")
    if height % 16 != 0:
        raise ValueError(f"Height must be a multiple of 16, got {height}")

    logger.info(
        f"Generating Kandinsky-5 batch: {len(prompts)} videos, "
        f"duration={duration}s, resolution={width}x{height}, steps={num_steps}, "
        f"guidance_weight={guidance_weight}, scheduler_scale={scheduler_scale}, seed={seed}"
    )

    payload = {
        "prompts": prompts,
        "duration": duration,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "scheduler_scale": scheduler_scale,
    }

    # Add optional parameters if provided
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if guidance_weight is not None:
        payload["guidance_weight"] = guidance_weight
    if seed is not None:
        payload["seed"] = seed

    try:
        # Submit batch task to API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{K5_API_URL}/generate/batch", json=payload)
            response.raise_for_status()
            submit_data = response.json()

            if "task_id" not in submit_data:
                logger.error("Kandinsky-5 batch API response missing task_id")
                raise RuntimeError("Kandinsky-5 API did not return task_id")

            task_id = submit_data["task_id"]
            logger.info(f"Kandinsky-5 batch task submitted: {task_id}")

        # Poll for completion with longer timeout for batch operations
        result_data = await simple_poll_task(
            task_id=task_id,
            base_url=K5_API_URL,
            status_endpoint=f"/task/{task_id}",
            poll_interval=20.0,  # Slightly longer interval for batch
            max_duration=7200.0,  # 2 hours max for batch
            on_progress=progress_callback,
        )

        # DEBUG: Log what we got back
        logger.info(f"Batch task {task_id} completed. Result keys: {list(result_data.keys())}")
        logger.debug(f"Full result_data: {result_data}")

        # Extract video data from result
        if "videos" not in result_data or not result_data["videos"]:
            logger.error(f"Kandinsky-5 batch API result missing video data. Got keys: {list(result_data.keys())}")
            logger.error(f"Full response: {result_data}")
            raise RuntimeError("Kandinsky-5 API did not return any videos")

        # Create cache directory
        cache_dir: Path = Path(".cache/kandinsky5")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Decode and save all videos
        video_paths: list[Path] = []

        for i, video_data in enumerate(result_data["videos"]):
            video_base64 = video_data.get("video_base64")
            video_prompt = video_data.get("prompt", prompts[i] if i < len(prompts) else "unknown")

            if not video_base64:
                logger.warning(f"Video {i} missing base64 data, skipping")
                continue

            # Decode video
            video_bytes: bytes = base64.b64decode(video_base64)

            # Generate filename
            safe_prompt = "".join(c if c.isalnum() else "_" for c in video_prompt[:30])
            filename = f"kandinsky5_batch_{i:04d}_{safe_prompt}.mp4"
            file_path = cache_dir / filename

            with open(file_path, "wb") as f:
                f.write(video_bytes)

            video_paths.append(file_path)
            logger.info(f"Kandinsky-5 batch video {i + 1}/{len(result_data['videos'])} saved: {file_path}")

        logger.info(f"Kandinsky-5 batch generation complete. Generated {len(video_paths)} videos.")
        return video_paths

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Kandinsky-5 API at {K5_API_URL}")
        raise RuntimeError(f"Cannot connect to Kandinsky-5 API at {K5_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Kandinsky-5 API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Kandinsky-5 API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error generating Kandinsky-5 batch videos: {e}")
        raise RuntimeError(f"Failed to generate Kandinsky-5 batch videos: {e!s}")


# --- Stable Diffusion 1.5 Services ---
async def check_sd_health() -> bool:
    """
    Check if the Stable Diffusion API is healthy and available.

    Returns:
        True if the API is healthy, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SD_API_URL}/health")
            response.raise_for_status()

            # If we got a 200 response, the API is healthy
            logger.info("Stable Diffusion API health check: OK (HTTP 200)")
            return True

    except httpx.ConnectError:
        logger.warning(f"Stable Diffusion API health check: Cannot connect to {SD_API_URL}")
        return False
    except httpx.TimeoutException:
        logger.warning(f"Stable Diffusion API health check: Connection timeout to {SD_API_URL}")
        return False
    except httpx.HTTPStatusError as e:
        logger.warning(f"Stable Diffusion API health check: HTTP error {e.response.status_code}")
        return False
    except Exception as e:
        logger.exception(f"Stable Diffusion API health check: Unexpected error - {type(e).__name__}: {e}")
        return False


async def get_sd_schedulers() -> list[dict[str, str]]:
    """
    Get the list of available schedulers from the Stable Diffusion API.

    Returns:
        List of scheduler dictionaries with 'name' and 'description' keys

    Raises:
        RuntimeError: If API call fails
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SD_API_URL}/schedulers")
            response.raise_for_status()
            schedulers: list[dict[str, str]] = response.json()

            logger.info(f"Retrieved {len(schedulers)} schedulers from SD API")
            return schedulers

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Stable Diffusion API at {SD_API_URL}")
        raise RuntimeError(f"Cannot connect to Stable Diffusion API at {SD_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Stable Diffusion API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Stable Diffusion API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error getting SD schedulers: {e}")
        raise RuntimeError(f"Failed to get SD schedulers: {e!s}")


async def list_sd_experiments() -> list[str]:
    """
    List all available experiment/run directories.

    Returns:
        List of experiment run paths (e.g., ["laion2b/initial", "laion2b/resume"])

    Raises:
        RuntimeError: If API call fails
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SD_API_URL}/experiments")
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            experiment_runs: list[str] = data.get("experiment_runs", [])
            logger.info(f"Retrieved {len(experiment_runs)} experiment runs from SD API")
            return experiment_runs

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Stable Diffusion API at {SD_API_URL}")
        raise RuntimeError(f"Cannot connect to Stable Diffusion API at {SD_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Stable Diffusion API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Stable Diffusion API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error listing SD experiments: {e}")
        raise RuntimeError(f"Failed to list SD experiments: {e!s}")


async def list_sd_checkpoints(experiment_run: str) -> list[str]:
    """
    List all checkpoints in a specific experiment/run.

    Args:
        experiment_run: The experiment run path (e.g., "laion2b/resume")

    Returns:
        List of checkpoint names (e.g., ["checkpoint-17500", "checkpoint-20000"])

    Raises:
        RuntimeError: If API call fails
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # URL encode the experiment_run path
            from urllib.parse import quote

            encoded_path = quote(experiment_run, safe="")

            response = await client.get(f"{SD_API_URL}/experiments/{encoded_path}/checkpoints")
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            checkpoints: list[str] = data.get("checkpoints", [])
            logger.info(f"Retrieved {len(checkpoints)} checkpoints from {experiment_run}")
            return checkpoints

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Stable Diffusion API at {SD_API_URL}")
        raise RuntimeError(f"Cannot connect to Stable Diffusion API at {SD_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Stable Diffusion API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Stable Diffusion API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error listing SD checkpoints for {experiment_run}: {e}")
        raise RuntimeError(f"Failed to list SD checkpoints: {e!s}")


async def load_sd_checkpoint(checkpoint_path: str | None) -> dict[str, Any]:
    """
    Load a Stable Diffusion checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint (e.g., "checkpoints/laion2b/resume/checkpoint-17500")
                        or None to load the base model

    Returns:
        Dictionary containing:
        - status: Status message
        - checkpoint_name: Name of loaded checkpoint
        - is_lora: Whether the checkpoint is a LoRA

    Raises:
        RuntimeError: If API call fails
    """
    try:
        payload = {"checkpoint_path": checkpoint_path}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{SD_API_URL}/load", json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            logger.info(f"Loaded SD checkpoint: {data.get('checkpoint_name', 'Base Model')}")
            return data

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Stable Diffusion API at {SD_API_URL}")
        raise RuntimeError(f"Cannot connect to Stable Diffusion API at {SD_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Stable Diffusion API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Stable Diffusion API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error loading SD checkpoint: {e}")
        raise RuntimeError(f"Failed to load SD checkpoint: {e!s}")


async def generate_sd_images(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 8,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    seed: int = -1,
    lora_scale: float = 1.0,
    scheduler_name: str = "DPM++ SDE",
) -> list[Path]:
    """
    Generate images using the Stable Diffusion API.

    Args:
        prompt: Text description of desired image
        negative_prompt: Things to avoid in the image
        num_inference_steps: Number of denoising steps (1-100)
        guidance_scale: CFG scale (1.0-20.0)
        width: Image width in pixels (must be multiple of 64)
        height: Image height in pixels (must be multiple of 64)
        num_images: Number of images to generate (1-4)
        seed: Random seed (-1 for random)
        lora_scale: LoRA intensity (0.0-2.0)
        scheduler_name: Sampling method

    Returns:
        List of paths to saved image files

    Raises:
        ValueError: If width/height are not multiples of 64
        RuntimeError: If API call fails or returns invalid data
    """
    # Validate resolution (width and height must be multiples of 64)
    if width % 64 != 0:
        raise ValueError(f"Width must be a multiple of 64, got {width}")
    if height % 64 != 0:
        raise ValueError(f"Height must be a multiple of 64, got {height}")

    logger.info(
        f"Generating SD images: prompt='{prompt[:50]}...', "
        f"resolution={width}x{height}, steps={num_inference_steps}, "
        f"guidance={guidance_scale}, num_images={num_images}, "
        f"scheduler={scheduler_name}, seed={seed}, lora_scale={lora_scale}"
    )

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "num_images": num_images,
        "seed": seed,
        "lora_scale": lora_scale,
        "scheduler_name": scheduler_name,
    }

    try:
        # Submit generation request (synchronous response)
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for generation
            response = await client.post(f"{SD_API_URL}/generate", json=payload)
            response.raise_for_status()
            result_data = response.json()

        # Extract images from result
        if "images" not in result_data or not result_data["images"]:
            logger.error(f"SD API result missing image data. Got keys: {list(result_data.keys())}")
            raise RuntimeError("Stable Diffusion API did not return any images")

        # Create cache directory
        cache_dir: Path = Path(".cache/sd_generated")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Decode and save all images
        image_paths: list[Path] = []

        for i, image_base64 in enumerate(result_data["images"]):
            # Decode image
            image_bytes: bytes = base64.b64decode(image_base64)

            # Generate filename
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
            filename = f"sd_{safe_prompt}_{i:02d}_{seed}.png"
            file_path = cache_dir / filename

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(file_path)
            logger.info(f"SD image {i + 1}/{len(result_data['images'])} saved: {file_path}")

        logger.info(f"SD generation complete. Generated {len(image_paths)} images.")
        return image_paths

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Stable Diffusion API at {SD_API_URL}")
        raise RuntimeError(f"Cannot connect to Stable Diffusion API at {SD_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Stable Diffusion API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Stable Diffusion API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error generating SD images: {e}")
        raise RuntimeError(f"Failed to generate SD images: {e!s}")


# --- Fal AI FLUX 2 Image Generation ---


async def generate_flux2_image(
    prompt: str,
    image_size: str | dict[str, int] = "landscape_4_3",
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    num_images: int = 1,
    seed: int | None = None,
    acceleration: str = "regular",
    enable_prompt_expansion: bool = False,
    enable_safety_checker: bool = True,
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Generate images using FLUX.2 [dev] from Black Forest Labs via Fal AI.

    Args:
        prompt: Text description of the image to generate
        image_size: Size preset or dict with width/height (512-2048).
                   Presets: square_hd, square, portrait_4_3, portrait_16_9,
                   landscape_4_3, landscape_16_9
        num_inference_steps: Number of inference steps (1-100, default: 28)
        guidance_scale: How closely to follow the prompt (1.0-20.0, default: 2.5)
        num_images: Number of images to generate (1-4)
        seed: Random seed for reproducibility (None for random)
        acceleration: Speed vs quality tradeoff (none, regular, high)
        enable_prompt_expansion: Whether to expand the prompt for better results
        enable_safety_checker: Whether to enable NSFW filtering
        output_format: Output format (jpeg, png, webp)

    Returns:
        Dictionary containing:
        - images: List of dicts with url, width, height
        - seed: Seed used for generation
        - prompt: The prompt used (may be expanded)
        - has_nsfw_concepts: List of booleans for each image

    Raises:
        RuntimeError: If API call fails
    """
    logger.info(
        f"FLUX 2: Generating image(s) with prompt='{prompt[:100]}...', "
        f"size={image_size}, steps={num_inference_steps}, guidance={guidance_scale}, "
        f"num_images={num_images}, acceleration={acceleration}"
    )

    # Build the request payload
    request_input: dict[str, Any] = {
        "prompt": prompt,
        "image_size": image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "acceleration": acceleration,
        "enable_prompt_expansion": enable_prompt_expansion,
        "enable_safety_checker": enable_safety_checker,
        "output_format": output_format,
    }

    # Only include seed if explicitly provided
    if seed is not None:
        request_input["seed"] = seed

    try:
        # Use fal_client.subscribe for async queue handling
        result = await asyncio.to_thread(
            fal_client.subscribe,
            "fal-ai/flux-2",
            arguments=request_input,
        )

        logger.info(f"FLUX 2: Generation complete, received {len(result.get('images', []))} image(s)")
        return result

    except Exception as e:
        logger.exception(f"FLUX 2 API error: {e}")
        raise RuntimeError(f"FLUX 2 generation failed: {e!s}")


# --- Z-Image-Turbo Image Generation ---


async def generate_zimage(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 9,
    seed: int = 42,
    use_oot_lora: bool = False,
    oot_lora_scale: float = 0.8,
    use_hk_lora: bool = False,
    hk_lora_scale: float = 0.8,
    use_mannequin_lora: bool = False,
    mannequin_lora_scale: float = 0.8,
) -> Path:
    """
    Generate an image using the Z-Image-Turbo API.

    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        num_inference_steps: Number of denoising steps (default: 9)
        seed: Random seed for reproducibility (default: 42)
        use_oot_lora: Enable the OOT64 LoRA adapter (default: False)
        oot_lora_scale: OOT64 LoRA weight/scale 0.0-2.0 (default: 0.8)
        use_hk_lora: Enable the HK (Hollow Knight) LoRA adapter (default: False)
        hk_lora_scale: HK LoRA weight/scale 0.0-2.0 (default: 0.8)
        use_mannequin_lora: Enable the Mannequin LoRA adapter (default: False)
        mannequin_lora_scale: Mannequin LoRA weight/scale 0.0-2.0 (default: 0.8)

    Returns:
        Path to the saved PNG image file

    Raises:
        RuntimeError: If API call fails or returns invalid data
    """
    lora_parts = []
    if use_oot_lora:
        lora_parts.append(f"oot={oot_lora_scale}")
    if use_hk_lora:
        lora_parts.append(f"hk={hk_lora_scale}")
    if use_mannequin_lora:
        lora_parts.append(f"mannequin={mannequin_lora_scale}")
    lora_info = f", lora=[{', '.join(lora_parts)}]" if lora_parts else ""
    logger.info(
        f"Z-Image: Generating image with prompt='{prompt[:50]}...', "
        f"size={width}x{height}, steps={num_inference_steps}, seed={seed}{lora_info}"
    )

    params = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "use_oot_lora": use_oot_lora,
        "oot_lora_weight": oot_lora_scale,
        "use_hk_lora": use_hk_lora,
        "hk_lora_weight": hk_lora_scale,
        "use_mannequin_lora": use_mannequin_lora,
        "mannequin_lora_weight": mannequin_lora_scale,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(f"{ZIMAGE_API_URL}/generate", params=params)
            response.raise_for_status()

        # API returns PNG directly, save to cache
        cache_dir: Path = Path(".cache/zimage_generated")
        cache_dir.mkdir(parents=True, exist_ok=True)

        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
        filename = f"zimage_{safe_prompt}_{seed}.png"
        file_path = cache_dir / filename

        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Z-Image: Image saved to {file_path}")
        return file_path

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Z-Image API at {ZIMAGE_API_URL}")
        raise RuntimeError(f"Cannot connect to Z-Image API at {ZIMAGE_API_URL}. Please check if it's running.")
    except httpx.HTTPStatusError as e:
        logger.exception(f"Z-Image API HTTP error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"Z-Image API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.exception(f"Error generating Z-Image: {e}")
        raise RuntimeError(f"Failed to generate Z-Image: {e!s}")
