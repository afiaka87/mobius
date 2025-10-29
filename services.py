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
import random
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
from utils import (  # Assuming these are correctly defined in utils.py
    convert_audio_to_waveform_video,
    # image_to_base64_url, # Not directly used in this file after refactor, but kept if utils uses it
    # create_mask_with_alpha, # Not directly used here, gptimg command prepares mask
)
from tasks import TaskProgress, simple_poll_task

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
        logger.info(
            f"Requesting OpenAI chat completion: model={model_name}, num_messages={len(messages)}, seed={seed}"
        )
        completion = client.chat.completions.create(**api_args)

        if completion.choices and completion.choices[0].message:
            response_content: str | None = completion.choices[0].message.content
            if response_content is not None:
                logger.info(
                    f"OpenAI chat completion successful for model {model_name}."
                )
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
        logger.exception(
            f"OpenAI API status error (code {e.status_code}): {e.response}"
        )
        raise
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred during OpenAI chat completion: {e}"
        )
        raise


# --- OpenAI Speech Generation Service ---
# Define a type alias for voice options to improve readability
VoiceType = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


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

    logger.info(
        f"Converting speech audio at {speech_file_path} to video at {video_file_path}"
    )
    # The convert_audio_to_waveform_video function is from utils.py
    # and is assumed to handle its own errors or let them propagate.
    convert_audio_to_waveform_video(
        audio_file=str(speech_file_path), video_file=str(video_file_path)
    )
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

    logger.info(
        f"Requesting Anthropic completion: model={model}, prompt='{prompt[:50]}...'"
    )
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
        logger.exception(
            f"Anthropic API error for model {model}, prompt '{prompt[:50]}...': {e}"
        )
        raise


# --- Fal AI Image Generation Services ---
async def generate_flux_image(
    prompt: str,
    model: str,  # e.g., "fal-ai/flux/schnell"
    image_size: str,  # e.g., "square_hd"
    guidance_scale: float,
) -> str:
    """
    Generates an image using Fal AI's FLUX models.

    Args:
        prompt: The text prompt for image generation.
        model: The specific Fal AI FLUX model to use.
        image_size: The desired image size.
        guidance_scale: The guidance scale for generation.

    Returns:
        The URL of the generated image.

    Raises:
        fal_client.FalClientError: If the Fal AI client encounters an error.
        KeyError: If the response structure is unexpected.
    """
    logger.info(
        f"Requesting Fal AI FLUX image: model={model}, size={image_size}, prompt='{prompt[:50]}...'"
    )
    try:
        # fal_client.submit is synchronous, but we're in an async function.
        # If fal_client has an async version, it would be preferred.
        # For now, assuming it's acceptable or handled by an executor elsewhere if truly blocking.
        # However, the command uses `await services.generate_flux_image`,
        # implying this should be async or run in executor.
        # The `fal_client.submit` returns a handler, then `handler.get()` blocks.
        # `fal_client.subscribe_async` is available and should be used for async.

        # Let's assume fal_client.submit is okay for now, or refactor if it blocks.
        # A better pattern if `submit` is blocking:
        # loop = asyncio.get_running_loop()
        # result = await loop.run_in_executor(None, fal_client.submit(...).get)

        # Using subscribe_async as it's more idiomatic for async contexts
        result: dict[str, Any] = await fal_client.subscribe_async(
            model,  # Model ID like "fal-ai/flux/schnell"
            arguments={
                "prompt": prompt,
                "image_size": image_size,
                "guidance_scale": guidance_scale,
            },
        )
        image_url: str = result["images"][0]["url"]
        logger.info(f"Fal AI FLUX image generated successfully: {image_url}")
        return image_url
    except Exception as e:  # Catch generic fal_client errors or KeyErrors
        logger.exception(
            f"Fal AI FLUX image generation error for prompt '{prompt[:50]}...': {e}"
        )
        raise


# Note: flux_img_to_img was in the original services.py but not used by any commands.
# If it's needed, it can be refactored similarly. For now, omitting to keep focused.


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
            logger.info(
                f"Google search successful for '{query}', found {len(links)} links."
            )
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
    headers: dict[str, str] = {
        "User-Agent": "MobiusDiscordBot/1.0 (github.com/afiaka87/mobius)"
    }

    logger.info(f"Fetching NWS gridpoint for coordinates: {lat}, {lon}")
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            # 1. Get gridpoint URL
            points_url: str = f"https://api.weather.gov/points/{lat},{lon}"
            response_points = await client.get(points_url)
            response_points.raise_for_status()
            points_data: dict[str, Any] = response_points.json()
            forecast_hourly_url: str = points_data.get("properties", {}).get(
                "forecastHourly"
            )

            if not forecast_hourly_url:
                logger.error(
                    f"Could not retrieve hourly forecast URL from NWS API for {lat},{lon}"
                )
                raise ValueError("NWS API did not return a valid hourly forecast URL.")

            # 2. Get hourly forecast
            logger.info(f"Fetching hourly forecast from: {forecast_hourly_url}")
            response_forecast = await client.get(forecast_hourly_url)
            response_forecast.raise_for_status()
            forecast_data: dict[str, Any] = response_forecast.json()

            periods: list[dict[str, Any]] = forecast_data.get("properties", {}).get(
                "periods", []
            )
            if not periods:
                logger.warning(
                    f"NWS API returned no forecast periods for {forecast_hourly_url}"
                )
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
            wind_chill_value: Any | None = current_period.get(
                "windChill"
            )  # This might be a dict or simple value

            if temperature is None or temp_unit is None:
                return "Temperature data is currently unavailable."

            result_str: str = (
                f"Current temperature in Fayetteville, AR: {temperature}°{temp_unit}"
            )
            # Attempt to parse wind chill if it's a simple numeric value
            # This part is speculative based on typical API structures; adjust if NWS is different.
            if isinstance(wind_chill_value, int | float):
                result_str += f" with wind chill of {wind_chill_value}°{temp_unit}"
            elif (
                isinstance(wind_chill_value, dict)
                and "value" in wind_chill_value
                and isinstance(wind_chill_value["value"], int | float)
            ):
                result_str += (
                    f" with wind chill of {wind_chill_value['value']}°{temp_unit}"
                )

            logger.info(f"Successfully fetched weather: {result_str}")
            return result_str

        except httpx.HTTPStatusError as e:
            logger.exception(
                f"NWS API HTTP error: {e.response.status_code} - {e.response.text}"
            )
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
                logger.error(
                    f"YouTube API response missing videoId for query '{search_query}': {top_result}"
                )
                return {
                    "error": "Malformed response from YouTube API (missing videoId)"
                }

            logger.info(
                f"YouTube search successful for '{search_query}', found videoId: {video_info['videoId']}"
            )
            return video_info

        except httpx.HTTPStatusError as e:
            logger.exception(
                f"YouTube API HTTP error for query '{search_query}': {e.response.status_code} - {e.response.text}"
            )
            return {"error": f"YouTube API error: {e.response.status_code}"}
        except Exception as e:
            logger.exception(
                f"Error during YouTube search for query '{search_query}': {e}"
            )
            return {"error": "An unexpected error occurred during YouTube search."}


# --- ComfyUI Text-to-Video Service ---
async def t2v(text: str, length: int = 33, steps: int = 30, seed: int = 0) -> Path:
    """
    Generates a video from text using a ComfyUI workflow.

    Args:
        text: The text prompt for video generation.
        length: The number of frames in the video.
        steps: The number of diffusion steps.
        seed: The seed for generation (0 for random).

    Returns:
        The file path to the generated video.

    Raises:
        ValueError: If ComfyUI API URL is not configured or if ComfyAPIWrapper is not available.
        Exception: If the ComfyUI API call fails.
    """
    if ComfyApiWrapper is Any or ComfyWorkflowWrapper is Any:
        logger.error("ComfyUI libraries (comfy_api_simplified) are not available.")
        raise ValueError("ComfyUI integration is not properly installed or imported.")

    comfyui_api_url: str | None = os.getenv("COMFYUI_API_URL")
    if not comfyui_api_url:
        logger.error("COMFYUI_API_URL not found in environment variables.")
        raise ValueError("ComfyUI API URL is not configured.")

    # nest_asyncio is used here as it was in the original code.
    # This might be needed if ComfyApiWrapper or its dependencies
    # have issues with an already running asyncio loop.
    import nest_asyncio

    nest_asyncio.apply()

    api: ComfyApiWrapper = ComfyApiWrapper(comfyui_api_url)
    # Ensure the workflow JSON path is correct relative to the project root
    workflow_path: Path = Path("workflows/t2v.json")
    if not workflow_path.exists():
        logger.error(f"ComfyUI workflow file not found at {workflow_path}")
        raise FileNotFoundError(f"ComfyUI workflow file not found: {workflow_path}")

    workflow: ComfyWorkflowWrapper = ComfyWorkflowWrapper(str(workflow_path))

    logger.info(
        f"Requesting ComfyUI t2v: prompt='{text[:50]}...', length={length}, steps={steps}, seed={seed}"
    )

    # Set workflow parameters
    workflow.set_node_param("CLIP Text Encode (Positive Prompt)", "text", text)
    workflow.set_node_param("EmptyHunyuanLatentVideo", "length", length)
    workflow.set_node_param("KSampler", "steps", steps)
    workflow.set_node_param(
        "KSampler", "seed", seed if seed != 0 else random.randint(1, 2**32 - 1)
    )

    try:
        # Assuming queue_and_wait_images returns Dict[filename_str, image_bytes]
        results: dict[str, bytes] = api.queue_and_wait_images(
            workflow, "SaveAnimatedWEBP"
        )

        if not results:
            logger.error("ComfyUI t2v returned no results.")
            raise RuntimeError("ComfyUI t2v process did not yield any output.")

        # Process the first result (assuming one video output)
        filename_str, video_data = next(iter(results.items()))

        cache_dir: Path = Path(".cache/t2v")
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize filename from ComfyUI if necessary, or use a generated one
        safe_filename: str = "".join(
            c if c.isalnum() or c in [".", "_", "-"] else "_" for c in filename_str
        )
        output_video_path: Path = cache_dir / safe_filename

        with open(output_video_path, "wb") as f:
            f.write(video_data)

        logger.info(f"ComfyUI t2v successful. Video saved to: {output_video_path}")
        return output_video_path
    except Exception as e:
        logger.exception(
            f"Error during ComfyUI t2v process for prompt '{text[:50]}...': {e}"
        )
        raise


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
        safe_prompt_suffix: str = "".join(
            c if c.isalnum() else "_" for c in prompt[:30]
        )
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
        raise RuntimeError(
            f"An unexpected error occurred during image generation: {e!s}"
        )


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
        logger.warning(
            f"More than 10 images provided ({len(images)}), only first 10 will be used."
        )
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

        result = await loop.run_in_executor(
            None, lambda: client.images.edit(**api_params)
        )

        if not result.data or not result.data[0].b64_json:
            logger.error("GPT Image editing returned no image data.")
            raise RuntimeError("GPT Image editing failed to return image data.")

        image_base64: str = result.data[0].b64_json
        image_bytes: bytes = base64.b64decode(image_base64)

        # Save the edited image as PNG to support transparency
        cache_dir: Path = Path(".cache/gptimg_edited")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt_suffix: str = "".join(
            c if c.isalnum() else "_" for c in prompt[:30]
        )
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
            raise ValueError(
                f"Invalid mask: {error_detail}. Ensure it's a PNG with a proper alpha channel."
            )
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

async def generate_celeste_image(
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    width: int = 512,
    height: int = 512,
) -> Path:
    """
    Generates an image using the Celeste Diffusion API.

    Args:
        prompt: The text prompt for image generation.
        negative_prompt: Optional negative prompt to avoid certain elements.
        num_inference_steps: Number of denoising steps (10-100).
        guidance_scale: How closely to follow the prompt (1.0-20.0).
        seed: Optional seed for reproducible results.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Path to the generated image file.

    Raises:
        httpx.HTTPStatusError: For HTTP errors from the API.
        RuntimeError: For other operational issues.
    """
    celeste_api_url: str = "http://192.168.1.216:8000"

    # Clamp values to valid ranges
    num_inference_steps = min(max(num_inference_steps, 10), 100)
    guidance_scale = min(max(guidance_scale, 1.0), 20.0)

    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "width": width,
        "height": height,
    }

    logger.info(
        f"Requesting Celeste image generation: prompt='{prompt[:50]}...', "
        f"steps={num_inference_steps}, guidance={guidance_scale}"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{celeste_api_url}/generate", json=payload)
            response.raise_for_status()

            data: dict[str, Any] = response.json()

            if "image" not in data:
                logger.error("Celeste API response missing image data")
                raise RuntimeError("Celeste API did not return image data")

            # Decode base64 image
            image_base64: str = data["image"]
            image_bytes: bytes = base64.b64decode(image_base64)
            actual_seed: int = data.get("seed", seed or 0)

            # Save the image
            cache_dir: Path = Path(".cache/celeste")
            cache_dir.mkdir(parents=True, exist_ok=True)

            safe_prompt_suffix: str = "".join(
                c if c.isalnum() else "_" for c in prompt[:30]
            )
            filename: str = f"celeste_{safe_prompt_suffix}_{actual_seed}.png"
            file_path: Path = cache_dir / filename

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            logger.info(f"Celeste image generated successfully: {file_path}")
            return file_path

        except httpx.ConnectError:
            logger.exception("Cannot connect to Celeste API server")
            raise RuntimeError(
                "Cannot connect to the Celeste API server. Please check if it's running."
            )
        except httpx.HTTPStatusError as e:
            logger.exception(
                f"Celeste API HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.exception(f"Error generating Celeste image: {e}")
            raise RuntimeError(f"Failed to generate image: {e!s}")



# --- GLIDE API Service ---

async def generate_glide_image(
    prompt: str,
    guidance_scale: float = 4.0,
    base_steps: int = 30,
    sr_steps: int = 30,
    sampler: str = "euler",
    seed: int | None = None,
    output_format: str = "png",
    skip_sr: bool = False,
    batch_size: int = 1,
) -> list[Path]:
    """
    Generates one or more images using the GLIDE API service.

    Args:
        prompt: The text prompt for image generation.
        guidance_scale: Classifier-free guidance scale (0.0-20.0).
        base_steps: Number of sampling steps for base model (1-100).
        sr_steps: Number of sampling steps for super-resolution model (1-100).
        sampler: Sampling method (euler, euler_a, dpm++, plms, ddim).
        seed: Optional seed for reproducible results.
        output_format: Output image format (png, jpg, jpeg).
        skip_sr: Skip super-resolution and return 64x64 image.
        batch_size: Number of images to generate (1-8).

    Returns:
        List of paths to the generated image files (256x256 or 64x64 if skip_sr).

    Raises:
        httpx.HTTPStatusError: For HTTP errors from the API.
        RuntimeError: For other operational issues.
    """
    glide_api_url: str = "http://100.70.95.57:8001"

    # Clamp values to valid ranges
    guidance_scale = min(max(guidance_scale, 0.0), 20.0)
    base_steps = min(max(base_steps, 1), 100)
    sr_steps = min(max(sr_steps, 1), 100)
    batch_size = min(max(batch_size, 1), 8)

    payload: dict[str, Any] = {
        "prompt": prompt,
        "batch_size": batch_size,
        "guidance_scale": guidance_scale,
        "base_steps": base_steps,
        "sr_steps": sr_steps,
        "sampler": sampler,
        "seed": seed,
        "output_format": output_format,
        "use_fp16": False,  # Use full precision for better quality
        "skip_sr": skip_sr,
        "return_all": batch_size > 1,  # Return all images if batch
    }

    logger.info(
        f"Requesting GLIDE image generation: prompt='{prompt[:50]}...', "
        f"batch_size={batch_size}, base_steps={base_steps}, sr_steps={sr_steps}, "
        f"guidance={guidance_scale}, sampler={sampler}"
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{glide_api_url}/generate", json=payload)
            response.raise_for_status()

            # Create cache directory
            cache_dir: Path = Path(".cache/glide")
            cache_dir.mkdir(parents=True, exist_ok=True)

            safe_prompt_suffix: str = "".join(
                c if c.isalnum() else "_" for c in prompt[:30]
            )
            actual_seed: int = seed if seed is not None else 0

            image_paths: list[Path] = []

            if batch_size == 1:
                # Single image returned as bytes
                image_bytes: bytes = response.content
                filename: str = f"glide_{safe_prompt_suffix}_{actual_seed}.{output_format}"
                file_path: Path = cache_dir / filename

                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                image_paths.append(file_path)
                logger.info(f"GLIDE image generated successfully: {file_path}")
            else:
                # Multiple images returned as JSON with base64 data
                import base64
                import json

                response_data = json.loads(response.content)
                images_data = response_data.get("images", [])

                for i, img_data in enumerate(images_data):
                    # Decode base64 image
                    image_bytes = base64.b64decode(img_data)
                    filename = f"glide_{safe_prompt_suffix}_{actual_seed}_{i+1}.{output_format}"
                    file_path = cache_dir / filename

                    with open(file_path, "wb") as f:
                        f.write(image_bytes)

                    image_paths.append(file_path)
                    logger.info(f"GLIDE image {i+1}/{batch_size} generated: {file_path}")

            return image_paths

        except httpx.ConnectError:
            logger.exception("Cannot connect to GLIDE API server")
            raise RuntimeError(
                "Cannot connect to the GLIDE API server. Please check if it's running."
            )
        except httpx.HTTPStatusError as e:
            logger.exception(
                f"GLIDE API HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.exception(f"Error generating GLIDE image: {e}")
            raise RuntimeError(f"Failed to generate image: {e!s}")


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
            return response.json()
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


async def search_obscast_media(
    query: str, media_type: str | None = None, limit: int = 25
) -> list[dict[str, Any]]:
    """Search for media on the Obscast server."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    params = {"q": query, "limit": limit}
    if media_type:
        params["media_type"] = media_type
    response = await obscast_client.get("/search", params=params)
    return response.get("files", [])


async def get_obscast_media_by_id(media_id: str) -> dict[str, Any]:
    """Get a single media file's details from Obscast."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    return await obscast_client.get(f"/media/{media_id}")


async def play_obscast_media(media_id: str) -> dict[str, Any]:
    """Request to play a media file on OBS via Obscast."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    return await obscast_client.post(f"/obs/play/{media_id}")


async def queue_obscast_media(media_id: str) -> dict[str, Any]:
    """Request to queue a media file on OBS via Obscast."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    return await obscast_client.post(f"/obs/queue/{media_id}")


async def get_obscast_current() -> dict[str, Any]:
    """Get the currently playing media from Obscast."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    return await obscast_client.get("/obs/current")


async def get_obscast_queue() -> dict[str, Any]:
    """Get the current OBS queue from Obscast."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    return await obscast_client.get("/obs/queue")


async def control_obscast_playback(action: str) -> dict[str, Any]:
    """Send a playback control action to Obscast (e.g., skip, pause)."""
    if not obscast_client:
        raise ObscastAPIError("Obscast service is not configured.")
    valid_actions = ["skip", "pause", "resume", "stop", "next", "previous"]
    if action not in valid_actions:
        raise ValueError(f"Invalid playback action: {action}")

    if action == "stop":  # Obscast uses pause for stop
        action = "pause"

    return await obscast_client.post(f"/obs/{action}")


# CLIP Retrieval API integration
async def search_clip_images(
    query: str,
    num_images: int = 4,
    index_name: str = "laion-aesthetic-9m-16gb",
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """
    Search for images using the CLIP retrieval backend.

    Args:
        query: Text query to search for
        num_images: Number of images to return (default 4)
        index_name: Name of the index to search
        deduplicate: Whether to remove duplicate results

    Returns:
        List of image results with URLs and metadata

    Raises:
        httpx.HTTPStatusError: For HTTP errors from the API
        RuntimeError: If CLIP API URL is not configured
    """
    clip_api_url = os.getenv("CLIP_API_URL", "http://archer:1234")
    if not clip_api_url:
        raise RuntimeError("CLIP_API_URL is not configured")

    payload = {
        "text": query,
        "modality": "text",
        "num_images": num_images,
        "indice_name": index_name,
        "deduplicate": deduplicate,
        "use_safety_model": True,  # Filter NSFW content
        "aesthetic_score": 7,  # Prefer higher aesthetic quality
        "aesthetic_weight": 0.3,
    }

    logger.info(f"Searching CLIP for: '{query}' (requesting {num_images} images)")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{clip_api_url}/knn-service", json=payload)
            response.raise_for_status()
            results = response.json()

            # Filter results to only include those with valid URLs
            valid_results = [
                result for result in results
                if result.get("url") and result["url"].startswith(("http://", "https://"))
            ]

            logger.info(f"CLIP search returned {len(valid_results)} valid results")
            return valid_results[:num_images]

        except httpx.HTTPStatusError as e:
            logger.exception(
                f"CLIP API HTTP error for query '{query}': {e.response.status_code} - {e.response.text}"
            )
            raise RuntimeError(f"CLIP API error: {e.response.text}") from e
        except httpx.RequestError as e:
            logger.exception(f"CLIP API connection error: {e}")
            raise RuntimeError(f"Could not connect to CLIP API at {clip_api_url}") from e


async def get_clip_indices() -> list[str]:
    """
    Get list of available CLIP indices.

    Returns:
        List of available index names

    Raises:
        RuntimeError: If API is not configured or request fails
    """
    clip_api_url = os.getenv("CLIP_API_URL", "http://archer:1234")
    if not clip_api_url:
        raise RuntimeError("CLIP_API_URL is not configured")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{clip_api_url}/indices-list")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.exception(f"CLIP API HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"CLIP API error: {e.response.text}") from e
        except httpx.RequestError as e:
            logger.exception(f"CLIP API connection error: {e}")
            raise RuntimeError(f"Could not connect to CLIP API at {clip_api_url}") from e


async def generate_stable_lora(
    caption: str,
    lora: str,
    lora_strength: float = 1.0,
    batch_size: int = 1,
    seed: int = 42,
    guidance_scale: float = 7.5
) -> list[bytes]:
    """
    Generate images using Stable Diffusion with specified LoRA.

    Args:
        caption: Text description for the image generation
        lora: LoRA style to use (e.g., 'pixelart', 'trippy')
        lora_strength: Strength of LoRA influence (0.0-2.0, default 1.0)
        batch_size: Number of images to generate (default 1)
        seed: Random seed for reproducible generation (default 42)
        guidance_scale: Guidance scale for generation strength (default 7.5)

    Returns:
        List of generated images as bytes

    Raises:
        RuntimeError: If the API request fails
    """
    lora_api_url = "http://100.70.95.57:8000/lora"

    # Pass caption through as-is (no enhancement)

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(
                f"Generating stable diffusion with LoRA '{lora}' (strength={lora_strength}): "
                f"caption='{caption}', batch_size={batch_size}, seed={seed}, "
                f"guidance_scale={guidance_scale}"
            )

            response = await client.post(
                lora_api_url,
                json={
                    "caption": caption,
                    "lora": lora,
                    "lora_strength": lora_strength,
                    "batch_size": batch_size,
                    "seed": seed,
                    "guidance_scale": guidance_scale
                }
            )
            response.raise_for_status()

            # The API returns images as a list of base64-encoded strings
            result = response.json()
            images = result.get("images", [])

            if not images:
                raise RuntimeError("No images returned from stable diffusion API")

            # Convert base64 strings to bytes
            image_bytes_list = []
            for img_b64 in images:
                # Remove data URL prefix if present
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                image_bytes_list.append(img_bytes)

            logger.info(f"Successfully generated {len(image_bytes_list)} images with LoRA '{lora}'")
            return image_bytes_list

        except httpx.HTTPStatusError as e:
            logger.exception(f"Stable diffusion API HTTP error: {e.response.status_code}")
            error_msg = f"Stable diffusion API error: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                if "detail" in error_detail:
                    error_msg = f"Stable diffusion API error: {error_detail['detail']}"
            except Exception:
                pass
            raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            logger.exception(f"Stable diffusion API connection error: {e}")
            raise RuntimeError(f"Could not connect to stable diffusion API at {lora_api_url}") from e
        except Exception as e:
            logger.exception(f"Unexpected error generating image with LoRA '{lora}': {e}")
            raise RuntimeError(f"Failed to generate image with LoRA '{lora}': {e!s}") from e


async def generate_dalle_blog_sdxl_image(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    output_format: str = "base64",
) -> Path:
    """
    Generates an image using the DALLE-blog SDXL API with hybrid LoRA approach.

    Args:
        prompt: The text prompt for image generation.
        negative_prompt: Optional negative prompt to avoid certain elements.
        width: Image width in pixels (default: 1024).
        height: Image height in pixels (default: 1024).
        num_inference_steps: Number of denoising steps (default: 25).
        guidance_scale: How closely to follow the prompt (default: 7.5).
        seed: Random seed for reproducible results (default: 42).
        output_format: Output format - "base64", "png", or "jpeg" (default: "base64").

    Returns:
        Path to the generated image file.

    Raises:
        httpx.HTTPStatusError: For HTTP errors from the API.
        RuntimeError: For other operational issues.
    """
    dalle_blog_api_url: str = "http://100.70.95.57:8002/generate"

    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_format": output_format,
    }

    logger.info(
        f"Requesting DALLE-blog SDXL image generation: prompt='{prompt[:50]}...', "
        f"steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}"
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(dalle_blog_api_url, json=payload)
            response.raise_for_status()

            data: dict[str, Any] = response.json()

            # Handle base64 response format
            if output_format == "base64":
                if "image" not in data:
                    logger.error("DALLE-blog SDXL API response missing image data")
                    raise RuntimeError("DALLE-blog SDXL API did not return image data")

                # Decode base64 image
                image_base64: str = data["image"]
                # Remove data URL prefix if present
                if "," in image_base64:
                    image_base64 = image_base64.split(",", 1)[1]
                image_bytes: bytes = base64.b64decode(image_base64)
            else:
                # For png/jpeg format, the response might be binary
                image_bytes = response.content

            # Save the image
            cache_dir: Path = Path(".cache/dalle_blog_sdxl")
            cache_dir.mkdir(parents=True, exist_ok=True)

            safe_prompt_suffix: str = "".join(
                c if c.isalnum() else "_" for c in prompt[:30]
            )
            filename: str = f"dalle_blog_sdxl_{safe_prompt_suffix}_{seed}.png"
            file_path: Path = cache_dir / filename

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            logger.info(f"DALLE-blog SDXL image generated successfully: {file_path}")
            return file_path

        except httpx.ConnectError:
            logger.exception("Cannot connect to DALLE-blog SDXL API server")
            raise RuntimeError(
                "Cannot connect to the DALLE-blog SDXL API server. Please check if it's running on port 8002."
            )
        except httpx.HTTPStatusError as e:
            logger.exception(
                f"DALLE-blog SDXL API HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise RuntimeError(
                f"DALLE-blog SDXL API error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            logger.exception(f"Error generating DALLE-blog SDXL image: {e}")
            raise RuntimeError(f"Failed to generate DALLE-blog SDXL image: {e!s}")


async def generate_sdxl_pixelart_image(
    prompt: str | list[str],
    negative_prompt: str | list[str] | None = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
    seed: int | None = None,
    lora_weight: float = 1.0,
    scheduler: str = "euler",
    api_url: str = "http://localhost:8000",
) -> list[Path]:
    """
    Generates images using the SDXL API with LoRA support (pixel art).

    This service supports three operating modes:
    1. Single prompt, multiple images: Generate variations from one prompt
    2. Multiple prompts, one image each: Batch generation of diverse images
    3. Single prompt, single image: Basic generation

    Args:
        prompt: Single text prompt or list of prompts for image generation.
        negative_prompt: Single negative prompt or list to guide what NOT to generate.
        width: Output image width in pixels (256-2048, default: 1024).
        height: Output image height in pixels (256-2048, default: 1024).
        num_inference_steps: Denoising steps (1-150, default: 30, more=higher quality but slower).
        guidance_scale: Classifier-free guidance (1.0-20.0, default: 7.5, higher=follows prompt more strictly).
        num_images_per_prompt: How many images per prompt (1-4, default: 1, only for single prompt).
        seed: Random seed for reproducible results (None=random seed).
        lora_weight: LoRA influence strength (0.0-2.0, default: 1.0, 0.0=disabled, 1.0=normal, >1.0=amplified).
        scheduler: Diffusion noise scheduler ("euler" or "ddim", default: "euler").
        api_url: Base URL of the SDXL API server (default: "http://localhost:8000").

    Returns:
        List of paths to the generated image files.

    Raises:
        httpx.HTTPStatusError: For HTTP errors from the API.
        RuntimeError: For other operational issues.
        ValueError: For invalid parameter combinations.
    """
    # Validate parameters
    if isinstance(prompt, list) and len(prompt) > 1 and num_images_per_prompt > 1:
        logger.warning(
            "Multiple prompts with num_images_per_prompt > 1 is not supported. "
            "Setting num_images_per_prompt to 1."
        )
        num_images_per_prompt = 1

    # Clamp values to valid ranges
    width = min(max(width, 256), 2048)
    height = min(max(height, 256), 2048)
    num_inference_steps = min(max(num_inference_steps, 1), 150)
    guidance_scale = min(max(guidance_scale, 1.0), 20.0)
    num_images_per_prompt = min(max(num_images_per_prompt, 1), 4)
    lora_weight = min(max(lora_weight, 0.0), 2.0)

    if scheduler not in ["euler", "ddim"]:
        raise ValueError(f"scheduler must be 'euler' or 'ddim', got '{scheduler}'")

    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_images_per_prompt,
        "seed": seed,
        "lora_weight": lora_weight,
        "scheduler": scheduler,
    }

    prompt_preview = prompt if isinstance(prompt, str) else f"[{len(prompt)} prompts]"
    logger.info(
        f"Requesting SDXL pixel art image generation: prompt='{str(prompt_preview)[:50]}...', "
        f"steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}, "
        f"lora_weight={lora_weight}, num_images={num_images_per_prompt}"
    )

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(f"{api_url}/generate", json=payload)
            response.raise_for_status()

            data: dict[str, Any] = response.json()

            if "images" not in data or not data["images"]:
                logger.error("SDXL API response missing image data")
                raise RuntimeError("SDXL API did not return any images")

            # Create cache directory
            cache_dir: Path = Path(".cache/sdxl_pixelart")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Decode and save all images
            image_paths: list[Path] = []
            metadata = data.get("metadata", {})
            # Handle case where API returns None for seed
            actual_seed = metadata.get("seed") or seed or 0

            for i, img_base64 in enumerate(data["images"]):
                # Decode base64 image
                image_bytes: bytes = base64.b64decode(img_base64)

                # Generate filename
                if isinstance(prompt, str):
                    safe_prompt_suffix = "".join(
                        c if c.isalnum() else "_" for c in prompt[:30]
                    )
                else:
                    safe_prompt_suffix = f"batch_{len(prompt)}_prompts"

                # Include image index and seed in filename
                # Ensure actual_seed is always an int
                seed_value = int(actual_seed) if actual_seed is not None else 0
                img_seed = seed_value + i if isinstance(prompt, str) else seed_value
                filename = f"sdxl_pixelart_{safe_prompt_suffix}_{img_seed}_{i+1}.png"
                file_path = cache_dir / filename

                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                image_paths.append(file_path)
                logger.info(f"SDXL pixel art image {i+1}/{len(data['images'])} saved: {file_path}")

            logger.info(
                f"SDXL pixel art generation complete. Generated {len(image_paths)} images. "
                f"Metadata: {metadata}"
            )
            return image_paths

        except httpx.ConnectError:
            logger.exception(f"Cannot connect to SDXL API server at {api_url}")
            raise RuntimeError(
                f"Cannot connect to the SDXL API server at {api_url}. Please check if it's running."
            )
        except httpx.HTTPStatusError as e:
            logger.exception(
                f"SDXL API HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise RuntimeError(
                f"SDXL API error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            logger.exception(f"Error generating SDXL pixel art image: {e}")
            raise RuntimeError(f"Failed to generate SDXL pixel art image: {e!s}")


# --- Kandinsky-5 Video Generation Services ---
async def check_kandinsky5_health() -> bool:
    """
    Check if the Kandinsky-5 API is healthy and available.

    Returns:
        True if the API is healthy, False otherwise
    """
    api_url = "http://100.70.95.57:8888"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_url}/health")
            response.raise_for_status()
            data = response.json()

            # Check if pipeline is initialized
            is_healthy = data.get("status") == "healthy" and data.get("pipeline_initialized", False)

            if is_healthy:
                logger.info("Kandinsky-5 API health check: OK")
            else:
                logger.warning(f"Kandinsky-5 API health check: DEGRADED - {data}")

            return is_healthy

    except httpx.ConnectError:
        logger.warning(f"Kandinsky-5 API health check: Cannot connect to {api_url}")
        return False
    except httpx.TimeoutException:
        logger.warning(f"Kandinsky-5 API health check: Connection timeout to {api_url}")
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
        duration: Duration of the video in seconds (must be 5 or 10)
        width: Video width in pixels (512 or 768, default: 512)
        height: Video height in pixels (512 or 768, default: 512)
        num_steps: Number of inference steps (default: 50)
        guidance_weight: Optional CFG weight (default: None, uses model default)
        scheduler_scale: Flow matching scheduler scale (default: 5.0)
        seed: Optional random seed for reproducible results
        progress_callback: Optional async callback function(TaskProgress) for progress updates

    Returns:
        Path to the saved video file

    Raises:
        ValueError: If duration is not 5 or 10, or if width/height are invalid
        RuntimeError: If API call fails or returns invalid data
    """
    api_url = "http://100.70.95.57:8888"

    # Validate duration
    if duration not in [5, 10]:
        raise ValueError(f"Duration must be 5 or 10 seconds, got {duration}")

    # Validate resolution (API supports 512 or 768 for each dimension)
    if width not in [512, 768]:
        raise ValueError(f"Width must be 512 or 768, got {width}")
    if height not in [512, 768]:
        raise ValueError(f"Height must be 512 or 768, got {height}")

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
            response = await client.post(f"{api_url}/generate", json=payload)
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
            base_url=api_url,
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
        logger.exception(f"Cannot connect to Kandinsky-5 API at {api_url}")
        raise RuntimeError(
            f"Cannot connect to Kandinsky-5 API at {api_url}. Please check if it's running."
        )
    except httpx.HTTPStatusError as e:
        logger.exception(
            f"Kandinsky-5 API HTTP error: {e.response.status_code} - {e.response.text}"
        )
        raise RuntimeError(
            f"Kandinsky-5 API error: {e.response.status_code} - {e.response.text}"
        )
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
        duration: Duration of each video in seconds (must be 5 or 10)
        width: Video width in pixels (512 or 768, default: 512)
        height: Video height in pixels (512 or 768, default: 512)
        num_steps: Number of inference steps (default: 50)
        guidance_weight: Optional CFG weight (default: None, uses model default)
        scheduler_scale: Flow matching scheduler scale (default: 5.0)
        seed: Optional starting seed for batch (auto-increments for each video)
        progress_callback: Optional async callback function(TaskProgress) for progress updates

    Returns:
        List of paths to the saved video files

    Raises:
        ValueError: If duration is not 5 or 10, or if width/height are invalid
        RuntimeError: If API call fails or returns invalid data
    """
    api_url = "http://100.70.95.57:8888"

    # Validate duration
    if duration not in [5, 10]:
        raise ValueError(f"Duration must be 5 or 10 seconds, got {duration}")

    # Validate resolution
    if width not in [512, 768]:
        raise ValueError(f"Width must be 512 or 768, got {width}")
    if height not in [512, 768]:
        raise ValueError(f"Height must be 512 or 768, got {height}")

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
            response = await client.post(f"{api_url}/generate/batch", json=payload)
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
            base_url=api_url,
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
            logger.info(f"Kandinsky-5 batch video {i+1}/{len(result_data['videos'])} saved: {file_path}")

        logger.info(
            f"Kandinsky-5 batch generation complete. Generated {len(video_paths)} videos."
        )
        return video_paths

    except httpx.ConnectError:
        logger.exception(f"Cannot connect to Kandinsky-5 API at {api_url}")
        raise RuntimeError(
            f"Cannot connect to Kandinsky-5 API at {api_url}. Please check if it's running."
        )
    except httpx.HTTPStatusError as e:
        logger.exception(
            f"Kandinsky-5 API HTTP error: {e.response.status_code} - {e.response.text}"
        )
        raise RuntimeError(
            f"Kandinsky-5 API error: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.exception(f"Error generating Kandinsky-5 batch videos: {e}")
        raise RuntimeError(f"Failed to generate Kandinsky-5 batch videos: {e!s}")
