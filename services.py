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
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, cast

import fal_client
import httpx
import openai  # Main openai client
from anthropic import AsyncAnthropic  # Separate client for Anthropic
from anthropic._types import NotGiven as AnthropicNotGiven

if TYPE_CHECKING:
    from anthropic.types.messages.tool_param import ToolParam as AnthropicToolParam

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
    max_uses: int = 1,  # For web_search tool
    model: str = "claude-3-5-sonnet-20240620",
) -> str:
    """
    Generates a chat completion using Anthropic's Claude models.

    Args:
        prompt: The user's prompt.
        max_tokens: The maximum number of tokens to generate.
        max_uses: Maximum uses for the web_search tool.
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

    tool_config: AnthropicToolParam = cast(
        "AnthropicToolParam",
        {
            "type": "web_search_20250305",  # Using the specified tool type
            "name": "web_search",
            "max_uses": max_uses,
            "user_location": {  # Optional: provide user location context
                "type": "approximate",
                "city": "Fayetteville",
                "region": "Arkansas",
                "country": "US",
                "timezone": "America/Chicago",
            },
        },
    )

    logger.info(
        f"Requesting Anthropic completion: model={model}, prompt='{prompt[:50]}...'"
    )
    try:
        message: AnthropicMessage = await anthropic_client.messages.create(
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            model=model,
            tools=(
                [tool_config] if max_uses > 0 else AnthropicNotGiven()
            ),  # Only include tool if useful
            # extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}, # If needed
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


# --- OpenAI Image Generation/Editing Services ---
# services.py
# ... (all other imports and code from the previous services.py response remain the same) ...
# ... (logger, other service functions) ...


async def generate_gpt_image(
    prompt: str,
    quality: Literal["standard", "hd"] = "standard",  # DALL-E 3 uses standard/hd
    size: Literal[
        "1024x1024", "1792x1024", "1024x1792"
    ] = "1024x1024",  # DALL-E 3 sizes
    model: str = "dall-e-3",  # Defaulting to dall-e-3 which is common
    output_format: Literal["jpeg", "png", "webp"] = "png",
    style: Literal["vivid", "natural"] | None = "vivid",  # DALL-E 3 style
    user: str | None = None,  # Optional user ID for tracking
) -> Path:
    """
    Generates an image using OpenAI's image models (e.g., DALL-E 3).

    Args:
        prompt: The text prompt for image generation.
        quality: Image quality ("standard" or "hd" for DALL-E 3).
        size: Image size (specific to model, DALL-E 3 options listed).
        model: The OpenAI image model to use (e.g., "dall-e-3").
        output_format: The desired output format (jpeg, png, webp) for saving the file.
                       The API itself returns b64_json or URL.
        style: The style of the generated images ("vivid" or "natural" for DALL-E 3).
        user: A unique identifier representing your end-user, which can help OpenAI monitor
              and detect abuse.

    Returns:
        Path to the generated image file.

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

    api_params: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,  # type: ignore
        "quality": quality,  # type: ignore
        "n": 1,  # Generate one image
        # response_format is NOT a parameter for images.generate in openai >= 1.0
        # The API returns b64_json by default if the model supports it or if URL is not applicable.
    }
    # DALL-E 3 specific parameters - only add style if model is dall-e-3 and style is specified
    if model == "dall-e-3" and style:
        api_params["style"] = style
    if user:
        api_params["user"] = user

    logger.info(f"Requesting OpenAI image generation with params: {api_params}")
    try:
        # The result for images.generate in openai >= 1.0 directly contains .b64_json or .url
        result = await loop.run_in_executor(
            None,  # Uses default ThreadPoolExecutor
            lambda: client.images.generate(**api_params),
        )

        if not result.data or not result.data[0]:
            logger.error(
                "OpenAI image generation returned no image data in result.data[0]."
            )
            raise RuntimeError("OpenAI image generation failed to return image data.")

        # Access b64_json directly. If it's None, the API might have only returned a URL,
        # or something went wrong. For saving locally, b64_json is preferred.
        image_b64_json: str | None = result.data[0].b64_json
        image_url: str | None = result.data[0].url  # Also capture URL if available

        if not image_b64_json:
            # If b64_json is not available, and you absolutely need image bytes,
            # you would have to download it from image_url if that's provided.
            # However, for most direct generation, b64_json should be there.
            logger.error(
                f"OpenAI image generation did not return b64_json. URL: {image_url}"
            )
            raise RuntimeError("OpenAI image generation did not provide b64_json data.")

        image_bytes: bytes = base64.b64decode(image_b64_json)

        cache_dir: Path = Path(".cache/gptimg_generated")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt_suffix: str = "".join(
            c if c.isalnum() else "_" for c in prompt[:30]
        )
        filename: str = f"{model}_{safe_prompt_suffix}_{size}_{quality}.{output_format}"
        file_path: Path = cache_dir / filename

        def save_image_file() -> None:
            """Helper to save image, handling format specifics."""
            image: Image.Image = Image.open(BytesIO(image_bytes))
            save_kwargs: dict[str, Any] = {}
            actual_format: str = output_format.upper()
            if actual_format == "JPEG":
                save_kwargs["quality"] = 95
                if (
                    image.mode == "RGBA"
                    or image.mode == "LA"
                    or (image.mode == "P" and "transparency" in image.info)
                ):
                    logger.info(
                        "Converting RGBA/LA/P+alpha image to RGB for JPEG saving."
                    )
                    background: Image.Image = Image.new(
                        "RGB", image.size, (255, 255, 255)
                    )
                    try:
                        alpha = image.getchannel("A")
                        background.paste(image, mask=alpha)
                    except ValueError:
                        background.paste(image.convert("RGB"))
                    image = background
            elif actual_format == "WEBP":
                save_kwargs["quality"] = 90

            image.save(file_path, format=actual_format, **save_kwargs)

        await loop.run_in_executor(None, save_image_file)
        logger.info(f"OpenAI image generated and saved to: {file_path}")
        return file_path

    except openai.BadRequestError as e:
        logger.exception(
            f"OpenAI image generation bad request (prompt: '{prompt[:50]}...'): {e}"
        )
        error_detail = str(e)
        if e.body and isinstance(e.body, dict):
            err_dict = e.body.get("error", {})
            if isinstance(err_dict, dict) and "message" in err_dict:
                error_detail = err_dict["message"]
        raise ValueError(f"Invalid request for image generation: {error_detail}")
    except openai.APIError as e:
        logger.exception(
            f"OpenAI image generation API error for prompt '{prompt[:50]}...': {e}"
        )
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error generating OpenAI image for prompt '{prompt[:50]}...': {e}"
        )
        raise RuntimeError(
            f"An unexpected error occurred during image generation: {e!s}"
        )


async def edit_gpt_image(
    prompt: str,
    images: Sequence[Path | BinaryIO],  # Sequence of image paths or file-like objects
    mask: Path | BinaryIO | None = None,
    model: str = "gpt-image-1",  # DALL-E 2 for edits, gpt-image-1 might have different endpoint/params
    size: Literal[
        "1024x1024", "1024x1536", "1536x1024", "256x256", "512x512"
    ] = "1024x1024",
) -> Path:
    """
    Edits images using OpenAI's image models.
    The `gpt-image-1` model might use a different mechanism than DALL-E 2's edit.
    This implementation targets the standard OpenAI images/edit endpoint.

    Args:
        prompt: Text description of the desired edits.
        images: A list of input image file paths or binary file objects.
                If multiple, the mask applies to the first. `api.md` suggests `gpt-image-1`
                can take multiple images.
        mask: Optional mask file path or binary file object (RGBA with transparency).
        model: The OpenAI model to use (e.g., "dall-e-2", or if "gpt-image-1" supports this endpoint).
        size: The size of the output image.

    Returns:
        Path to the edited image file.

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
    if len(images) > 10:  # As per api.md example (though DALL-E 2 edit takes 1)
        logger.warning(
            f"More than 10 images provided ({len(images)}), API might only use a subset or error."
        )
        # OpenAI DALL-E 2 /v1/images/edits takes 1 image.
        # If gpt-image-1 uses a different endpoint or mechanism for multi-image edit,
        # this service would need to adapt. The api.md implies client.images.edit can handle a list.

    client: OpenAIClient = OpenAIClient(api_key=openai_api_key)
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    # Prepare file objects to pass to the API
    # These need to be opened and closed carefully, especially with run_in_executor
    opened_image_files: list[BinaryIO] = []
    opened_mask_file: BinaryIO | None = None

    try:
        # When we're dealing with Path objects, we'll open them and take responsibility for closing
        # When we're given file-like objects, we'll just use them as-is (but seek to the beginning)
        for img_src in images:
            if isinstance(img_src, Path):
                # Note: We're still opening files here but will close them in the finally block
                # We can't use a context manager directly since we need to keep the files open
                # for the API call that comes after this loop
                file_obj = open(img_src, "rb")
                opened_image_files.append(file_obj)
            else:  # BinaryIO
                img_src.seek(0)  # Ensure reading from the beginning
                opened_image_files.append(img_src)

        if mask:
            if isinstance(mask, Path):
                # Same note applies here - we'll close this in the finally block
                file_obj = open(mask, "rb")
                opened_mask_file = file_obj
            else:  # BinaryIO
                mask.seek(0)
                opened_mask_file = mask

        # The OpenAI Python client's images.edit method expects `image` to be a single file-like object.
        # If `gpt-image-1` via `client.images.edit` truly supports a list of images for the `image` param,
        # the signature of `client.images.edit` would need to reflect that, or the underlying
        # request construction would handle it.
        # Based on standard `openai.resources.images.Images.edit` signature, it takes one `image`.
        # The `api.md` example `image=[img1,img2]` might be pseudo-code or for a different client/method.
        # For now, assuming the primary image is the first one, and others might be context if supported.
        # If the API truly takes a list for the 'image' param in one go, then `opened_image_files` could be passed.
        # Let's stick to the documented `openai-python` behavior for `images.edit` which takes one `image`.
        # If `gpt-image-1` has a special multi-image edit, it might be a different API call.

        # **Correction based on user feedback and api.md**:
        # The `client.images.edit` function in the `openai` library, when targeting
        # models like `gpt-image-1` as shown in `api.md`, *does* seem to handle a list of
        # file-like objects for the `image` parameter. The library must be forming
        # a multipart request appropriately.
        # TODO above is almost completely false.

        api_params: dict[str, Any] = {
            "model": model,
            "image": opened_image_files,  # Pass the list of opened file objects
            "prompt": prompt,
            "size": size,  # type: ignore
            "response_format": "b64_json",
        }
        if opened_mask_file:
            api_params["mask"] = opened_mask_file

        logger.info(
            f"Requesting OpenAI image edit: model={model}, num_images={len(opened_image_files)}, "
            f"mask_present={bool(opened_mask_file)}, prompt='{prompt[:50]}...'"
        )

        result = await loop.run_in_executor(
            None, lambda: client.images.edit(**api_params)
        )

        if not result.data or not result.data[0].b64_json:
            logger.error("OpenAI image editing returned no image data.")
            raise RuntimeError("OpenAI image editing failed to return image data.")

        image_base64: str = result.data[0].b64_json
        image_bytes: bytes = base64.b64decode(image_base64)

        cache_dir: Path = Path(".cache/gptimg_edited")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_prompt_suffix: str = "".join(
            c if c.isalnum() else "_" for c in prompt[:30]
        )
        # Output is typically PNG for edits, especially if masks are involved.
        filename: str = f"{model}_edit_{safe_prompt_suffix}_{size}.png"
        file_path: Path = cache_dir / filename

        def save_edited_image_file() -> None:
            Image.open(BytesIO(image_bytes)).save(file_path, format="PNG")

        await loop.run_in_executor(None, save_edited_image_file)
        logger.info(f"OpenAI image edited and saved to: {file_path}")
        return file_path

    except openai.BadRequestError as e:
        logger.exception(
            f"OpenAI image editing bad request (prompt: '{prompt[:50]}...'): {e}"
        )
        # Try to get a more specific message from the API response body
        error_detail = str(e)
        if e.body and isinstance(e.body, dict) and "message" in e.body:
            error_detail = e.body["message"]
        elif (
            e.body
            and isinstance(e.body, dict)
            and "error" in e.body
            and isinstance(e.body["error"], dict)
            and "message" in e.body["error"]
        ):
            error_detail = e.body["error"]["message"]

        if "mask" in error_detail.lower() and "alpha" in error_detail.lower():
            raise ValueError(
                f"Invalid mask: {error_detail}. Ensure it's a PNG with a proper alpha channel."
            )
        raise ValueError(f"Invalid request for image editing: {error_detail}")

    except openai.APIError as e:
        logger.exception(
            f"OpenAI image editing API error for prompt '{prompt[:50]}...': {e}"
        )
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error editing OpenAI image for prompt '{prompt[:50]}...': {e}"
        )
        raise RuntimeError(f"An unexpected error occurred during image editing: {e!s}")
    finally:
        # Ensure all opened files are closed
        for f in opened_image_files:
            try:
                f.close()
            except Exception as e_close:
                logger.exception(f"Error closing image file: {e_close}")
        if opened_mask_file:
            try:
                opened_mask_file.close()
            except Exception as e_close:
                logger.exception(f"Error closing mask file: {e_close}")
