# commands.py

"""
Discord bot slash commands.

This module defines all the slash commands available to the bot,
handling user interactions and calling appropriate services.
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
)

import discord
import fal_client
from discord import app_commands

# Local application/library specific imports
import services
import utils
from tasks import TaskProgress, TaskStatus

logger: logging.Logger = logging.getLogger(__name__)


# Kandinsky5 ETA estimates based on empirical testing
# Format: (duration_seconds, num_steps) -> eta_minutes
KANDINSKY5_ETA_MAP = {
    # 5 second duration
    (5, 10): 1.0,
    (5, 25): 2.5,
    (5, 50): 5.0,
    # 10 second duration
    (10, 10): 2.4,
    (10, 25): 6.0,
    (10, 50): 12.0,
}


def estimate_kandinsky5_eta(duration: int, num_steps: int) -> float:
    """
    Estimate video generation time based on duration and steps.

    Uses empirical timing data with linear interpolation for intermediate values.

    Args:
        duration: Video duration in seconds (5 or 10)
        num_steps: Number of inference steps

    Returns:
        Estimated time in minutes
    """
    # Check for exact match
    if (duration, num_steps) in KANDINSKY5_ETA_MAP:
        return KANDINSKY5_ETA_MAP[(duration, num_steps)]

    # Interpolate based on steps for the given duration
    duration_etas = {steps: eta for (d, steps), eta in KANDINSKY5_ETA_MAP.items() if d == duration}

    if not duration_etas:
        # Fallback if duration not in map
        return num_steps * 0.12  # Rough average

    steps_list = sorted(duration_etas.keys())

    # If below minimum steps, extrapolate
    if num_steps <= steps_list[0]:
        return duration_etas[steps_list[0]] * (num_steps / steps_list[0])

    # If above maximum steps, extrapolate
    if num_steps >= steps_list[-1]:
        return duration_etas[steps_list[-1]] * (num_steps / steps_list[-1])

    # Interpolate between two known points
    for i in range(len(steps_list) - 1):
        step_low, step_high = steps_list[i], steps_list[i + 1]
        if step_low <= num_steps <= step_high:
            eta_low, eta_high = duration_etas[step_low], duration_etas[step_high]
            # Linear interpolation
            ratio = (num_steps - step_low) / (step_high - step_low)
            return eta_low + (eta_high - eta_low) * ratio

    # Fallback
    return num_steps * 0.12


# Command descriptions for the /help command
COMMANDS_INFO: Final[dict[str, str]] = {
    "help": "List all commands and their descriptions.",
    "claude": "Chat completion with Anthropic Claude models.",
    "gpt": "Chat with GPT-5. Supports history. Outputs as a discord embed.",
    "say": "Generate speech from text using OpenAI's TTS API.",
    "youtube": "Search YouTube. Returns top result.",
    "temp": "Get the current temperature in Fayetteville, AR.",
    "google": "Uses Google Custom Search API to get results from the web.",
    "rembg": "Remove image background using Rembg.",
    "gptimg": "Generate or edit images using OpenAI's GPT Image model.",
    "k5": "Generate a video using Kandinsky-5 text-to-video model.",
    "sd": "Generate images using Stable Diffusion 1.5 model.",
    "sd-load": "Load a Stable Diffusion checkpoint.",
    "sd-list": "List available Stable Diffusion checkpoints.",
    "z": "Generate an image using Z-Image-Turbo.",
}

# Type alias for model choice values
ModelChoiceValue = str | float

# Available model choices for various commands
MODEL_CHOICES: Final[dict[str, list[ModelChoiceValue]]] = {
    "claude": [
        "claude-sonnet-4-5",  # Claude Sonnet 4.5 (auto-updated)
        "claude-haiku-4-5",  # Claude Haiku 4.5 (auto-updated)
        "claude-opus-4-1",  # Claude Opus 4.1 (auto-updated)
    ],
    "gpt": ["gpt-5", "gpt-5-mini", "gpt-5-nano"],
    "voices": ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"],
    "speeds": [0.5, 1.0, 1.25, 1.5, 2.0],
    "gptimg_models": ["gpt-image-1"],  # Simplified to just the GPT Image model
    "gptimg_sizes": [
        "auto",
        "1024x1024",
        "1536x1024",
        "1024x1536",  # GPT Image supported sizes
    ],
    "gptimg_quality": ["auto", "low", "medium", "high"],  # GPT Image quality options
    # Stable Diffusion schedulers
    "sd_schedulers": ["DDIM", "Euler", "Euler A", "Heun", "DPM++ 2M", "DPM++ 3M", "DPM++ SDE"],
}

# Type aliases for specific string literals used in choices
AnthropicModel = Literal[
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-1",
]
GPTModel = Literal["gpt-5", "gpt-5-mini", "gpt-5-nano"]
TTSVoice = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
TTSSpeed = Literal["0.5", "1.0", "1.25", "1.5", "2.0"]  # Stored as string from choice

GPTImageModel = Literal["gpt-image-1"]
GPTImageSize = Literal["auto", "1024x1024", "1536x1024", "1024x1536"]
GPTImageQuality = Literal["auto", "low", "medium", "high"]


@app_commands.command(name="help", description="List all commands and their descriptions.")
async def help_command(interaction: discord.Interaction) -> None:
    """Displays a list of all available slash commands and their descriptions."""
    help_message = """**🤖 Mobius Bot Commands**

**💬 AI Chat**
`/claude` - Chat with Claude (Sonnet 4.5, Haiku 4.5, Opus 4.1)
`/gpt` - Chat with GPT-5 (gpt-5, gpt-5-mini, gpt-5-nano)

**🎨 Media Generation**
`/say` - Generate speech from text (OpenAI TTS, 6 voices)
`/gptimg` - Generate or edit images (GPT Image model, PNG output)
`/k5` - Generate video (Kandinsky-5 text-to-video)

**🔧 Media Processing**
`/rembg` - Remove image background (fal.ai/rembg)

**🔍 Search & Info**
`/youtube` - Search YouTube (returns top result)
`/google` - Search the web (Google Custom Search)
`/temp` - Get current temperature in Fayetteville, AR

**🖼️ Local Models**
`/z` - Generate image (Z-Image-Turbo)

**Info**
`/help` - Show this help message"""

    await interaction.response.send_message(help_message, ephemeral=True)


@app_commands.command(
    name="say",
    description="Generate speech from text using OpenAI's TTS API. Max 4096 chars.",
)
@app_commands.choices(
    voice=[app_commands.Choice(name=str(voice_name), value=voice_name) for voice_name in MODEL_CHOICES["voices"]],  # type: ignore # Mypy can't infer the type argument
    speed=[app_commands.Choice(name=f"{speed_val}x", value=str(speed_val)) for speed_val in MODEL_CHOICES["speeds"]],
)
async def say_command(
    interaction: discord.Interaction,
    text: str,
    voice: TTSVoice = "echo",
    speed: TTSSpeed = "1.0",
) -> None:
    """
    Generates speech from the provided text using OpenAI's TTS API and sends it as an audio file.
    """
    if len(text) > 4096:
        await interaction.response.send_message("Text cannot exceed 4096 characters.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        speech_speed: float = float(speed)
        waveform_video_file_path: Path = await services.generate_speech(text, voice, speech_speed)
        discord_file: discord.File = discord.File(waveform_video_file_path, filename=waveform_video_file_path.name)
        await interaction.followup.send(
            content=f'Audio for "{text[:50]}..." using voice: {voice}, speed: {speed}x',
            file=discord_file,
        )
    except ValueError as e:  # Catches float conversion error or errors from services.generate_speech
        logger.exception(f"Error in 'say' command processing: {e}")
        await interaction.followup.send(f"An error occurred: {e}. Please check your input.", ephemeral=True)
    except Exception:
        logger.exception(f"Unexpected error in 'say' command: {text}, {voice}, {speed}")
        await interaction.followup.send("An unexpected error occurred while generating speech.", ephemeral=True)


@app_commands.command(
    name="rembg",
    description="Remove background from an image using fal.ai/imageutils/rembg.",
)
async def rembg_command(interaction: discord.Interaction, image: discord.Attachment) -> None:
    """Removes the background from the provided image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.response.send_message("Please upload a valid image file (PNG, JPG, WEBP).", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        # fal_client.subscribe_async returns a dict, ensure key access is safe
        result: Any = await fal_client.subscribe_async(
            "fal-ai/imageutils/rembg",
            arguments={"image_url": image.url},
            # with_logs=True, # Enable if debugging is needed
        )
        processed_image_url: str | None = result.get("image", {}).get("url")
        if processed_image_url:
            await interaction.followup.send(processed_image_url)
        else:
            logger.error(f"rembg service did not return an image URL. Result: {result}")
            await interaction.followup.send(
                "Failed to process the image: No image URL returned from service.",
                ephemeral=True,
            )
    except Exception:
        logger.exception(f"Error removing background for image: {image.url}")
        await interaction.followup.send("An error occurred while removing the image background.", ephemeral=True)


@app_commands.command(name="claude", description="Chat completion with Anthropic Claude models.")
@app_commands.choices(model=[app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["claude"]])
async def anthropic_command(
    interaction: discord.Interaction,
    prompt: str,
    max_tokens: app_commands.Range[int, 1, 4096] = 1024,  # Adjusted max_tokens
    suppress_embeds: bool = True,
    model: AnthropicModel = "claude-sonnet-4-5",
) -> None:
    """Gets a chat completion from an Anthropic Claude model."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        message_text: str = await services.anthropic_chat_completion(prompt=prompt, max_tokens=max_tokens, model=model)

        # Format with escaped prompt for safety
        formatted_response: str = (
            f"### _{interaction.user.display_name}_:\n\n"
            f"```\n{discord.utils.escape_markdown(prompt)}\n```\n"
            f"### {model}:\n\n{message_text}"
        )

        if len(formatted_response) >= 2000:
            temp_file_path: Path = utils.create_temp_file(formatted_response, ".md")
            discord_file: discord.File = discord.File(temp_file_path, filename="response.md")
            await interaction.followup.send(
                content="Response too long, sending as a file.",
                file=discord_file,
            )
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.exception(f"Error deleting temporary file {temp_file_path}: {e}")
        else:
            await interaction.followup.send(content=formatted_response, suppress_embeds=suppress_embeds)
    except Exception:
        logger.exception(f"Error with Anthropic command for prompt: {prompt}")
        await interaction.followup.send(
            "An error occurred while communicating with the Anthropic API.",
            ephemeral=True,
        )


async def _handle_long_response(
    interaction: discord.Interaction,
    content: str,
    prompt: str,
    model_name: str,
    seed: int | None = None,
) -> None:
    """
    Handles long responses by sending them as a Discord file attachment.

    Args:
        interaction: The Discord interaction object.
        content: The long text content to send.
        prompt: The original prompt that generated the content.
        model_name: The name of the model used.
        seed: The optional seed used for generation.
    """
    temp_file_path: Path = Path()
    try:
        temp_file_path = utils.create_temp_file(content, ".md")
        discord_file: discord.File = discord.File(temp_file_path, filename=f"{model_name}_response.md")

        embed: discord.Embed = discord.Embed(
            title=f"Response from {model_name}",
            description="The response was too long to display directly. Please see the attached file.",
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="Original Prompt",
            value=f"```{discord.utils.escape_markdown(prompt[:1000])}```",
            inline=False,
        )
        if seed is not None:
            embed.add_field(name="Seed", value=str(seed), inline=True)

        await interaction.followup.send(embed=embed, file=discord_file)
    except Exception:
        logger.exception("Failed to handle long response and send as file.")
        await interaction.followup.send("Error sending long response as a file.", ephemeral=True)
    finally:
        if temp_file_path:
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.exception(f"Error deleting temporary file {temp_file_path}: {e}")


@app_commands.command(
    name="gpt",
    description="Chat with OpenAI's GPT models. Supports history. Outputs as embed.",
)
@app_commands.choices(model_name=[app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["gpt"]])
async def gpt_command(
    interaction: discord.Interaction,
    prompt: str,
    seed: int | None = None,  # OpenAI API seed is Optional
    model_name: GPTModel = "gpt-5-mini",
) -> None:
    """Chats with an OpenAI GPT model and displays the response in an embed."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        # For simple one-turn, history is just the user prompt
        history: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        # Ensure seed is passed as int if provided, or None
        api_seed: int | None = int(seed) if seed is not None and seed != -1 else None

        assistant_response: str = await services.gpt_chat_completion(history, model_name, api_seed)

        if len(assistant_response) >= 4000:  # Embed description limit is 4096
            await _handle_long_response(interaction, assistant_response, prompt, model_name, api_seed)
        else:
            embed: discord.Embed = discord.Embed(
                title=f"Response from {model_name}",
                description=assistant_response,
                color=discord.Color.green(),
            )
            embed.add_field(
                name="Prompt",
                value=f"```{discord.utils.escape_markdown(prompt[:1000])}```",
                inline=False,
            )
            if api_seed is not None:
                embed.add_field(name="Seed", value=str(api_seed), inline=True)
            await interaction.followup.send(embed=embed)
    except Exception:
        logger.exception(f"Error with GPT command for prompt: {prompt}")
        await interaction.followup.send(
            "An error occurred while communicating with the OpenAI API.",
            ephemeral=True,
        )


@app_commands.command(name="youtube", description="Search YouTube and return the top video result.")
async def youtube_command(interaction: discord.Interaction, query: str) -> None:
    """Searches YouTube for the given query and returns the top video result."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    youtube_api_key: str | None = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        logger.error("YOUTUBE_API_KEY environment variable not set.")
        await interaction.followup.send(
            "YouTube API key is not configured. This command is unavailable.",
            ephemeral=True,
        )
        return

    try:
        result: dict[str, Any] = await services.get_top_youtube_result(query, youtube_api_key)
        if "error" in result:
            await interaction.followup.send(f"YouTube search error: {result['error']}", ephemeral=True)
        elif "videoId" in result:
            video_url: str = f"https://www.youtube.com/watch?v={result['videoId']}"
            await interaction.followup.send(
                f"Top YouTube result for '{discord.utils.escape_markdown(query)}':\n{video_url}"
            )
        else:
            await interaction.followup.send(
                f"No results found for '{discord.utils.escape_markdown(query)}'.",
                ephemeral=True,
            )
    except Exception:
        logger.exception(f"Error during YouTube search for query: {query}")
        await interaction.followup.send("An error occurred during the YouTube search.", ephemeral=True)


@app_commands.command(name="temp", description="Get the current temperature in Fayetteville, AR.")
async def temp_command(interaction: discord.Interaction) -> None:
    """Fetches and displays the current temperature for Fayetteville, AR."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        temperature_info: str = await services.temp_callback()
        await interaction.followup.send(temperature_info)
    except Exception:
        logger.exception("Error fetching temperature data.")
        await interaction.followup.send("An error occurred while fetching temperature data.", ephemeral=True)


@app_commands.command(name="google", description="Search the web using Google Custom Search API.")
async def google_command(interaction: discord.Interaction, query: str) -> None:
    """Performs a Google search using the Custom Search API and returns top results."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    if not os.getenv("GOOGLE_SEARCH_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        logger.error("Google Search API key or CSE ID not configured.")
        await interaction.followup.send(
            "Google Search is not configured. This command is unavailable.",
            ephemeral=True,
        )
        return
    try:
        search_results: str = await services.google_search(query)
        if search_results:
            await interaction.followup.send(
                f"Google search results for '{discord.utils.escape_markdown(query)}':\n{search_results}"
            )
        else:
            await interaction.followup.send(
                f"No Google results found for '{discord.utils.escape_markdown(query)}'.",
                ephemeral=True,
            )
    except Exception:
        logger.exception(f"Error during Google search for query: {query}")
        await interaction.followup.send("An error occurred during the Google search.", ephemeral=True)


@app_commands.command(
    name="gptimg",
    description="Generate or edit images using OpenAI's GPT Image model (saves as PNG).",
)
@app_commands.choices(
    model=[app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["gptimg_models"]],
    size=[app_commands.Choice(name=str(s), value=s) for s in MODEL_CHOICES["gptimg_sizes"]],
    quality=[app_commands.Choice(name=str(q), value=q) for q in MODEL_CHOICES["gptimg_quality"]],
)
async def gptimg_command(
    interaction: discord.Interaction,
    prompt: str,
    edit_image1: discord.Attachment | None = None,
    edit_image2: discord.Attachment | None = None,
    edit_image3: discord.Attachment | None = None,  # Added more image slots
    edit_image4: discord.Attachment | None = None,
    edit_image5: discord.Attachment | None = None,
    mask_image: discord.Attachment | None = None,
    model: GPTImageModel = "gpt-image-1",
    size: GPTImageSize = "auto",
    quality: GPTImageQuality = "auto",
    transparent_background: bool = False,  # Re-added transparency support
) -> None:
    """
    Generates or edits images using OpenAI's GPT Image model.
    Images are saved as PNG to support transparency features.
    - Text-to-image: Provide a prompt.
    - Image editing: Provide prompt + edit_image1 (and optionally up to 5 images total).
    - Masked editing: Provide prompt + edit_image1 + mask_image (mask applies to edit_image1).
    """
    await interaction.response.defer(ephemeral=False, thinking=True)

    # Collect all provided edit images
    edit_images: list[discord.Attachment] = [
        img for img in [edit_image1, edit_image2, edit_image3, edit_image4, edit_image5] if img is not None
    ]

    is_editing: bool = len(edit_images) > 0
    operation_type: str = "editing" if is_editing else "generating"
    start_time: float = time.time()

    # Temporary file paths
    temp_image_paths: list[Path] = []
    temp_mask_path: Path | None = None
    generated_image_path: Path | None = None

    initial_message: discord.WebhookMessage | None = None

    # Check if the message will exceed Discord's 2000 character limit
    escaped_prompt = discord.utils.escape_markdown(prompt)
    progress_message_content: str = (
        f"🖌️ {operation_type.capitalize()} your image with {model}...\n\n"
        f"⏳ This can take 30-90 seconds.\n\n"
        f"**Prompt:** {escaped_prompt}\n"
        f"**Settings:** Size: {size} | Quality: {quality}" + (f" | Images: {len(edit_images)}" if is_editing else "")
    )

    # Check length and provide clear error if too long
    if len(progress_message_content) > 2000:
        error_msg = (
            "❌ **Error:** Your prompt is too long for Discord messages.\n"
            f"Current length: {len(escaped_prompt)} characters\n"
            f"Maximum allowed: ~{2000 - len(progress_message_content) + len(escaped_prompt)} characters\n\n"
            "Please shorten your prompt and try again."
        )
        await interaction.followup.send(error_msg)
        return None

    try:
        initial_message = await interaction.followup.send(progress_message_content)
    except discord.HTTPException as e:
        logger.exception(f"Failed to send initial progress message for gptimg: {e}")
        error_msg = "❌ **Error:** Failed to send progress message. Your prompt may be too long."
        await interaction.followup.send(error_msg)
        return None

    operation_complete_event: asyncio.Event = asyncio.Event()
    update_task: asyncio.Task[None] | None = None
    generation_task: asyncio.Task[Path] | None = None

    try:
        if is_editing:
            # Validate and save all edit images
            for i, edit_img in enumerate(edit_images):
                if not edit_img.content_type or not edit_img.content_type.startswith("image/"):
                    raise ValueError(f"edit_image{i + 1} is not a valid image type.")

                with tempfile.NamedTemporaryFile(
                    suffix=Path(edit_img.filename).suffix or ".png", delete=False
                ) as tmp_file:
                    await edit_img.save(Path(tmp_file.name))
                    temp_image_paths.append(Path(tmp_file.name))

            # Validate and save mask_image if provided
            if mask_image:
                if not mask_image.content_type or not mask_image.content_type.startswith("image/"):
                    raise ValueError("mask_image is not a valid image type.")

                with tempfile.NamedTemporaryFile(
                    suffix=Path(mask_image.filename).suffix or ".png", delete=False
                ) as tmp_mask_file:
                    await mask_image.save(Path(tmp_mask_file.name))
                    temp_mask_path = Path(tmp_mask_file.name)

        max_generation_time_seconds: Final[int] = 300
        progress_update_interval_seconds: Final[int] = 3
        estimated_completion_time_seconds: Final[int] = 120

        async def update_progress_periodically() -> None:
            nonlocal initial_message
            update_count: int = 0
            max_updates: int = max_generation_time_seconds // progress_update_interval_seconds + 5

            while update_count < max_updates and not operation_complete_event.is_set():
                elapsed_time: int = int(time.time() - start_time)
                minutes, seconds = divmod(elapsed_time, 60)

                if elapsed_time > max_generation_time_seconds:
                    logger.warning(f"gptimg: Generation timeout suspected after {elapsed_time}s.")
                    break

                progress_percentage: float = min(100.0, (elapsed_time / estimated_completion_time_seconds) * 100)
                segments: int = 10
                filled_segments: int = int((progress_percentage / 100) * segments)
                progress_bar: str = "⬛" * filled_segments + "⬜" * (segments - filled_segments)
                est_remaining: int = max(0, estimated_completion_time_seconds - elapsed_time)
                est_min, est_sec = divmod(est_remaining, 60)

                updated_content: str = (
                    f"🖌️ {operation_type.capitalize()} your image with {model}...\n\n"
                    f"⏳ Elapsed: {minutes:02d}:{seconds:02d} • Est. Rem: {est_min:02d}:{est_sec:02d}\n"
                    f"Progress: {progress_bar} {progress_percentage:.0f}%\n\n"
                    f"**Prompt:** {discord.utils.escape_markdown(prompt)}\n"
                    f"**Settings:** Size: {size} | Quality: {quality}"
                    + (f" | Images: {len(edit_images)}" if is_editing else "")
                )

                try:
                    if initial_message:
                        await initial_message.edit(content=updated_content)
                    update_count += 1
                except discord.NotFound:
                    logger.warning("gptimg: Progress message not found, likely deleted by user.")
                    initial_message = None
                    break
                except discord.HTTPException as e_http:
                    logger.exception(f"gptimg: Error updating progress message: {e_http}")

                try:
                    await asyncio.wait_for(
                        operation_complete_event.wait(),
                        timeout=progress_update_interval_seconds,
                    )
                except TimeoutError:
                    continue
                except Exception as e_wait:
                    logger.exception(f"gptimg: Error in update_progress wait: {e_wait}")
                    break

            logger.info(f"gptimg: Progress update loop finished. Updates: {update_count}")
            return None

        update_task = asyncio.create_task(update_progress_periodically())

        if is_editing:
            generation_task = asyncio.create_task(
                services.edit_gpt_image(
                    prompt=prompt,
                    images=temp_image_paths,
                    mask=temp_mask_path,
                    model=model,
                    size=size,
                )
            )
        else:  # Generating a new image
            generation_task = asyncio.create_task(
                services.generate_gpt_image(
                    prompt=prompt,
                    model=model,
                    quality=quality,
                    size=size,
                    transparent_background=transparent_background,
                )
            )

        try:
            generated_image_path = await asyncio.wait_for(generation_task, timeout=max_generation_time_seconds)
        except TimeoutError:
            logger.exception(f"gptimg: Image {operation_type} timed out after {max_generation_time_seconds}s.")
            if generation_task and not generation_task.done():
                generation_task.cancel()
                try:
                    await generation_task
                except asyncio.CancelledError:
                    logger.info("gptimg: Generation task successfully cancelled on timeout.")
                except Exception as e_cancel:
                    logger.exception(f"gptimg: Error during generation task cancellation: {e_cancel}")
            raise RuntimeError(f"Image {operation_type} timed out. The API may be experiencing delays.")
        finally:
            operation_complete_event.set()
            if update_task and not update_task.done():
                try:
                    await asyncio.wait_for(update_task, timeout=progress_update_interval_seconds + 1)
                except TimeoutError:
                    logger.warning("gptimg: Update task did not finish cleanly after generation.")
                    if not update_task.done():
                        update_task.cancel()
                except Exception as e_await_update:
                    logger.exception(f"gptimg: Error awaiting update task: {e_await_update}")

        total_time_seconds: float = time.time() - start_time
        mins, secs = divmod(int(total_time_seconds), 60)

        if not generated_image_path or not generated_image_path.exists():
            raise RuntimeError("Image generation service did not return a valid image file.")

        final_discord_file: discord.File = discord.File(generated_image_path, filename=generated_image_path.name)
        final_message_content: str = (
            f"✅ Image {operation_type} complete! (Took {mins}m {secs}s)\n\n"
            f"**Prompt:** {discord.utils.escape_markdown(prompt)}\n"
            f"**Settings:** Model: {model} | Size: {size} | Quality: {quality}"
            + (f" | Images: {len(edit_images)}" if is_editing else "")
            + (f" | Transparent: {transparent_background}" if not is_editing else "")
            + "\n**Format:** PNG (supports transparency)"
        )

        await interaction.followup.send(content=final_message_content, file=final_discord_file)

        if initial_message:
            try:
                await initial_message.delete()
            except discord.HTTPException as e_del:
                logger.warning(f"gptimg: Could not delete progress message: {e_del}")

    except ValueError as ve:
        logger.exception(f"gptimg: Input error - {ve}")
        await interaction.followup.send(f"❌ Input error: {ve}", ephemeral=True)
    except RuntimeError as re:
        logger.exception(f"gptimg: Runtime error - {re}")
        await interaction.followup.send(f"❌ Runtime error: {re}", ephemeral=True)
    except Exception as e:
        logger.exception(f"gptimg: Unexpected error during {operation_type} for prompt: {prompt}")
        await interaction.followup.send(f"❌ An unexpected error occurred: {e!s}", ephemeral=True)
    finally:
        # Clean up temporary files
        for p in temp_image_paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError as e_unlink:
                    logger.exception(f"gptimg: Error deleting temp image file {p}: {e_unlink}")
        if temp_mask_path and temp_mask_path.exists():
            try:
                temp_mask_path.unlink()
            except OSError as e_unlink:
                logger.exception(f"gptimg: Error deleting temp mask file {temp_mask_path}: {e_unlink}")

        operation_complete_event.set()
        # Final cleanup for any remaining tasks
        tasks_to_cancel = [t for t in [update_task, generation_task] if t and not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                logger.info("gptimg: Remaining tasks cancelled in final cleanup.")
            except Exception as e_gather:
                logger.exception(f"gptimg: Error during final task cleanup: {e_gather}")

        return None


@app_commands.command(name="k5", description="Generate a video using Kandinsky-5 text-to-video model")
@app_commands.describe(
    prompt="Text description of the video to generate",
    negative_prompt="Describe what to avoid in the video (optional)",
    duration="Video duration in seconds (any positive number, default: 5)",
    width="Video width in pixels (must be a multiple of 16, default: 512)",
    height="Video height in pixels (must be a multiple of 16, default: 512)",
    num_steps="Number of inference steps (default: 50, higher = better quality but slower)",
    guidance_weight="CFG weight for prompt adherence (optional, uses model default if not set)",
    scheduler_scale="Flow matching scheduler scale (default: 5.0, advanced parameter)",
    seed="Random seed for reproducible results (optional)",
)
async def kandinsky5_command(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: str | None = None,
    duration: app_commands.Range[int, 1, 1000] = 3,
    width: int = 512,
    height: int = 512,
    num_steps: app_commands.Range[int, 1, 100] = 20,
    guidance_weight: app_commands.Range[float, 0.0, 100.0] | None = None,
    scheduler_scale: app_commands.Range[float, 0.0, 100.0] = 5.0,
    seed: int | None = None,
) -> None:
    """Generate a video using Kandinsky-5 text-to-video model."""
    await interaction.response.defer(thinking=True)

    # Validate that width and height are multiples of 16
    def round_to_multiple_of_16(value: int) -> int:
        """Round a value to the nearest multiple of 16."""
        return round(value / 16) * 16

    width_valid = width % 16 == 0
    height_valid = height % 16 == 0

    if not width_valid or not height_valid:
        # Calculate nearest valid resolutions
        nearest_width = round_to_multiple_of_16(width)
        nearest_height = round_to_multiple_of_16(height)

        error_message = (
            f"❌ **Invalid Resolution**\n\n"
            f"Width and height must be multiples of 16.\n\n"
            f"**Your input:** {width}x{height}\n"
        )

        if not width_valid and not height_valid:
            error_message += f"**Recommended:** {nearest_width}x{nearest_height}\n"
        elif not width_valid:
            error_message += f"**Recommended:** {nearest_width}x{height}\n"
        else:  # height not valid
            error_message += f"**Recommended:** {width}x{nearest_height}\n"

        error_message += "\nPlease try again with a valid resolution."

        await interaction.followup.send(error_message, ephemeral=True)
        return

    try:
        logger.info(
            f"kandinsky5: User {interaction.user} requested video generation. "
            f"Prompt: '{prompt[:50]}...', Duration: {duration}s, Steps: {num_steps}, Seed: {seed}"
        )

        # Check API health first
        is_healthy = await services.check_kandinsky5_health()
        if not is_healthy:
            await interaction.followup.send("⚠️ The Kandinsky-5 API is currently down. Sorry! Please try again later.")
            return

        start_time = time.time()
        last_followup_time = time.time()
        last_keepalive_msg = None

        # Calculate estimated ETA
        estimated_eta_mins = estimate_kandinsky5_eta(duration, num_steps)
        eta_display = (
            f"{estimated_eta_mins:.1f} minutes"
            if estimated_eta_mins >= 1
            else f"{int(estimated_eta_mins * 60)} seconds"
        )

        # Edit original message with initial status
        seed_info = f" | **Seed:** {seed}" if seed is not None else ""
        guidance_info = f" | **CFG:** {guidance_weight}" if guidance_weight is not None else ""
        await interaction.edit_original_response(
            content=f"🎬 Generating video with Kandinsky-5...\n"
            f"**Prompt:** {discord.utils.escape_markdown(prompt[:100])}\n"
            f"**Resolution:** {width}x{height} | **Duration:** {duration}s | "
            f"**Steps:** {num_steps}{guidance_info}{seed_info}\n"
            f"⏳ Submitting task... (ETA: ~{eta_display})"
        )

        # Create progress callback to update Discord message
        async def on_progress(progress: TaskProgress) -> None:
            nonlocal last_followup_time, last_keepalive_msg
            """Update Discord status message with progress."""
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            elapsed_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

            # Build status emoji
            if progress.status == TaskStatus.PENDING:
                status_emoji = "⏳"
                status_text = "Queued"
            elif progress.status == TaskStatus.PROCESSING:
                status_emoji = "🎨"
                status_text = "Generating"
            else:
                status_emoji = "⏳"
                status_text = progress.status.value.title()

            # Build time display with ETA
            time_display = f"🕐 Elapsed: {elapsed_str} / ETA: ~{eta_display}"

            # Build message
            message_str = ""
            if progress.message:
                message_str = f"\n💬 {progress.message}"

            updated_content = (
                f"{status_emoji} **{status_text}** video with Kandinsky-5\n"
                f"**Prompt:** {discord.utils.escape_markdown(prompt[:100])}\n"
                f"**Resolution:** {width}x{height} | **Duration:** {duration}s | "
                f"**Steps:** {num_steps}{guidance_info}{seed_info}\n"
                f"{time_display}{message_str}"
            )

            # Discord interaction tokens expire after 15 minutes
            # Send a new followup every 10 minutes to keep the interaction alive
            time_since_last_followup = time.time() - last_followup_time
            if time_since_last_followup > 600:  # 10 minutes
                # Delete previous keepalive message to reduce spam
                if last_keepalive_msg:
                    try:
                        await last_keepalive_msg.delete()
                    except discord.HTTPException as e:
                        logger.warning(f"kandinsky5: Could not delete old keepalive: {e}")

                # Send new keepalive
                try:
                    last_keepalive_msg = await interaction.followup.send(
                        f"⏳ Still working on your video... (Elapsed: {elapsed_str} / ETA: ~{eta_display})",
                        ephemeral=True,
                        wait=True,
                    )
                    last_followup_time = time.time()
                    logger.info(f"kandinsky5: Sent keepalive followup at {elapsed_str}")
                except discord.HTTPException as e:
                    logger.warning(f"kandinsky5: Could not send keepalive followup: {e}")

            try:
                await interaction.edit_original_response(content=updated_content)
            except discord.HTTPException as e:
                logger.warning(f"kandinsky5: Could not update original message: {e}")

        # Generate the video with progress updates
        video_path = await services.generate_kandinsky5_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            seed=seed,
            progress_callback=on_progress,
        )

        # Calculate generation time
        total_time = time.time() - start_time
        mins, secs = divmod(int(total_time), 60)

        # Clean up keepalive message if it exists
        if last_keepalive_msg:
            try:
                await last_keepalive_msg.delete()
            except discord.HTTPException as e:
                logger.warning(f"kandinsky5: Could not delete final keepalive: {e}")

        # Edit original message with the video
        discord_file = discord.File(video_path, filename=video_path.name)
        seed_info = f" | Seed: {seed}" if seed is not None else ""
        guidance_info = f" | CFG: {guidance_weight}" if guidance_weight is not None else ""
        neg_prompt_info = (
            f"\n**Negative Prompt:** {discord.utils.escape_markdown(negative_prompt)}" if negative_prompt else ""
        )
        final_message = (
            f"✅ Video generation complete! (Took {mins}m {secs}s)\n\n"
            f"**Prompt:** {discord.utils.escape_markdown(prompt)}{neg_prompt_info}\n"
            f"**Settings:** {width}x{height} | Duration: {duration}s | Steps: {num_steps}{guidance_info}{seed_info}\n"
            f"**Format:** MP4"
        )

        # For long-running tasks (>15 min), the interaction token may expire
        # Try editing original response first, fall back to channel message
        try:
            await interaction.edit_original_response(content=final_message, attachments=[discord_file])
        except discord.HTTPException as e:
            if e.code == 50027:  # Invalid Webhook Token (interaction expired)
                logger.info("kandinsky5: Interaction expired, sending to channel instead")
                # Send to the channel where the command was invoked
                channel = interaction.channel
                if channel is not None and hasattr(channel, "send"):
                    await channel.send(content=final_message, file=discord_file)  # type: ignore[union-attr]
            else:
                raise

    except RuntimeError as re:
        logger.exception(f"kandinsky5: Runtime error - {re}")
        await interaction.followup.send(f"❌ Error generating video: {re}")
    except Exception as e:
        logger.exception(f"kandinsky5: Unexpected error during video generation: {e}")
        await interaction.followup.send(f"❌ An unexpected error occurred: {e!s}")


# --- Stable Diffusion Commands ---

# Create scheduler choices for the /sd command
SD_SCHEDULER_CHOICES = [
    app_commands.Choice(name="DDIM", value="DDIM"),
    app_commands.Choice(name="Euler", value="Euler"),
    app_commands.Choice(name="Euler A", value="Euler A"),
    app_commands.Choice(name="Heun", value="Heun"),
    app_commands.Choice(name="DPM++ 2M", value="DPM++ 2M"),
    app_commands.Choice(name="DPM++ 3M", value="DPM++ 3M"),
    app_commands.Choice(name="DPM++ SDE", value="DPM++ SDE"),
]


@app_commands.command(name="sd", description="Generate images using Stable Diffusion 1.5 model")
@app_commands.describe(
    prompt="Text description of the image to generate",
    negative_prompt="Things to avoid in the image (optional)",
    num_inference_steps="Number of denoising steps (1-100, default: 8)",
    guidance_scale="CFG scale - higher = more prompt adherence (1.0-20.0, default: 7.5)",
    width="Image width in pixels (must be multiple of 64, default: 512)",
    height="Image height in pixels (must be multiple of 64, default: 512)",
    num_images="Number of images to generate (1-4, default: 1)",
    seed="Random seed for reproducibility (-1 for random, default: -1)",
    lora_scale="LoRA intensity (0.0-2.0, default: 1.0)",
    scheduler_name="Sampling method (default: DPM++ SDE)",
)
@app_commands.choices(scheduler_name=SD_SCHEDULER_CHOICES)
async def sd_command(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: app_commands.Range[int, 1, 100] = 8,
    guidance_scale: app_commands.Range[float, 1.0, 20.0] = 7.5,
    width: app_commands.Range[int, 64, 1024] = 512,
    height: app_commands.Range[int, 64, 1024] = 512,
    num_images: app_commands.Range[int, 1, 4] = 1,
    seed: int = -1,
    lora_scale: app_commands.Range[float, 0.0, 2.0] = 1.0,
    scheduler_name: str = "DPM++ SDE",
) -> None:
    """Generate images using Stable Diffusion 1.5."""

    # Validate resolution (must be multiples of 64)
    if width % 64 != 0:
        await interaction.response.send_message(
            f"❌ **Error:** Width must be a multiple of 64. Got {width}.\n"
            f"💡 **Suggestion:** Try {(width // 64) * 64} or {((width // 64) + 1) * 64}",
            ephemeral=True,
        )
        return

    if height % 64 != 0:
        await interaction.response.send_message(
            f"❌ **Error:** Height must be a multiple of 64. Got {height}.\n"
            f"💡 **Suggestion:** Try {(height // 64) * 64} or {((height // 64) + 1) * 64}",
            ephemeral=True,
        )
        return

    # Log the request
    logger.info(
        f"sd: User {interaction.user} requested image generation: "
        f"prompt='{prompt[:50]}...', resolution={width}x{height}, "
        f"steps={num_inference_steps}, scheduler={scheduler_name}, num_images={num_images}"
    )

    # Defer the response (this will be quick, no need for thinking=True)
    await interaction.response.defer()

    try:
        # Update status
        await interaction.edit_original_response(
            content=f"🎨 **Generating {num_images} image(s)...**\n"
            f"**Prompt:** {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
            f"**Scheduler:** {scheduler_name} | **Steps:** {num_inference_steps} | **Size:** {width}x{height}"
        )

        # Generate images
        start_time = time.time()
        image_paths = await services.generate_sd_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed,
            lora_scale=lora_scale,
            scheduler_name=scheduler_name,
        )
        generation_time = time.time() - start_time

        # Create Discord files from all generated images
        discord_files = [discord.File(str(img_path)) for img_path in image_paths]

        # Send the result with all images
        result_message = (
            f"✅ **Generated {len(image_paths)} image(s)** in {generation_time:.1f}s\n"
            f"**Prompt:** {prompt}\n"
            f"**Scheduler:** {scheduler_name} | **Steps:** {num_inference_steps} | "
            f"**Size:** {width}x{height} | **Seed:** {seed}"
        )

        await interaction.edit_original_response(
            content=result_message,
            attachments=discord_files,
        )

    except ValueError as ve:
        logger.exception(f"sd: Validation error - {ve}")
        await interaction.edit_original_response(content=f"❌ **Invalid parameters:** {ve}")
    except RuntimeError as re:
        logger.exception(f"sd: Runtime error - {re}")
        await interaction.edit_original_response(content=f"❌ **Error generating image:** {re}")
    except Exception as e:
        logger.exception(f"sd: Unexpected error: {e}")
        await interaction.edit_original_response(content=f"❌ **An unexpected error occurred:** {e!s}")


@app_commands.command(name="sd-load", description="Load a Stable Diffusion checkpoint")
@app_commands.describe(
    checkpoint_path="Path to checkpoint (e.g., 'checkpoints/laion2b/resume/checkpoint-17500') or 'base' for base model"
)
async def sd_load_command(
    interaction: discord.Interaction,
    checkpoint_path: str,
) -> None:
    """Load a Stable Diffusion checkpoint."""
    await interaction.response.defer(thinking=True)

    try:
        logger.info(f"sd-load: User {interaction.user} requested to load checkpoint: {checkpoint_path}")

        # Convert "base" to None for the API
        api_checkpoint_path = None if checkpoint_path.lower() == "base" else checkpoint_path

        # Load the checkpoint
        result = await services.load_sd_checkpoint(api_checkpoint_path)

        checkpoint_name = result.get("checkpoint_name", "Unknown")
        is_lora = result.get("is_lora", False)
        status = result.get("status", "Loaded successfully")

        checkpoint_type = "LoRA" if is_lora else "Full Model"

        await interaction.followup.send(
            f"✅ **Checkpoint loaded successfully**\n"
            f"**Name:** {checkpoint_name}\n"
            f"**Type:** {checkpoint_type}\n"
            f"**Status:** {status}"
        )

    except RuntimeError as re:
        logger.exception(f"sd-load: Runtime error - {re}")
        await interaction.followup.send(f"❌ **Error loading checkpoint:** {re}")
    except Exception as e:
        logger.exception(f"sd-load: Unexpected error: {e}")
        await interaction.followup.send(f"❌ **An unexpected error occurred:** {e!s}")


@app_commands.command(name="sd-list", description="List available Stable Diffusion checkpoints")
async def sd_list_command(interaction: discord.Interaction) -> None:
    """List available Stable Diffusion checkpoints."""
    await interaction.response.defer(thinking=True)

    try:
        logger.info(f"sd-list: User {interaction.user} requested checkpoint list")

        # Get list of experiments
        experiments = await services.list_sd_experiments()

        if not experiments:
            await interaction.followup.send("📦 No experiment runs found.")
            return

        # Build a formatted message with all experiments and their checkpoints
        message_parts = ["📦 **Available Stable Diffusion Checkpoints**\n"]

        for experiment_run in experiments:
            try:
                checkpoints = await services.list_sd_checkpoints(experiment_run)

                message_parts.append(f"\n**{experiment_run}**")
                if checkpoints:
                    for checkpoint in checkpoints:
                        full_path = f"checkpoints/{experiment_run}/{checkpoint}"
                        message_parts.append(f"  • `{checkpoint}` → `{full_path}`")
                else:
                    message_parts.append("  _No checkpoints found_")
            except Exception as e:
                logger.warning(f"sd-list: Error listing checkpoints for {experiment_run}: {e}")
                message_parts.append(f"  _Error: {e!s}_")

        message_parts.append("\n💡 **Usage:** `/sd-load checkpoint_path:<full_path>`")
        message_parts.append("💡 **Example:** `/sd-load checkpoint_path:checkpoints/laion2b/resume/checkpoint-17500`")

        full_message = "\n".join(message_parts)

        # Discord has a 2000 character limit, split if necessary
        if len(full_message) <= 2000:
            await interaction.followup.send(full_message)
        else:
            # Split into multiple messages
            chunks = []
            current_chunk = message_parts[0]  # Start with header

            for part in message_parts[1:]:
                if len(current_chunk) + len(part) + 1 <= 2000:
                    current_chunk += "\n" + part
                else:
                    chunks.append(current_chunk)
                    current_chunk = part

            if current_chunk:
                chunks.append(current_chunk)

            # Send first chunk as response, rest as followups
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(chunk)

    except RuntimeError as re:
        logger.exception(f"sd-list: Runtime error - {re}")
        await interaction.followup.send(f"❌ **Error listing checkpoints:** {re}")
    except Exception as e:
        logger.exception(f"sd-list: Unexpected error: {e}")
        await interaction.followup.send(f"❌ **An unexpected error occurred:** {e!s}")


# --- FLUX 2 Commands ---

# Image size choices for FLUX 2
FLUX2_SIZE_CHOICES = [
    app_commands.Choice(name="Landscape 4:3 (default)", value="landscape_4_3"),
    app_commands.Choice(name="Landscape 16:9", value="landscape_16_9"),
    app_commands.Choice(name="Portrait 4:3", value="portrait_4_3"),
    app_commands.Choice(name="Portrait 16:9", value="portrait_16_9"),
    app_commands.Choice(name="Square HD", value="square_hd"),
    app_commands.Choice(name="Square", value="square"),
]

# Acceleration choices for FLUX 2
FLUX2_ACCELERATION_CHOICES = [
    app_commands.Choice(name="Regular (default)", value="regular"),
    app_commands.Choice(name="High (fastest)", value="high"),
    app_commands.Choice(name="None (highest quality)", value="none"),
]

# Output format choices for FLUX 2
FLUX2_FORMAT_CHOICES = [
    app_commands.Choice(name="PNG (default)", value="png"),
    app_commands.Choice(name="JPEG", value="jpeg"),
    app_commands.Choice(name="WebP", value="webp"),
]


@app_commands.command(name="flux2", description="Generate images with FLUX.2 [dev] from Black Forest Labs")
@app_commands.describe(
    prompt="Text description of the image to generate",
    image_size="Image aspect ratio and size preset",
    num_inference_steps="Number of denoising steps (1-100, default: 28)",
    guidance_scale="How closely to follow prompt (1.0-20.0, default: 2.5)",
    num_images="Number of images to generate (1-4, default: 1)",
    seed="Random seed for reproducibility (leave empty for random)",
    acceleration="Speed vs quality tradeoff",
    expand_prompt="Expand prompt for better results",
    safety_checker="Enable NSFW filtering",
    output_format="Output image format",
    custom_width="Custom width in pixels (512-2048, overrides image_size)",
    custom_height="Custom height in pixels (512-2048, overrides image_size)",
)
@app_commands.choices(image_size=FLUX2_SIZE_CHOICES)
@app_commands.choices(acceleration=FLUX2_ACCELERATION_CHOICES)
@app_commands.choices(output_format=FLUX2_FORMAT_CHOICES)
async def flux2_command(
    interaction: discord.Interaction,
    prompt: str,
    image_size: str = "landscape_4_3",
    num_inference_steps: app_commands.Range[int, 1, 100] = 28,
    guidance_scale: app_commands.Range[float, 1.0, 20.0] = 2.5,
    num_images: app_commands.Range[int, 1, 4] = 1,
    seed: int | None = None,
    acceleration: str = "regular",
    expand_prompt: bool = False,
    safety_checker: bool = True,
    output_format: str = "png",
    custom_width: app_commands.Range[int, 512, 2048] | None = None,
    custom_height: app_commands.Range[int, 512, 2048] | None = None,
) -> None:
    """Generate images using FLUX.2 [dev] via Fal AI."""

    # Determine image size - custom dimensions override preset
    final_size: str | dict[str, int]
    if custom_width is not None and custom_height is not None:
        final_size = {"width": custom_width, "height": custom_height}
        size_display = f"{custom_width}x{custom_height}"
    elif custom_width is not None or custom_height is not None:
        await interaction.response.send_message(
            "❌ **Error:** Both custom_width and custom_height must be provided together.",
            ephemeral=True,
        )
        return
    else:
        final_size = image_size
        size_display = image_size

    # Log the request
    logger.info(
        f"flux2: User {interaction.user} requested image generation: "
        f"prompt='{prompt[:50]}...', size={size_display}, "
        f"steps={num_inference_steps}, acceleration={acceleration}, num_images={num_images}"
    )

    # Defer the response
    await interaction.response.defer()

    try:
        # Update status
        await interaction.edit_original_response(
            content=f"🎨 **Generating {num_images} image(s) with FLUX.2...**\n"
            f"**Prompt:** {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
            f"**Size:** {size_display} | **Steps:** {num_inference_steps} | **Acceleration:** {acceleration}"
        )

        # Generate images
        start_time = time.time()
        result = await services.generate_flux2_image(
            prompt=prompt,
            image_size=final_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            seed=seed,
            acceleration=acceleration,
            enable_prompt_expansion=expand_prompt,
            enable_safety_checker=safety_checker,
            output_format=output_format,
        )
        generation_time = time.time() - start_time

        # Extract images from result
        images = result.get("images", [])
        result_seed = result.get("seed", "unknown")
        used_prompt = result.get("prompt", prompt)
        nsfw_flags = result.get("has_nsfw_concepts", [])

        if not images:
            await interaction.edit_original_response(content="❌ **Error:** No images returned from FLUX.2 API")
            return

        # Download images and create Discord files
        import httpx

        discord_files = []
        async with httpx.AsyncClient() as client:
            for i, img_data in enumerate(images):
                img_url = img_data.get("url")
                if not img_url:
                    continue

                response = await client.get(img_url)
                response.raise_for_status()

                # Create filename
                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:30])
                filename = f"flux2_{safe_prompt}_{i:02d}.{output_format}"

                discord_files.append(
                    discord.File(
                        fp=io.BytesIO(response.content),
                        filename=filename,
                    )
                )

        # Check for NSFW content
        nsfw_warning = ""
        if any(nsfw_flags):
            nsfw_warning = "\n⚠️ **Warning:** Some images may contain NSFW content"

        # Show expanded prompt if it was used
        prompt_info = f"**Prompt:** {prompt}"
        if expand_prompt and used_prompt != prompt:
            truncated = used_prompt[:200] + ("..." if len(used_prompt) > 200 else "")
            prompt_info = f"**Original Prompt:** {prompt}\n**Expanded:** {truncated}"

        # Send the result
        result_message = (
            f"✅ **Generated {len(discord_files)} image(s)** in {generation_time:.1f}s\n"
            f"{prompt_info}\n"
            f"**Size:** {size_display} | **Steps:** {num_inference_steps} | "
            f"**Guidance:** {guidance_scale} | **Seed:** {result_seed}{nsfw_warning}"
        )

        await interaction.edit_original_response(
            content=result_message,
            attachments=discord_files,
        )

    except RuntimeError as re:
        logger.exception(f"flux2: Runtime error - {re}")
        await interaction.edit_original_response(content=f"❌ **Error generating image:** {re}")
    except Exception as e:
        logger.exception(f"flux2: Unexpected error: {e}")
        await interaction.edit_original_response(content=f"❌ **An unexpected error occurred:** {e!s}")


# --- Z-Image-Turbo Command ---


@app_commands.command(name="z", description="Generate an image using Z-Image-Turbo")
@app_commands.describe(
    prompt="Text description of the image to generate",
    width="Image width in pixels (default: 1024)",
    height="Image height in pixels (default: 1024)",
    num_inference_steps="Number of denoising steps (default: 9)",
    seed="Random seed for reproducibility (default: 42)",
    use_oot_lora="Enable the OOT64 LoRA adapter (default: False)",
    oot_lora_scale="OOT64 LoRA weight/scale 0.0-2.0 (default: 0.8)",
    use_hk_lora="Enable the HK (Hollow Knight) LoRA adapter (default: False)",
    hk_lora_scale="HK LoRA weight/scale 0.0-2.0 (default: 0.8)",
    use_mannequin_lora="Enable the Mannequin LoRA adapter (default: False)",
    mannequin_lora_scale="Mannequin LoRA weight/scale 0.0-2.0 (default: 0.8)",
    use_tlou2_lora="Enable the TLOU2 LoRA adapter (default: False)",
    tlou2_lora_scale="TLOU2 LoRA weight/scale 0.0-2.0 (default: 0.8)",
)
async def z_command(
    interaction: discord.Interaction,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: app_commands.Range[int, 1, 100] = 9,
    seed: int = 42,
    use_oot_lora: bool = False,
    oot_lora_scale: app_commands.Range[float, 0.0, 2.0] = 0.8,
    use_hk_lora: bool = False,
    hk_lora_scale: app_commands.Range[float, 0.0, 2.0] = 0.8,
    use_mannequin_lora: bool = False,
    mannequin_lora_scale: app_commands.Range[float, 0.0, 2.0] = 0.8,
    use_tlou2_lora: bool = False,
    tlou2_lora_scale: app_commands.Range[float, 0.0, 2.0] = 0.8,
) -> None:
    """Generate an image using Z-Image-Turbo."""
    await interaction.response.defer(thinking=True)

    lora_parts = []
    if use_oot_lora:
        lora_parts.append(f"oot={oot_lora_scale}")
    if use_hk_lora:
        lora_parts.append(f"hk={hk_lora_scale}")
    if use_mannequin_lora:
        lora_parts.append(f"mannequin={mannequin_lora_scale}")
    if use_tlou2_lora:
        lora_parts.append(f"tlou2={tlou2_lora_scale}")
    lora_info = f", lora=[{', '.join(lora_parts)}]" if lora_parts else ""
    logger.info(
        f"z: User {interaction.user} requested image: "
        f"prompt='{prompt[:50]}...', size={width}x{height}, steps={num_inference_steps}, seed={seed}{lora_info}"
    )

    try:
        image_path = await services.generate_zimage(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            use_oot_lora=use_oot_lora,
            oot_lora_scale=oot_lora_scale,
            use_hk_lora=use_hk_lora,
            hk_lora_scale=hk_lora_scale,
            use_mannequin_lora=use_mannequin_lora,
            mannequin_lora_scale=mannequin_lora_scale,
            use_tlou2_lora=use_tlou2_lora,
            tlou2_lora_scale=tlou2_lora_scale,
        )

        # Build LoRA display string
        lora_displays = []
        if use_oot_lora:
            lora_displays.append(f"OOT: {oot_lora_scale}")
        if use_hk_lora:
            lora_displays.append(f"HK: {hk_lora_scale}")
        if use_mannequin_lora:
            lora_displays.append(f"Mannequin: {mannequin_lora_scale}")
        if use_tlou2_lora:
            lora_displays.append(f"TLOU2: {tlou2_lora_scale}")
        lora_display = f" | **LoRA:** [{', '.join(lora_displays)}]" if lora_displays else ""

        discord_file = discord.File(image_path, filename=image_path.name)
        result_content = (
            f"**Prompt:** {prompt}\n"
            f"**Size:** {width}x{height} | **Steps:** {num_inference_steps} | **Seed:** {seed}{lora_display}"
        )
        await interaction.followup.send(content=result_content, file=discord_file)
    except RuntimeError as re:
        logger.exception(f"z: Runtime error - {re}")
        await interaction.followup.send(f"❌ Error: {re}", ephemeral=True)
    except Exception as e:
        logger.exception(f"z: Unexpected error - {e}")
        await interaction.followup.send(f"❌ An unexpected error occurred: {e!s}", ephemeral=True)
