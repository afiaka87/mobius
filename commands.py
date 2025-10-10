# commands.py

"""
Discord bot slash commands.

This module defines all the slash commands available to the bot,
handling user interactions and calling appropriate services.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    cast,
)

import discord
import fal_client
from discord import app_commands

# Local application/library specific imports
import services
import utils

logger: logging.Logger = logging.getLogger(__name__)

# Command descriptions for the /help command
COMMANDS_INFO: Final[dict[str, str]] = {
    "help": "List all commands and their descriptions.",
    "anthropic": "Chat completion with Anthropic LLM models.",
    "gpt": "Chat with GPT-4o. Supports history. Outputs as a discord embed.",
    "o1": "Generate a response using OpenAI's `o1` models.",
    "say": "Generate speech from text using OpenAI's TTS API.",
    "youtube": "Search YouTube. Returns top result.",
    "temp": "Get the current temperature in Fayetteville, AR.",
    "google": "Uses Google Custom Search API to get results from the web.",
    "flux": "Generate images with FLUX models (e.g., FLUX.1-schnell).",
    "sd3_5_large": "Generate images with Stable Diffusion 3.5 Large models.",
    "rembg": "Remove image background using Rembg.",
    "gptimg": "Generate or edit images using OpenAI's GPT Image model.",
    "t2v": "Generate text-to-video using WAN models.",
    "pixel": "Generate images with SDXL model with LoRA support.",
}

# Type alias for model choice values
ModelChoiceValue = str | float

# Available model choices for various commands
MODEL_CHOICES: Final[dict[str, list[ModelChoiceValue]]] = {
    "anthropic": [
        "claude-3-5-sonnet-20240620",  # Explicitly list claude-3.5-sonnet
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ],
    "gpt": ["gpt-4o", "gpt-4o-mini"],
    "o1": ["o1-preview", "o1-mini", "o1"],  # Assuming 'o1' is a valid model identifier
    "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "speeds": [0.5, 1.0, 1.25, 1.5, 2.0],
    "flux_models": ["fal-ai/flux/dev", "fal-ai/flux/schnell", "fal-ai/flux-pro/new"],
    "sd_models": [
        "fal-ai/stable-diffusion-v35-large/turbo",
        "fal-ai/stable-diffusion-v35-large",
    ],
    "image_sizes": [  # For fal-ai models
        "landscape_4_3",
        "landscape_16_9",
        "portrait_3_4",
        "portrait_9_16",
        "square",
        "square_hd",
    ],
    "gptimg_models": ["gpt-image-1"],  # Simplified to just the GPT Image model
    "gptimg_sizes": [
        "auto",
        "1024x1024",
        "1536x1024",
        "1024x1536",  # GPT Image supported sizes
    ],
    "gptimg_quality": ["auto", "low", "medium", "high"],  # GPT Image quality options
}

# Type aliases for specific string literals used in choices
AnthropicModel = Literal[
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]
GPTModel = Literal["gpt-4o", "gpt-4o-mini"]
O1Model = Literal["o1-preview", "o1-mini", "o1"]
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
TTSSpeed = Literal["0.5", "1.0", "1.25", "1.5", "2.0"]  # Stored as string from choice
FluxModel = Literal["fal-ai/flux/dev", "fal-ai/flux/schnell", "fal-ai/flux-pro/new"]
SDModel = Literal[
    "fal-ai/stable-diffusion-v35-large/turbo", "fal-ai/stable-diffusion-v35-large"
]
FalImageSize = Literal[
    "landscape_4_3",
    "landscape_16_9",
    "portrait_3_4",
    "portrait_9_16",
    "square",
    "square_hd",
]

GPTImageModel = Literal["gpt-image-1"]
GPTImageSize = Literal["auto", "1024x1024", "1536x1024", "1024x1536"]
GPTImageQuality = Literal["auto", "low", "medium", "high"]

SDXLScheduler = Literal["euler", "ddim"]


@app_commands.command(
    name="help", description="List all commands and their descriptions."
)
async def help_command(interaction: discord.Interaction) -> None:
    """Displays a list of all available slash commands and their descriptions."""
    help_lines: list[str] = [f"`/{cmd}`: {desc}" for cmd, desc in COMMANDS_INFO.items()]
    help_message: str = "Here are the available commands:\n\n" + "\n".join(help_lines)
    await interaction.response.send_message(help_message, ephemeral=True)


@app_commands.command(
    name="say",
    description="Generate speech from text using OpenAI's TTS API. Max 4096 chars.",
)
@app_commands.choices(
    voice=[
        app_commands.Choice(name=str(voice_name), value=voice_name)
        for voice_name in MODEL_CHOICES["voices"]
    ],  # type: ignore # Mypy can't infer the type argument
    speed=[
        app_commands.Choice(name=f"{speed_val}x", value=str(speed_val))
        for speed_val in MODEL_CHOICES["speeds"]
    ],
)
async def say_command(
    interaction: discord.Interaction,
    text: str,
    voice: TTSVoice = "onyx",
    speed: TTSSpeed = "1.0",
) -> None:
    """
    Generates speech from the provided text using OpenAI's TTS API and sends it as an audio file.
    """
    if len(text) > 4096:
        await interaction.response.send_message(
            "Text cannot exceed 4096 characters.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        speech_speed: float = float(speed)
        waveform_video_file_path: Path = await services.generate_speech(
            text, voice, speech_speed
        )
        discord_file: discord.File = discord.File(
            waveform_video_file_path, filename=waveform_video_file_path.name
        )
        await interaction.followup.send(
            content=f'Audio for "{text[:50]}..." using voice: {voice}, speed: {speed}x',
            file=discord_file,
        )
    except (
        ValueError
    ) as e:  # Catches float conversion error or errors from services.generate_speech
        logger.exception(f"Error in 'say' command processing: {e}")
        await interaction.followup.send(
            f"An error occurred: {e}. Please check your input.", ephemeral=True
        )
    except Exception:
        logger.exception(f"Unexpected error in 'say' command: {text}, {voice}, {speed}")
        await interaction.followup.send(
            "An unexpected error occurred while generating speech.", ephemeral=True
        )


@app_commands.command(name="flux", description="Generate images with FLUX models.")
@app_commands.choices(
    model=[
        app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["flux_models"]
    ],
    image_size=[
        app_commands.Choice(name=str(s), value=s) for s in MODEL_CHOICES["image_sizes"]
    ],
)
async def flux_command(
    interaction: discord.Interaction,
    prompt: str,
    model: FluxModel = "fal-ai/flux-pro/new",
    image_size: FalImageSize = "square_hd",
    guidance_scale: app_commands.Range[float, 0.0, 10.0] = 3.5,
) -> None:
    """Generates an image using a FLUX model from fal.ai based on the prompt."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        image_url: str = await services.generate_flux_image(
            prompt, model, image_size, guidance_scale
        )
        output: str = (
            f"Prompt: **`{discord.utils.escape_markdown(prompt)}`**\n"
            f"Model: **`{model}`**\n"
            f"Image Size: **`{image_size}`**\n"
            f"Guidance: **`{guidance_scale}`**\n\n{image_url}"
        )
        await interaction.followup.send(content=output)
    except Exception:
        logger.exception(f"Error generating FLUX image for prompt: {prompt}")
        await interaction.followup.send(
            "An error occurred while generating the image with FLUX.", ephemeral=True
        )


@app_commands.command(
    name="sd3_5_large", description="Generate images with Stable Diffusion 3.5 Large."
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["sd_models"]
    ]
)
async def sd3_5_large_command(
    interaction: discord.Interaction,
    prompt: str,
    model: SDModel = "fal-ai/stable-diffusion-v35-large",
    guidance_scale: app_commands.Range[float, 0.0, 10.0] = 4.5,  # Default for SD3.5
) -> None:
    """Generates an image using Stable Diffusion 3.5 Large from fal.ai."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        # SD3.5 typically uses square_hd or similar high-res sizes
        image_url: str = await services.generate_flux_image(
            prompt, model, "square_hd", guidance_scale
        )
        output: str = (
            f"Prompt: **`{discord.utils.escape_markdown(prompt)}`**\n"
            f"Model: **`{model}`**\n"
            f"Guidance: **`{guidance_scale}`**\n\n{image_url}"
        )
        await interaction.followup.send(content=output)
    except Exception:
        logger.exception(f"Error generating SD3.5 image for prompt: {prompt}")
        await interaction.followup.send(
            "An error occurred while generating the image with SD3.5.", ephemeral=True
        )


@app_commands.command(
    name="rembg",
    description="Remove background from an image using fal.ai/imageutils/rembg.",
)
async def rembg_command(
    interaction: discord.Interaction, image: discord.Attachment
) -> None:
    """Removes the background from the provided image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.response.send_message(
            "Please upload a valid image file (PNG, JPG, WEBP).", ephemeral=True
        )
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
        await interaction.followup.send(
            "An error occurred while removing the image background.", ephemeral=True
        )


@app_commands.command(
    name="anthropic", description="Chat completion with Anthropic Claude models."
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["anthropic"]
    ]
)
async def anthropic_command(
    interaction: discord.Interaction,
    prompt: str,
    max_tokens: app_commands.Range[int, 1, 4096] = 1024,  # Adjusted max_tokens
    suppress_embeds: bool = True,
    model: AnthropicModel = "claude-3-5-sonnet-20240620",
) -> None:
    """Gets a chat completion from an Anthropic Claude model."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        message_text: str = await services.anthropic_chat_completion(
            prompt=prompt, max_tokens=max_tokens, model=model
        )

        # Format with escaped prompt for safety
        formatted_response: str = (
            f"### _{interaction.user.display_name}_:\n\n"
            f"```\n{discord.utils.escape_markdown(prompt)}\n```\n"
            f"### {model}:\n\n{message_text}"
        )

        if len(formatted_response) >= 2000:
            temp_file_path: Path = utils.create_temp_file(formatted_response, ".md")
            discord_file: discord.File = discord.File(
                temp_file_path, filename="response.md"
            )
            await interaction.followup.send(
                content="Response too long, sending as a file.",
                file=discord_file,
            )
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.exception(f"Error deleting temporary file {temp_file_path}: {e}")
        else:
            await interaction.followup.send(
                content=formatted_response, suppress_embeds=suppress_embeds
            )
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
        discord_file: discord.File = discord.File(
            temp_file_path, filename=f"{model_name}_response.md"
        )

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
        await interaction.followup.send(
            "Error sending long response as a file.", ephemeral=True
        )
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
@app_commands.choices(
    model_name=[app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["gpt"]]
)
async def gpt_command(
    interaction: discord.Interaction,
    prompt: str,
    seed: int | None = None,  # OpenAI API seed is Optional
    model_name: GPTModel = "gpt-4o-mini",
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

        assistant_response: str = await services.gpt_chat_completion(
            history, model_name, api_seed
        )

        if len(assistant_response) >= 4000:  # Embed description limit is 4096
            await _handle_long_response(
                interaction, assistant_response, prompt, model_name, api_seed
            )
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


@app_commands.command(
    name="o1",
    description="Generate a response using OpenAI's o1 series models (stateless).",
)
@app_commands.choices(
    model_name=[app_commands.Choice(name=str(m), value=m) for m in MODEL_CHOICES["o1"]]
)
async def o1_command(
    interaction: discord.Interaction,
    prompt: str,
    model_name: O1Model = "o1",  # Default to 'o1' if it's a general model
    seed: int | None = None,
) -> None:
    """Generates a response from an OpenAI o1 series model (stateless)."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        api_seed: int | None = int(seed) if seed is not None and seed != -1 else None
        assistant_response: str = await services.gpt_chat_completion(
            messages, model_name, api_seed
        )

        if len(assistant_response) >= 4000:  # Embed description limit
            await _handle_long_response(
                interaction, assistant_response, prompt, model_name, api_seed
            )
        else:
            embed: discord.Embed = discord.Embed(
                title=f"Response from {model_name}",
                description=assistant_response,
                color=discord.Color.purple(),
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
        logger.exception(f"Error with o1 command for prompt: {prompt}")
        await interaction.followup.send(
            "An error occurred while communicating with the OpenAI o1 API.",
            ephemeral=True,
        )


@app_commands.command(
    name="youtube", description="Search YouTube and return the top video result."
)
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
        result: dict[str, Any] = await services.get_top_youtube_result(
            query, youtube_api_key
        )
        if "error" in result:
            await interaction.followup.send(
                f"YouTube search error: {result['error']}", ephemeral=True
            )
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
        await interaction.followup.send(
            "An error occurred during the YouTube search.", ephemeral=True
        )


@app_commands.command(
    name="temp", description="Get the current temperature in Fayetteville, AR."
)
async def temp_command(interaction: discord.Interaction) -> None:
    """Fetches and displays the current temperature for Fayetteville, AR."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        temperature_info: str = await services.temp_callback()
        await interaction.followup.send(temperature_info)
    except Exception:
        logger.exception("Error fetching temperature data.")
        await interaction.followup.send(
            "An error occurred while fetching temperature data.", ephemeral=True
        )


@app_commands.command(
    name="google", description="Search the web using Google Custom Search API."
)
async def google_command(interaction: discord.Interaction, query: str) -> None:
    """Performs a Google search using the Custom Search API and returns top results."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    if not os.getenv("GOOGLE_SEARCH_API_KEY") or not os.getenv(
        "GOOGLE_SEARCH_ENGINE_ID"
    ):
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
        await interaction.followup.send(
            "An error occurred during the Google search.", ephemeral=True
        )


@app_commands.command(
    name="t2v", description="Generate a short video from text using WAN model."
)
@app_commands.describe(
    text="Text prompt for video generation.",
    length="Number of frames in the video (e.g., 33).",
    steps="Number of diffusion steps (e.g., 30).",
    seed="Seed for generation (0 for random).",
)
async def t2v_command(
    interaction: discord.Interaction,
    text: str,
    length: app_commands.Range[int, 1, 100] = 33,
    steps: app_commands.Range[int, 1, 100] = 30,
    seed: int = 0,  # 0 typically means random in many generation systems
) -> None:
    """Generates a text-to-video using a ComfyUI workflow with WAN models."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        video_path: Path = await services.t2v(
            text=text, length=length, steps=steps, seed=seed
        )
        discord_file: discord.File = discord.File(video_path, filename=video_path.name)
        await interaction.followup.send(
            f'Video generated for prompt: "{discord.utils.escape_markdown(text[:50])}..."',
            file=discord_file,
        )
    except Exception:
        logger.exception(f"Error generating t2v for prompt: {text}")
        await interaction.followup.send(
            "An error occurred while generating the video.", ephemeral=True
        )


@app_commands.command(
    name="gptimg",
    description="Generate or edit images using OpenAI's GPT Image model (saves as PNG).",
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=str(m), value=m)
        for m in MODEL_CHOICES["gptimg_models"]
    ],
    size=[
        app_commands.Choice(name=str(s), value=s) for s in MODEL_CHOICES["gptimg_sizes"]
    ],
    quality=[
        app_commands.Choice(name=str(q), value=q)
        for q in MODEL_CHOICES["gptimg_quality"]
    ],
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
        img
        for img in [edit_image1, edit_image2, edit_image3, edit_image4, edit_image5]
        if img is not None
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
        f"**Settings:** Size: {size} | Quality: {quality}"
        + (f" | Images: {len(edit_images)}" if is_editing else "")
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
                if not edit_img.content_type or not edit_img.content_type.startswith(
                    "image/"
                ):
                    raise ValueError(f"edit_image{i+1} is not a valid image type.")

                with tempfile.NamedTemporaryFile(
                    suffix=Path(edit_img.filename).suffix or ".png", delete=False
                ) as tmp_file:
                    await edit_img.save(Path(tmp_file.name))
                    temp_image_paths.append(Path(tmp_file.name))

            # Validate and save mask_image if provided
            if mask_image:
                if (
                    not mask_image.content_type
                    or not mask_image.content_type.startswith("image/")
                ):
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
            max_updates: int = (
                max_generation_time_seconds // progress_update_interval_seconds + 5
            )

            while update_count < max_updates and not operation_complete_event.is_set():
                elapsed_time: int = int(time.time() - start_time)
                minutes, seconds = divmod(elapsed_time, 60)

                if elapsed_time > max_generation_time_seconds:
                    logger.warning(
                        f"gptimg: Generation timeout suspected after {elapsed_time}s."
                    )
                    break

                progress_percentage: float = min(
                    100.0, (elapsed_time / estimated_completion_time_seconds) * 100
                )
                segments: int = 10
                filled_segments: int = int((progress_percentage / 100) * segments)
                progress_bar: str = "⬛" * filled_segments + "⬜" * (
                    segments - filled_segments
                )
                est_remaining: int = max(
                    0, estimated_completion_time_seconds - elapsed_time
                )
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
                    logger.warning(
                        "gptimg: Progress message not found, likely deleted by user."
                    )
                    initial_message = None
                    break
                except discord.HTTPException as e_http:
                    logger.exception(
                        f"gptimg: Error updating progress message: {e_http}"
                    )

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

            logger.info(
                f"gptimg: Progress update loop finished. Updates: {update_count}"
            )
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
            generated_image_path = await asyncio.wait_for(
                generation_task, timeout=max_generation_time_seconds
            )
        except TimeoutError:
            logger.exception(
                f"gptimg: Image {operation_type} timed out after {max_generation_time_seconds}s."
            )
            if generation_task and not generation_task.done():
                generation_task.cancel()
                try:
                    await generation_task
                except asyncio.CancelledError:
                    logger.info(
                        "gptimg: Generation task successfully cancelled on timeout."
                    )
                except Exception as e_cancel:
                    logger.exception(
                        f"gptimg: Error during generation task cancellation: {e_cancel}"
                    )
            raise RuntimeError(
                f"Image {operation_type} timed out. The API may be experiencing delays."
            )
        finally:
            operation_complete_event.set()
            if update_task and not update_task.done():
                try:
                    await asyncio.wait_for(
                        update_task, timeout=progress_update_interval_seconds + 1
                    )
                except TimeoutError:
                    logger.warning(
                        "gptimg: Update task did not finish cleanly after generation."
                    )
                    if not update_task.done():
                        update_task.cancel()
                except Exception as e_await_update:
                    logger.exception(
                        f"gptimg: Error awaiting update task: {e_await_update}"
                    )

        total_time_seconds: float = time.time() - start_time
        mins, secs = divmod(int(total_time_seconds), 60)

        if not generated_image_path or not generated_image_path.exists():
            raise RuntimeError(
                "Image generation service did not return a valid image file."
            )

        final_discord_file: discord.File = discord.File(
            generated_image_path, filename=generated_image_path.name
        )
        final_message_content: str = (
            f"✅ Image {operation_type} complete! (Took {mins}m {secs}s)\n\n"
            f"**Prompt:** {discord.utils.escape_markdown(prompt)}\n"
            f"**Settings:** Model: {model} | Size: {size} | Quality: {quality}"
            + (f" | Images: {len(edit_images)}" if is_editing else "")
            + (f" | Transparent: {transparent_background}" if not is_editing else "")
            + f"\n**Format:** PNG (supports transparency)"
        )

        await interaction.followup.send(
            content=final_message_content, file=final_discord_file
        )

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
        logger.exception(
            f"gptimg: Unexpected error during {operation_type} for prompt: {prompt}"
        )
        await interaction.followup.send(
            f"❌ An unexpected error occurred: {e!s}", ephemeral=True
        )
    finally:
        # Clean up temporary files
        for p in temp_image_paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError as e_unlink:
                    logger.exception(
                        f"gptimg: Error deleting temp image file {p}: {e_unlink}"
                    )
        if temp_mask_path and temp_mask_path.exists():
            try:
                temp_mask_path.unlink()
            except OSError as e_unlink:
                logger.exception(
                    f"gptimg: Error deleting temp mask file {temp_mask_path}: {e_unlink}"
                )

        operation_complete_event.set()
        # Final cleanup for any remaining tasks
        tasks_to_cancel = [
            t for t in [update_task, generation_task] if t and not t.done()
        ]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                logger.info("gptimg: Remaining tasks cancelled in final cleanup.")
            except Exception as e_gather:
                logger.exception(f"gptimg: Error during final task cleanup: {e_gather}")

        return None


@app_commands.command(
    name="pixel", description="Generate images with SDXL model with LoRA support."
)
@app_commands.choices(
    scheduler=[
        app_commands.Choice(name="euler", value="euler"),
        app_commands.Choice(name="ddim", value="ddim"),
    ]
)
async def pixel_command(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: str | None = None,
    width: app_commands.Range[int, 256, 2048] = 1024,
    height: app_commands.Range[int, 256, 2048] = 1024,
    num_inference_steps: app_commands.Range[int, 1, 150] = 30,
    guidance_scale: app_commands.Range[float, 1.0, 20.0] = 7.5,
    num_images_per_prompt: app_commands.Range[int, 1, 4] = 1,
    seed: int | None = None,
    lora_weight: app_commands.Range[float, 0.0, 2.0] = 1.0,
    scheduler: SDXLScheduler = "euler",
) -> None:
    """Generates images using the SDXL model with optional LoRA support."""
    await interaction.response.defer(ephemeral=False, thinking=True)
    start_time: float = time.time()

    try:
        image_paths: list[Path] = await services.generate_sdxl_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed,
            lora_weight=lora_weight,
            scheduler=scheduler,
        )

        total_time_seconds: float = time.time() - start_time
        mins, secs = divmod(int(total_time_seconds), 60)

        # Prepare Discord files
        discord_files: list[discord.File] = [
            discord.File(img_path, filename=img_path.name) for img_path in image_paths
        ]

        # Build response message
        response_lines: list[str] = [
            f"✅ Generated {len(image_paths)} image(s) with SDXL! (Took {mins}m {secs}s)",
            f"**Prompt:** {discord.utils.escape_markdown(prompt)}",
            f"**Size:** {width}x{height} | **Steps:** {num_inference_steps} | **Guidance:** {guidance_scale}",
            f"**Scheduler:** {scheduler} | **LoRA Weight:** {lora_weight}",
        ]

        if negative_prompt:
            response_lines.append(
                f"**Negative Prompt:** {discord.utils.escape_markdown(negative_prompt)}"
            )

        if seed is not None:
            response_lines.append(f"**Seed:** {seed}")

        response_message: str = "\n".join(response_lines)

        await interaction.followup.send(content=response_message, files=discord_files)

    except ValueError as ve:
        logger.exception(f"pixel: Configuration error - {ve}")
        await interaction.followup.send(
            f"❌ Configuration error: {ve}\n\nPlease ensure SDXL_API_URL is set in your environment.",
            ephemeral=True,
        )
    except Exception as e:
        logger.exception(f"pixel: Error generating image for prompt: {prompt}")
        await interaction.followup.send(
            f"❌ An error occurred while generating the image: {e!s}", ephemeral=True
        )
