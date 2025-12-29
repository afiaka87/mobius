# prefix_adapter.py
"""
Prefix command adapter for Discord bot.

Converts slash commands (e.g., /z prompt: a moon shaped pool) to
dot-prefix commands (e.g., .z a moon shaped pool).

Coexists with existing slash commands - both work simultaneously.
"""

import logging
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

import services

logger = logging.getLogger(__name__)

PREFIX = "."


@dataclass
class PrefixCommand:
    """Metadata for a prefix command."""

    handler: Callable[[discord.Message, str], Coroutine[Any, Any, None]]
    arg_name: str | None = None  # None for zero-arg commands
    description: str = ""


# Registry of prefix commands
COMMANDS: dict[str, PrefixCommand] = {}


CommandHandler = Callable[[discord.Message, str], Coroutine[Any, Any, None]]


def register_command(
    name: str,
    arg_name: str | None = None,
    description: str = "",
) -> Callable[[CommandHandler], CommandHandler]:
    """Decorator to register a prefix command."""

    def decorator(func: CommandHandler) -> CommandHandler:
        COMMANDS[name] = PrefixCommand(
            handler=func,
            arg_name=arg_name,
            description=description,
        )
        return func

    return decorator


async def send_error(message: discord.Message, title: str, description: str) -> None:
    """Send an error embed as a reply."""
    embed = discord.Embed(
        title=title,
        description=description,
        color=discord.Color.red(),
    )
    await message.reply(embed=embed)


async def handle_prefix_command(message: discord.Message) -> bool:
    """
    Handle a potential prefix command.

    Returns True if the message was handled as a command.
    """
    if not message.content.startswith(PREFIX):
        return False

    content = message.content[len(PREFIX) :]
    parts = content.split(maxsplit=1)
    cmd_name = parts[0].lower() if parts else ""
    arg_text = parts[1].strip() if len(parts) > 1 else ""

    if cmd_name not in COMMANDS:
        return False

    cmd = COMMANDS[cmd_name]

    # Check for missing required argument
    if cmd.arg_name and not arg_text:
        await send_error(
            message,
            "Missing Argument",
            f"Usage: `{PREFIX}{cmd_name} <{cmd.arg_name}>`",
        )
        return True

    try:
        async with message.channel.typing():
            await cmd.handler(message, arg_text)
    except Exception as e:
        logger.exception(f"Error in prefix command '{cmd_name}': {e}")
        await send_error(message, "Error", str(e))

    return True


# =============================================================================
# Command Handlers
# =============================================================================


@register_command("help", description="List available prefix commands")
async def cmd_help(message: discord.Message, _arg: str) -> None:
    """List all available prefix commands."""
    lines = ["**Available Prefix Commands**\n"]

    for name, cmd in sorted(COMMANDS.items()):
        usage = f"`{PREFIX}{name} <{cmd.arg_name}>`" if cmd.arg_name else f"`{PREFIX}{name}`"
        lines.append(f"{usage} - {cmd.description}")

    embed = discord.Embed(
        title="Prefix Commands",
        description="\n".join(lines),
        color=discord.Color.blue(),
    )
    await message.reply(embed=embed)


@register_command("temp", description="Get current temperature in Fayetteville, AR")
async def cmd_temp(message: discord.Message, _arg: str) -> None:
    """Get current temperature."""
    result = await services.temp_callback()
    await message.reply(result)


@register_command("sd-list", description="List available Stable Diffusion checkpoints")
async def cmd_sd_list(message: discord.Message, _arg: str) -> None:
    """List available SD checkpoints."""
    experiments = await services.list_sd_experiments()

    if not experiments:
        await message.reply("No experiment runs found.")
        return

    message_parts = ["**Available Stable Diffusion Checkpoints**\n"]

    for experiment_run in experiments:
        try:
            checkpoints = await services.list_sd_checkpoints(experiment_run)
            message_parts.append(f"\n**{experiment_run}**")
            if checkpoints:
                for checkpoint in checkpoints:
                    full_path = f"checkpoints/{experiment_run}/{checkpoint}"
                    message_parts.append(f"  `{checkpoint}` -> `{full_path}`")
            else:
                message_parts.append("  _No checkpoints found_")
        except Exception as e:
            logger.warning(f"sd-list: Error listing checkpoints for {experiment_run}: {e}")
            message_parts.append(f"  _Error: {e!s}_")

    message_parts.append(f"\n**Usage:** `{PREFIX}sd-load <checkpoint_path>`")

    full_message = "\n".join(message_parts)

    # Discord 2000 char limit
    if len(full_message) <= 2000:
        await message.reply(full_message)
    else:
        # Send in chunks
        chunks = []
        current = message_parts[0]
        for part in message_parts[1:]:
            if len(current) + len(part) + 1 <= 2000:
                current += "\n" + part
            else:
                chunks.append(current)
                current = part
        if current:
            chunks.append(current)

        for chunk in chunks:
            await message.channel.send(chunk)


@register_command("say", arg_name="text", description="Text-to-speech using OpenAI")
async def cmd_say(message: discord.Message, arg: str) -> None:
    """Generate speech from text."""
    video_path = await services.generate_speech(arg, "echo", 1.0)
    await message.reply(file=discord.File(video_path))
    # Cleanup
    Path(video_path).unlink(missing_ok=True)


@register_command("claude", arg_name="prompt", description="Chat with Claude")
async def cmd_claude(message: discord.Message, arg: str) -> None:
    """Chat with Anthropic Claude."""
    result = await services.anthropic_chat_completion(arg)
    # Discord 2000 char limit
    if len(result) <= 2000:
        await message.reply(result)
    else:
        # Send as file
        await message.reply(
            "Response too long, sent as file:",
            file=discord.File(
                fp=__import__("io").BytesIO(result.encode()),
                filename="response.txt",
            ),
        )


@register_command("gpt", arg_name="prompt", description="Chat with GPT")
async def cmd_gpt(message: discord.Message, arg: str) -> None:
    """Chat with OpenAI GPT."""
    messages = [{"role": "user", "content": arg}]
    result = await services.gpt_chat_completion(messages, "gpt-4o-mini")
    # Discord 2000 char limit
    if len(result) <= 2000:
        await message.reply(result)
    else:
        await message.reply(
            "Response too long, sent as file:",
            file=discord.File(
                fp=__import__("io").BytesIO(result.encode()),
                filename="response.txt",
            ),
        )


@register_command("youtube", arg_name="query", description="Search YouTube")
async def cmd_youtube(message: discord.Message, arg: str) -> None:
    """Search YouTube and return top result."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YouTube API key is not configured.")

    result = await services.get_top_youtube_result(arg, api_key)

    if not result:
        await message.reply(f"No YouTube results found for '{arg}'.")
        return

    video_url = f"https://www.youtube.com/watch?v={result.get('video_id', '')}"
    title = result.get("title", "Unknown")
    channel = result.get("channel_title", "Unknown")

    await message.reply(f"**{title}**\nChannel: {channel}\n{video_url}")


@register_command("google", arg_name="query", description="Search Google")
async def cmd_google(message: discord.Message, arg: str) -> None:
    """Search Google."""
    if not os.getenv("GOOGLE_SEARCH_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        raise ValueError("Google Search is not configured.")

    result = await services.google_search(arg)
    if result:
        await message.reply(f"Google search results for '{arg}':\n{result}")
    else:
        await message.reply(f"No Google results found for '{arg}'.")


@register_command("sd-load", arg_name="path", description="Load SD checkpoint")
async def cmd_sd_load(message: discord.Message, arg: str) -> None:
    """Load a Stable Diffusion checkpoint."""
    # Convert "base" to None for the API
    api_checkpoint_path = None if arg.lower() == "base" else arg

    result = await services.load_sd_checkpoint(api_checkpoint_path)

    checkpoint_name = result.get("checkpoint_name", "Unknown")
    is_lora = result.get("is_lora", False)
    status = result.get("status", "Loaded successfully")
    checkpoint_type = "LoRA" if is_lora else "Full Model"

    await message.reply(
        f"**Checkpoint loaded successfully**\n"
        f"**Name:** {checkpoint_name}\n"
        f"**Type:** {checkpoint_type}\n"
        f"**Status:** {status}"
    )


@register_command("gptimg", arg_name="prompt", description="Generate image with GPT")
async def cmd_gptimg(message: discord.Message, arg: str) -> None:
    """Generate an image using GPT Image model."""
    image_path = await services.generate_gpt_image(
        prompt=arg,
        user=str(message.author.id),
    )
    await message.reply(file=discord.File(image_path))
    # Cleanup
    Path(image_path).unlink(missing_ok=True)


@register_command("k5", arg_name="prompt", description="Generate video with Kandinsky-5")
async def cmd_k5(message: discord.Message, arg: str) -> None:
    """Generate a video using Kandinsky-5."""
    video_path = await services.generate_kandinsky5_video(prompt=arg)
    await message.reply(file=discord.File(video_path))
    # Cleanup
    Path(video_path).unlink(missing_ok=True)


@register_command("sd", arg_name="prompt", description="Generate image with Stable Diffusion")
async def cmd_sd(message: discord.Message, arg: str) -> None:
    """Generate images using Stable Diffusion."""
    image_paths = await services.generate_sd_images(prompt=arg)

    # Send all generated images
    files = [discord.File(p) for p in image_paths]
    await message.reply(files=files)

    # Cleanup
    for p in image_paths:
        Path(p).unlink(missing_ok=True)


@register_command("flux2", arg_name="prompt", description="Generate image with FLUX 2")
async def cmd_flux2(message: discord.Message, arg: str) -> None:
    """Generate images using FLUX 2."""
    result = await services.generate_flux2_image(prompt=arg)

    images = result.get("images", [])
    if not images:
        await message.reply("No images were generated.")
        return

    # Download and send images
    import httpx

    files = []
    for i, img_data in enumerate(images):
        img_url = img_data.get("url")
        if img_url:
            async with httpx.AsyncClient() as client:
                resp = await client.get(img_url)
                resp.raise_for_status()
                files.append(
                    discord.File(
                        fp=__import__("io").BytesIO(resp.content),
                        filename=f"flux2_{i}.png",
                    )
                )

    if files:
        await message.reply(files=files)
    else:
        await message.reply("Failed to download generated images.")


@register_command("z", arg_name="prompt", description="Generate image with Z-Image-Turbo")
async def cmd_z(message: discord.Message, arg: str) -> None:
    """Generate an image using Z-Image-Turbo."""
    image_path = await services.generate_zimage(prompt=arg)
    await message.reply(file=discord.File(image_path))
    # Cleanup
    Path(image_path).unlink(missing_ok=True)


# =============================================================================
# Setup function for bot integration
# =============================================================================


def setup_prefix_commands(bot: commands.Bot) -> None:
    """
    Register the prefix command handler with the bot.

    Call this after creating the bot instance but before running it.
    """

    @bot.event
    async def on_message(message: discord.Message) -> None:
        # Ignore messages from bots
        if message.author.bot:
            return

        # Try to handle as prefix command
        handled = await handle_prefix_command(message)

        # If not handled, process other commands (if any)
        if not handled:
            await bot.process_commands(message)

    logger.info(f"Registered {len(COMMANDS)} prefix commands with prefix '{PREFIX}'")
