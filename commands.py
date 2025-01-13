from pathlib import Path
import os
from typing import Optional

import discord
from discord import app_commands
import fal_client

import services
import utils

# Command configurations
COMMANDS_INFO = {
    "anthropic": "Chat completion with Anthropic LLM models.",
    "gpt": "Chat with GPT-4o. Supports history. Outputs as a discord embed.",
    "o1": "Generate a response using OpenAI's `o1` models.",
    "say": "Generate speech from text using OpenAI's TTS API.",
}
# Model choices

MODEL_CHOICES = {
    "anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ],
    "gpt": ["gpt-4o", "gpt-4o-mini"],
    "o1": ["o1-preview", "o1-mini"],
    "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "speeds": [0.5, 1.0, 1.25, 1.5, 2.0],
    "flux_models": ["fal-ai/flux/dev", "fal-ai/flux/schnell", "fal-ai/flux-pro/new"],
    "sd_models": [
        "fal-ai/stable-diffusion-v35-large/turbo",
        "fal-ai/stable-diffusion-v35-large",
    ],
    "image_sizes": [
        "landscape_4_3",
        "landscape_16_9",
        "portrait_3_4",
        "portrait_9_16",
        "square",
        "square_hd",
    ],
}


async def handle_long_response(
    interaction: discord.Interaction,
    content: str,
    prompt: str,
    model_name: str,
    seed: Optional[int] = None,
) -> None:
    filename = utils.create_temp_file(content, ".md")
    file = discord.File(filename)

    embed = discord.Embed(
        description="The response was too long. Please see the attached file."
    )
    embed.add_field(name="Prompt", value=prompt[:1000], inline=False)
    embed.add_field(name="Model Name", value=model_name)
    if seed is not None:
        embed.add_field(name="Seed", value=seed)

    await interaction.followup.send(embed=embed, file=file)
    os.remove(filename)


# Command Definitions
@app_commands.command(
    name="help", description="List all commands and their descriptions."
)
async def help(interaction: discord.Interaction):
    help_message = "Here are the available commands:\n\n"
    help_message += "\n".join(
        f"`/{cmd}`: {desc}" for cmd, desc in COMMANDS_INFO.items()
    )
    await interaction.response.send_message(help_message)


@app_commands.command(
    name="say",
    description="Generate speech from text using OpenAI's TTS API. Maximum length of text is 4096 characters.",
)
@app_commands.choices(
    voice=[
        app_commands.Choice(name=voice, value=voice)
        for voice in MODEL_CHOICES["voices"]
    ],
    speed=[
        app_commands.Choice(name=f"{speed}x", value=str(speed))
        for speed in MODEL_CHOICES["speeds"]
    ],
)
async def say(
    interaction: discord.Interaction, text: str, voice: str = "onyx", speed: str = "1.0"
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    waveform_video_file_path = Path(
        await services.generate_speech(text, voice, float(speed))
    )
    await interaction.followup.send(
        content=f"Audio response using voice: {voice}",
        file=discord.File(waveform_video_file_path),
    )


@app_commands.command(name="flux", description="Run txt to img w FLUX SCHNELL")
@app_commands.choices(
    model=[app_commands.Choice(name=m, value=m) for m in MODEL_CHOICES["flux_models"]],
    image_size=[
        app_commands.Choice(name=s, value=s) for s in MODEL_CHOICES["image_sizes"]
    ],
)
async def flux(
    interaction: discord.Interaction,
    prompt: str,
    model: str = "fal-ai/flux-pro/new",
    image_size: str = "square_hd",
    guidance_scale: float = 3.5,
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    image_url = await services.generate_flux_image(
        prompt, model, image_size, guidance_scale
    )
    output = f"Prompt: **`{prompt}`** \nModel: **`{model}`**\nImage Size: **`{image_size}`**\n\n{image_url}"
    await interaction.followup.send(content=output)


@app_commands.command(name="sd3_5_large", description="Run txt to img w SD3.5 LARGE")
@app_commands.choices(
    model=[app_commands.Choice(name=m, value=m) for m in MODEL_CHOICES["sd_models"]]
)
async def sd3_5_large(
    interaction: discord.Interaction,
    prompt: str,
    model: str = "fal-ai/stable-diffusion-v35-large",
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    image_url = await services.generate_flux_image(prompt, model, "square_hd", 4.5)
    output = f"Prompt: **`{prompt}`** \nModel: **`{model}`**\n\n{image_url}"
    await interaction.followup.send(content=output)


@app_commands.command(name="rembg", description="Remove image background using Rembg")
async def rembg(interaction: discord.Interaction, image_url: discord.Attachment):
    await interaction.response.defer(ephemeral=False, thinking=True)
    result = await fal_client.subscribe_async(
        "fal-ai/imageutils/rembg",
        arguments={"image_url": image_url.url},
        with_logs=True,
    )
    await interaction.followup.send(result["image"]["url"])


@app_commands.command(
    name="anthropic", description="Chat completion with Anthropic LLM models."
)
@app_commands.choices(
    model=[app_commands.Choice(name=m, value=m) for m in MODEL_CHOICES["anthropic"]]
)
async def anthropic(
    interaction: discord.Interaction,
    prompt: str,
    max_tokens: int = 1024,
    model: str = "claude-3-5-sonnet-20240620",
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    message_text = await services.anthropic_chat_completion(prompt, max_tokens, model)

    if len(message_text) >= 2000:
        await interaction.followup.send(
            content="Response too long, sending as a file.",
            file=discord.File(
                utils.create_temp_file(message_text), filename="response.txt"
            ),
        )
    else:
        formatted_response = f"### _{interaction.user.name}_: \n\n```txt\n{prompt}\n```\n### anthropic:\n\n {message_text}"
        await interaction.followup.send(content=formatted_response)


@app_commands.command(
    name="gpt",
    description="Chat with GPT-4o. Supports history. Outputs as a discord embed.",
)
@app_commands.choices(
    model_name=[app_commands.Choice(name=m, value=m) for m in MODEL_CHOICES["gpt"]]
)
async def gpt(
    interaction: discord.Interaction,
    prompt: str,
    seed: int = -1,
    model_name: str = "gpt-4o-mini",
):
    await interaction.response.defer(ephemeral=False, thinking=True)

    history = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    assistant_response = await services.gpt_chat_completion(history, model_name, seed)

    embed = discord.Embed(description=assistant_response)
    embed.add_field(name="Prompt", value=prompt[:1000], inline=False)
    embed.add_field(name="Model", value=model_name)

    await interaction.followup.send(embed=embed)


@app_commands.command(
    name="o1",
    description="Generate a response using GPT without history (o1-preview or o1-mini only)",
)
@app_commands.choices(
    model_name=[app_commands.Choice(name=m, value=m) for m in MODEL_CHOICES["o1"]]
)
async def o1(
    interaction: discord.Interaction,
    prompt: str,
    model_name: str = "o1-mini",
    seed: Optional[int] = None,
):
    await interaction.response.defer(ephemeral=False, thinking=True)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    assistant_response = await services.gpt_chat_completion(messages, model_name, seed)

    if len(assistant_response) >= 4096:
        await handle_long_response(
            interaction, assistant_response, prompt, model_name, seed
        )
    else:
        embed = discord.Embed(description=assistant_response)
        embed.add_field(name="Prompt", value=prompt[:1000], inline=False)
        embed.add_field(name="Model", value=model_name)
        await interaction.followup.send(embed=embed)


# Utility Commands
@app_commands.command(name="youtube", description="Search youtube. Returns top result.")
async def youtube(interaction: discord.Interaction, query: str):
    await interaction.response.defer(ephemeral=False, thinking=True)
    result = await services.get_top_youtube_result_httpx(
        query, os.getenv("YOUTUBE_API_KEY")
    )
    await interaction.followup.send(
        result.get("error") or f"https://www.youtube.com/watch?v={result['videoId']}"
    )


@app_commands.command(name="temp", description="Get the temperature.")
async def temp(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False, thinking=True)
    await interaction.followup.send(await services.temp_callback())


@app_commands.command(
    name="google",
    description="Uses the google custom search api to get results from the web.",
)
async def google(interaction: discord.Interaction, query: str):
    await interaction.response.defer(ephemeral=False, thinking=True)
    await interaction.followup.send(await services.google_search(query))
