from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from services import chatgpt
from utils import audio_utils, video_utils


@app_commands.command(
    name="wav",
    description="Generate a song or sound from text using stable audio open.",
)
async def wav(
    interaction: discord.Interaction,
    prompt: str,
    duration: int = 10,
    steps: int = 100,
    cfg_scale: int = 7,
    sigma_min: float = 0.3,
    sigma_max: int = 500,
    sampler_type: str = "dpmpp-3m-sde",
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    output, sample_rate = await audio_utils.load_and_run_sao_model(
        prompt, duration, steps, cfg_scale, sigma_min, sigma_max, sampler_type
    )
    waveform_video_filename = video_utils.convert_audio_to_waveform_video(
        output, sample_rate, duration
    )
    await interaction.followup.send(
        content=f"Generated audio from prompt: `{prompt}`\nTurn down the volume before playing the audio.\n",
        files=[discord.File(waveform_video_filename)],
    )


@app_commands.command(
    name="say",
    description="Generate speech from text using OpenAI's TTS API. Maximum length of text is 4096 characters.",
)
@app_commands.choices(
    voice=[
        app_commands.Choice(name=voice, value=voice)
        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    ],
    speed=[
        app_commands.Choice(name=f"{speed}x", value=str(speed))
        for speed in [0.5, 1.0, 1.25, 1.5, 2.0]
    ],
)
async def say(
    interaction: discord.Interaction,
    text: str,
    voice: str = "onyx",
    speed: str = "1.0",
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    speech_file_path: str = await chatgpt.generate_speech(text, voice, float(speed))
    speech_file_path = Path(speech_file_path)
    # name of video file should be same as audio file but with .mp4 extension
    video_file_path = speech_file_path.with_suffix(".mp4")
    video_file_path = video_utils.convert_audio_to_waveform_video(
        speech_file_path, video_file_path
    )
    await interaction.followup.send(
        content=f"Audio response using voice: {voice}",
        file=discord.File(video_file_path),
    )


def setup(bot: commands.Bot):
    bot.add_command(wav)
    bot.add_command(say)
