import os

import discord
from discord import app_commands

from services import google, weather, youtube


@app_commands.command(name="youtube", description="Search youtube. Returns top result.")
async def youtube_search(interaction: discord.Interaction, query: str):
    """
    Search youtube. Returns top result.
    """
    await interaction.response.defer(ephemeral=False, thinking=True)
    result = await youtube.get_top_youtube_result_httpx(
        query, api_key=os.getenv("YOUTUBE_API_KEY")
    )
    if "error" in result:
        await interaction.followup.send(result["error"])
    else:
        url = f"https://www.youtube.com/watch?v={result['videoId']}"
        await interaction.followup.send(url)


@app_commands.command(name="temp", description="Get the temperature.")
async def temp(interaction: discord.Interaction):
    """
    Get the temperature.
    """
    await interaction.response.defer(ephemeral=False, thinking=True)
    friendly_current_temperature = await weather.temp_callback()
    await interaction.followup.send(friendly_current_temperature)


@app_commands.command(
    name="google",
    description="Uses the google custom search api to get results from the web.",
)
async def google_search(interaction: discord.Interaction, query: str):
    """
    Uses the google custom search api to get results from the web.
    """
    await interaction.response.defer(ephemeral=False, thinking=True)
    results = await google.google_search(query)
    await interaction.followup.send(results)
