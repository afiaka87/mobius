import inspect
import os
from typing import Coroutine, Any
import logging

import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv

from commands import image_generation, audio, text, utility, video

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize bot
bot = commands.Bot(command_prefix="!", intents=intents)

# Define guild IDs from hard-coded environment variables
GUILD_IDS = [
    discord.Object(id=os.environ["DISCORD_GUILD_ID_FUNZONE"]),
    discord.Object(id=os.environ["DISCORD_GUILD_ID_BOTTEST"]),
]

# Register commands
def register_commands(bot) -> None:
    """
    Register all app commands from the command modules.
    """
    command_modules = [image_generation, audio, text, utility, video]
    
    for module in command_modules:
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, app_commands.Command):
                print(f"Adding app command: {obj.name}")
                bot.tree.add_command(obj)

    print(f"Registered {len(bot.tree.get_commands())} app commands.")



def sync_commands_to_guilds(bot: commands.Bot) -> Coroutine[Any, Any, None]:
    """
    Sync commands to the specified guilds.
    """
    async def sync():
        for guild in GUILD_IDS:
            logging.info(f"Syncing commands to guild: {guild.id}")
            bot.tree.copy_global_to(guild=guild)
            try:
                await bot.tree.sync(guild=guild)
                logging.info(f"Successfully synced commands to guild: {guild.id}")
            except Exception as e:
                logging.error(f"Failed to sync commands to guild {guild.id}: {str(e)}")
    return sync()

@bot.event
async def on_ready():
    """
    Log in as the bot and register commands.
    """
    logging.info(f"Logged in as {bot.user}!")
    register_commands(bot)
    await sync_commands_to_guilds(bot)
    
    logging.info("Slash commands synced! Ready to go!")

def run_bot():
    """
    Run the bot.
    """
    bot.run(os.environ["DISCORD_API_TOKEN"])

if __name__ == "__main__":
    run_bot()