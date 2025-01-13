from dotenv import load_dotenv  # python-dotenv

load_dotenv()  # TODO: if env errors happen, check if this is the right place to load the env

import inspect
import logging
import os
from typing import Any, Coroutine

import commands as my_commands  # avoid collision with discord.ext.commands

import discord
from discord import app_commands
from discord.ext import commands

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize bot
bot = commands.Bot(command_prefix="!", intents=intents)


# Register commands
def register_commands(bot) -> None:
    """
    Register all app commands from the command modules.
    """
    command_modules = [my_commands]

    for module in command_modules:
        for _, obj in inspect.getmembers(module):
            if isinstance(obj, app_commands.Command) or isinstance(
                obj, app_commands.ContextMenu
            ):
                print(f"Adding app command: {obj.name}")
                bot.tree.add_command(obj)
            else:
                print(f"Skipping {obj}")

    # If none of the commands were registered, raise an error. Something is wrong.
    if len(bot.tree.get_commands()) == 0:
        logging.error("No app commands registered.")
        raise Exception("No app commands registered.")

    print(f"Registered {len(bot.tree.get_commands())} app commands.")


def sync_commands_to_guilds(bot: commands.Bot) -> Coroutine[Any, Any, None]:
    """
    Sync commands to the specified guilds.
    """
    # Define guild IDs from hard-coded environment variables
    GUILD_IDS = [
        discord.Object(id=os.environ["DISCORD_GUILD_ID_FUNZONE"]),
        discord.Object(id=os.environ["DISCORD_GUILD_ID_BOTTEST"]),
    ]  # TODO: should be a single env variable with comma-separated guild IDs

    async def sync():
        for guild in GUILD_IDS:
            logging.info(f"Syncing commands to guild: {guild.id}")
            bot.tree.copy_global_to(guild=guild)
            try:
                await bot.tree.sync(guild=guild)
                logging.info(f"Successfully synced commands to guild: {guild.id}")
            except Exception as e:
                logging.error(f"Failed to sync commands to guild {guild.id}: {str(e)}")
                raise e

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


if __name__ == "__main__":
    # Load environment variables
    bot.run(os.environ["DISCORD_API_TOKEN"])
