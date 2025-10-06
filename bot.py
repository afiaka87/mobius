#!/usr/bin/env python3
# bot.py

"""
Main entry point for the Mobius Discord Bot.

This script initializes the bot, loads commands, and connects to Discord.
"""

import inspect
import logging
import os
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# Renamed to avoid collision with discord.ext.commands
import commands as my_bot_commands

# Load environment variables from .env file
# It's good practice to call this early, before other modules might need env vars.
load_dotenv()

# Set up logging
# Basic configuration is set here.
# For more complex setups, consider a dedicated logging config.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# Set up intents
# These define what events the bot will receive from Discord.
intents: discord.Intents = discord.Intents.default()
intents.message_content = True  # Required for commands that read message content

# Initialize bot
# The command_prefix is for legacy message-based commands,
# not strictly needed for slash commands.
bot: commands.Bot = commands.Bot(command_prefix="!", intents=intents)


def register_all_commands(client: commands.Bot) -> None:
    """
    Register all app commands from the command modules with the bot.

    Dynamically inspects modules listed in `command_modules` for instances
    of `app_commands.Command` or `app_commands.ContextMenu` and adds them
    to the bot's command tree.

    Args:
        client: The discord.ext.commands.Bot instance to register commands with.

    Raises:
        RuntimeError: If no application commands are found and registered.
    """
    command_modules: list[Any] = [
        my_bot_commands
    ]  # List of modules containing commands
    registered_command_count: int = 0

    for module in command_modules:
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, app_commands.Command | app_commands.ContextMenu):
                logger.info(
                    f"Adding app command: {obj.name} from module {module.__name__}"
                )
                client.tree.add_command(obj)
                registered_command_count += 1
            # Optional: Log skipped members if needed for debugging
            # else:
            #     logger.debug(f"Skipping {name} in {module.__name__} "
    #                  f"(not a command object).")

    if registered_command_count == 0:
        logger.error("No application commands were found or registered.")

        # Depending on desired behavior, this could raise an error or just log a warning
        # For now, raising an error as it likely indicates a setup problem.
        # Create a custom exception class for better error handling
        class CommandRegistrationError(RuntimeError):
            """Raised when no commands could be registered."""

        raise CommandRegistrationError("No application commands registered")

    logger.info(
        f"Successfully registered {registered_command_count} application commands."
    )


async def sync_commands_to_guilds(client: commands.Bot) -> None:
    """
    Sync application commands to specified Discord guilds.

    Reads guild IDs from the `DISCORD_GUILD_IDS` environment variable
    (comma-separated), copies global commands to these guilds, and syncs them.

    Args:
        client: The discord.ext.commands.Bot instance.

    Raises:
        ValueError: If `DISCORD_GUILD_IDS` environment variable is not set or is empty.
        Exception: Propagates exceptions from `bot.tree.sync` if syncing fails.
    """
    guild_ids_str: str | None = os.getenv("DISCORD_GUILD_IDS")
    if not guild_ids_str:
        logger.error(
            "DISCORD_GUILD_IDS environment variable not set or empty. "
            "Cannot sync commands to specific guilds."
        )

        # Depending on requirements, you might want to fall back to global sync
        # or raise an error. For now, raising an error.
        # Define a custom exception for configuration errors
        class ConfigError(ValueError):
            """Raised when required configuration is missing."""

        raise ConfigError("DISCORD_GUILD_IDS environment variable is required")

    guild_id_list: list[str] = [gid.strip() for gid in guild_ids_str.split(",")]
    discord_guild_objects: list[discord.Object] = []

    for gid_str in guild_id_list:
        if gid_str.isdigit():
            discord_guild_objects.append(discord.Object(id=int(gid_str)))
        else:
            logger.warning(f"Invalid guild ID format: '{gid_str}'. Skipping.")

    if not discord_guild_objects:
        logger.error("No valid guild IDs found to sync commands to.")
        return  # Or raise an error if at least one guild is mandatory

    for guild in discord_guild_objects:
        logger.info(f"Syncing commands to guild: {guild.id}")
        # This copies global commands to the guild.
        # If you have guild-specific commands, they should be added with @app_commands.guilds()
        client.tree.copy_global_to(guild=guild)
        try:
            await client.tree.sync(guild=guild)
            logger.info(f"Successfully synced commands to guild: {guild.id}")
        except discord.HTTPException as e:
            logger.exception(f"Failed to sync commands to guild {guild.id}: {e}")
            # Optionally re-raise or handle specific HTTP error codes
            raise
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred while syncing to guild {guild.id}: {e}"
            )
            raise


# Add command error handler for debugging
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    """Handle application command errors."""
    logger.error(f"Command error: {error} for command: {interaction.command.name if interaction.command else 'Unknown'}")
    if interaction.response.is_done():
        await interaction.followup.send(f"An error occurred: {error}", ephemeral=True)
    else:
        await interaction.response.send_message(f"An error occurred: {error}", ephemeral=True)

@bot.event
async def on_ready() -> None:
    """
    Called when the bot is ready and connected to Discord.

    This function registers and syncs slash commands and sets the bot's presence.
    """
    if bot.user:
        logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    else:
        logger.info(
            "Logged in, but bot.user is not yet available."
        )  # Should not happen if event is correct

    logger.info("Registering commands...")
    try:
        register_all_commands(bot)
        logger.info("Commands registered. Syncing globally first...")
        # Try global sync first
        await bot.tree.sync()
        logger.info("Global sync complete. Now syncing to guilds...")
        await sync_commands_to_guilds(bot)
        logger.info("Slash commands synced! Bot is ready.")
    except RuntimeError as e:
        logger.critical(f"Failed to initialize commands: {e}")
        # Consider shutting down the bot or preventing further operation if commands are critical
        return
    except ValueError as e:
        logger.critical(f"Configuration error for command syncing: {e}")
        return
    except Exception as e:
        logger.critical(f"An unexpected error during on_ready setup: {e}")
        return

    # Set bot presence (e.g., "Playing a game", "Listening to...", or invisible)
    try:
        await bot.change_presence(status=discord.Status.invisible)
        logger.info("Bot presence set to invisible.")
    except Exception as e:
        logger.exception(f"Failed to set bot presence: {e}")


def main() -> None:
    """
    Main function to run the Discord bot.

    Retrieves the Discord API token from environment variables and starts the bot.
    """
    discord_api_token: str | None = os.getenv("DISCORD_API_TOKEN")
    if not discord_api_token:
        logger.critical(
            "DISCORD_API_TOKEN environment variable not set. Bot cannot start."
        )
        return

    logger.info("Starting bot...")
    try:
        bot.run(discord_api_token)
    except discord.LoginFailure:
        logger.critical("Failed to log in: Invalid Discord API token.")
    except discord.HTTPException as e:
        logger.critical(f"Discord HTTP error during connection/startup: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred while running the bot: {e}")


if __name__ == "__main__":
    main()
