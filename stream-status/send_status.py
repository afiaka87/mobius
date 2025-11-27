#!/usr/bin/env python3
"""
Stream Status Discord Notification Script
Polls Cloudflare Stream API and sends Discord notifications when stream goes live.
"""

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import discord
from discord.ext import tasks
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables (check parent directory too)
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Configuration from .env
DISCORD_TOKEN = os.getenv("DISCORD_API_TOKEN")
DISCORD_MENTION_USER_ID = os.getenv("DISCORD_MENTION_USER_ID")
CF_API_KEY = os.getenv("CLOUDFLARE_LIVE_INPUT_KEY")

# Configuration from cf-live-inputs-info.txt
LIVE_INPUT_ID = "e951bde02e1000f55e68106c049a7859"
CUSTOMER_CODE = "6qfdxdt6v9r0omsq"

# Test Discord server/channel
# SERVER_ID = 1249087687965671475 # Test server
# CHANNEL_ID = 1249087688519454762
SERVER_ID = 870344451770449990
CHANNEL_ID = 1375319100124827748

# Cloudflare API endpoint
CF_API_URL = f"https://customer-{CUSTOMER_CODE}.cloudflarestream.com/{LIVE_INPUT_ID}/lifecycle"

# Polling interval (seconds)
POLL_INTERVAL = 3

# Health check interval (log "still alive" every N polls)
HEALTH_CHECK_INTERVAL = 1200  # Every hour at 3s intervals

# Require N consecutive "live" readings before sending notification
CONFIRMATION_POLLS = 2


class StreamMonitor(discord.Client):
    """Monitors Cloudflare Stream status and sends Discord notifications."""

    def __init__(self, mention_user_id: int) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # Required for get_member()
        super().__init__(intents=intents)

        self.mention_user_id = mention_user_id
        self.is_live = False
        self.admin: discord.Member | None = None
        self.stream_channel: discord.TextChannel | None = None
        self.session: aiohttp.ClientSession | None = None
        self.poll_count = 0
        self.consecutive_failures = 0
        self.consecutive_live_count = 0  # For confirmation polling

    async def setup_hook(self) -> None:
        """Called when the client is starting up."""
        self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
        await super().close()

    async def on_ready(self) -> None:
        """Called when Discord connection is established."""
        logger.info(f"Discord bot connected as {self.user}")

        # Get the guild, channel, and admin member
        for guild in self.guilds:
            if guild.id == SERVER_ID:
                self.admin = guild.get_member(self.mention_user_id)
                channel = guild.get_channel(CHANNEL_ID)
                if isinstance(channel, discord.TextChannel):
                    self.stream_channel = channel
                break

        if not self.stream_channel:
            logger.error(f"Could not find channel {CHANNEL_ID}")
            await self.close()
            return

        if not self.admin:
            logger.error(f"Could not find admin user {self.mention_user_id}")
            await self.close()
            return

        logger.info(f"Monitoring channel: {self.stream_channel.name} in {self.stream_channel.guild.name}")
        logger.info(f"Admin user: {self.admin.display_name}")
        logger.info(f"Polling every {POLL_INTERVAL} seconds (confirmation: {CONFIRMATION_POLLS} polls)")

        # Check initial stream state before starting monitor loop
        initial_status = await self.check_stream_status()
        if initial_status is not None:
            self.is_live = initial_status
            status_str = "LIVE" if initial_status else "OFFLINE"
            logger.info(f"Initial stream state: {status_str}")
        else:
            logger.warning("Could not determine initial stream state, assuming OFFLINE")

        # Start the monitoring task if not already running
        if not self.monitor_task.is_running():
            self.monitor_task.start()

    async def on_disconnect(self) -> None:
        """Called when Discord connection is lost."""
        logger.warning("Discord connection lost")

    async def on_resumed(self) -> None:
        """Called when Discord connection is resumed after disconnect."""
        logger.info("Discord connection resumed")
        # Refresh admin reference in case member cache changed
        for guild in self.guilds:
            if guild.id == SERVER_ID:
                self.admin = guild.get_member(self.mention_user_id)
                break

    async def check_stream_status(self, retry: bool = True) -> bool | None:
        """Poll Cloudflare API to check if stream is live (async, non-blocking).

        Args:
            retry: If True, retry once on failure before giving up.
        """
        if not self.session:
            logger.error("HTTP session not initialized")
            return None

        try:
            headers = {"Authorization": f"Bearer {CF_API_KEY}"}
            timeout = aiohttp.ClientTimeout(total=10)
            async with self.session.get(CF_API_URL, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                data: dict[str, bool] = await response.json()
                self.consecutive_failures = 0  # Reset on success
                return data.get("live", False)

        except aiohttp.ClientError as e:
            # Extract HTTP status if available
            status_info = ""
            if hasattr(e, "status"):
                status_info = f" (HTTP {e.status})"

            # Retry once before counting as failure
            if retry:
                logger.warning(f"API request failed{status_info}, retrying...")
                await asyncio.sleep(0.5)
                return await self.check_stream_status(retry=False)

            self.consecutive_failures += 1
            logger.exception(f"Error checking stream status{status_info} (failure #{self.consecutive_failures})")
            if self.consecutive_failures >= 10:
                logger.warning("10 consecutive API failures - stream status unknown!")
            return None

    async def send_live_notification(self) -> None:
        """Send Discord notification that stream is live."""
        if self.admin is None or self.stream_channel is None:
            logger.error("Admin or stream_channel is not set; cannot send notification")
            return

        try:
            message = f"**{self.admin.display_name}** started streaming at [delicious-donuts.com](<https://delicious-donuts.com/>)"
            await self.stream_channel.send(message)
            logger.info("Sent live notification to Discord")
        except discord.DiscordException:
            logger.exception("Error sending Discord message")

    @tasks.loop(seconds=POLL_INTERVAL)
    async def monitor_task(self) -> None:
        """Main monitoring loop using discord.ext.tasks for proper reconnection handling."""
        self.poll_count += 1

        # Periodic health check
        if self.poll_count % HEALTH_CHECK_INTERVAL == 0:
            failures = self.consecutive_failures
            logger.info(f"Health check: {self.poll_count} polls completed, consecutive failures: {failures}")

        current_status = await self.check_stream_status()

        if current_status is None:
            # API failure - reset confirmation counter
            self.consecutive_live_count = 0
            return

        if current_status and not self.is_live:
            # Stream appears to be live - increment confirmation counter
            self.consecutive_live_count += 1

            if self.consecutive_live_count < CONFIRMATION_POLLS:
                # Still waiting for confirmation
                logger.info(f"Stream live detected ({self.consecutive_live_count}/{CONFIRMATION_POLLS} confirmations)")
            else:
                # Confirmed live! Send notification
                logger.info(f"Stream confirmed LIVE after {CONFIRMATION_POLLS} polls! Sending notification...")
                await self.send_live_notification()
                self.is_live = True
                self.consecutive_live_count = 0

        elif not current_status and self.is_live:
            # Stream went offline
            logger.info("Stream went offline (no notification sent)")
            self.is_live = False
            self.consecutive_live_count = 0

        elif not current_status:
            # Still offline - reset any partial confirmation
            self.consecutive_live_count = 0

        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            status_str = "LIVE" if current_status else "OFFLINE"
            logger.debug(f"Stream status: {status_str}")

    @monitor_task.before_loop
    async def before_monitor(self) -> None:
        """Wait for Discord to be ready before starting the monitor loop."""
        await self.wait_until_ready()

    @monitor_task.after_loop
    async def after_monitor(self) -> None:
        """Called when the monitor loop stops."""
        if self.monitor_task.is_being_cancelled():
            logger.info("Monitor task cancelled")
        else:
            logger.warning("Monitor task stopped unexpectedly")


async def main() -> None:
    """Main entry point."""
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN is not set")
        return
    if not DISCORD_MENTION_USER_ID:
        logger.error("DISCORD_MENTION_USER_ID is not set")
        return

    mention_user_id = int(DISCORD_MENTION_USER_ID)

    logger.info("=" * 50)
    logger.info("Starting Stream Status Monitor")
    logger.info("=" * 50)
    logger.info(f"Stream ID: {LIVE_INPUT_ID}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Confirmation polls required: {CONFIRMATION_POLLS}")
    health_minutes = HEALTH_CHECK_INTERVAL * POLL_INTERVAL // 60
    logger.info(f"Health check interval: {HEALTH_CHECK_INTERVAL} polls (~{health_minutes} min)")
    logger.info("=" * 50)

    monitor = StreamMonitor(mention_user_id)
    try:
        await monitor.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        if not monitor.is_closed():
            await monitor.close()


def cli() -> None:
    """CLI entry point for uv/pip installations."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")


if __name__ == "__main__":
    cli()
