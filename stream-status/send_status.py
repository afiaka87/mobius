#!/usr/bin/env python3
"""
Stream Status Discord Notification Script
Polls Cloudflare Stream API and sends Discord notifications when stream goes live.
"""

import os
import asyncio
import logging
from datetime import datetime
import discord
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration from .env
DISCORD_TOKEN = os.getenv('DISCORD_API_TOKEN')
DISCORD_MENTION_USER_ID = os.getenv('DISCORD_MENTION_USER_ID')
CF_API_KEY = os.getenv('CLOUDFLARE_LIVE_INPUT_KEY')

# Configuration from cf-live-inputs-info.txt
LIVE_INPUT_ID = "e951bde02e1000f55e68106c049a7859"
CUSTOMER_CODE = "6qfdxdt6v9r0omsq"

# Test Discord server/channel
SERVER_ID = 1249087687965671475
CHANNEL_ID = 1249087688519454762

# Cloudflare API endpoint
CF_API_URL = f"https://customer-{CUSTOMER_CODE}.cloudflarestream.com/{LIVE_INPUT_ID}/lifecycle"

# Polling interval (seconds)
POLL_INTERVAL = 5


class StreamMonitor:
    """Monitors Cloudflare Stream status and sends Discord notifications."""

    def __init__(self):
        self.is_live = False  # Track current stream status
        self.client = None
        self.admin = None  # Discord member to reference
        self.stream_channel = None  # Channel to send notifications to

    def check_stream_status(self):
        """Poll Cloudflare API to check if stream is live."""
        try:
            headers = {
                'Authorization': f'Bearer {CF_API_KEY}'
            }
            response = requests.get(CF_API_URL, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('live', False)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking stream status: {e}")
            return None

    async def send_live_notification(self):
        """Send Discord notification that stream is live."""
        try:
            message = f"**{self.admin.display_name}** started streaming at [delicious-donuts.com](<https://delicious-donuts.com/>)"
            await self.stream_channel.send(message)
            logger.info("Sent live notification to Discord")
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")

    async def monitor_loop(self):
        """Main monitoring loop."""
        # Set up Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # Required for get_member()
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            logger.info(f'Discord bot connected as {self.client.user}')

            # Get the guild, channel, and admin member
            for guild in self.client.guilds:
                if guild.id == SERVER_ID:
                    self.admin = guild.get_member(int(DISCORD_MENTION_USER_ID))
                    self.stream_channel = guild.get_channel(CHANNEL_ID)
                    break

            if not self.stream_channel:
                logger.error(f"Could not find channel {CHANNEL_ID}")
                await self.client.close()
                return

            if not self.admin:
                logger.error(f"Could not find admin user {DISCORD_MENTION_USER_ID}")
                await self.client.close()
                return

            logger.info(f"Monitoring channel: {self.stream_channel.name} in {self.stream_channel.guild.name}")
            logger.info(f"Admin user: {self.admin.display_name}")
            logger.info(f"Polling every {POLL_INTERVAL} seconds")

            # Start monitoring loop
            while True:
                try:
                    current_status = self.check_stream_status()

                    if current_status is not None:
                        # Detect offline -> live transition
                        if current_status and not self.is_live:
                            logger.info("Stream went LIVE! Sending notification...")
                            await self.send_live_notification()
                            self.is_live = True

                        # Detect live -> offline transition (silent)
                        elif not current_status and self.is_live:
                            logger.info("Stream went offline (no notification sent)")
                            self.is_live = False

                        # Log current status
                        status_str = "LIVE" if current_status else "OFFLINE"
                        logger.debug(f"Stream status: {status_str}")

                    await asyncio.sleep(POLL_INTERVAL)

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(POLL_INTERVAL)

        # Start the Discord client
        try:
            await self.client.start(DISCORD_TOKEN)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.client.close()


async def main():
    """Main entry point."""
    logger.info("Starting Stream Status Monitor")
    logger.info(f"Monitoring stream: {LIVE_INPUT_ID}")

    monitor = StreamMonitor()
    try:
        await monitor.monitor_loop()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        if monitor.client:
            await monitor.client.close()


def cli():
    """CLI entry point for uv/pip installations."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")


if __name__ == "__main__":
    cli()
