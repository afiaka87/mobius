from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands


class MockBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=discord.Intents.all())


@pytest.fixture
async def bot():
    return MockBot()


class MockInteraction:
    def __init__(self):
        self.response = AsyncMock()
        self.followup = AsyncMock()
        self.user = MagicMock()
        self.user.name = "TestUser"


@pytest.fixture
async def mock_interaction():
    return MockInteraction()
