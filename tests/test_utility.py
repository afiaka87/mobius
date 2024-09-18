from unittest.mock import AsyncMock, patch

import pytest

from commands import utility

pytestmark = pytest.mark.asyncio


async def test_youtube_search(bot, mock_interaction):
    with patch(
        "services.youtube.get_top_youtube_result_httpx", new_callable=AsyncMock
    ) as mock_youtube:
        mock_youtube.return_value = {"videoId": "dQw4w9WgXcQ"}

        # Access the inner function (callback) of the command
        youtube_search_callback = utility.youtube_search.callback
        await youtube_search_callback(mock_interaction, "test query")

        mock_interaction.response.defer.assert_called_once()
        mock_youtube.assert_called_once_with(
            "test query", api_key="AIzaSyBpXYnpSiabv0VqwVKysnZuZNri6NE8ERE"
        )
        mock_interaction.followup.send.assert_called_once()

        call_args = mock_interaction.followup.send.call_args
        assert call_args.args[0] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


async def test_temp(bot, mock_interaction):
    with patch("services.weather.temp_callback", new_callable=AsyncMock) as mock_temp:
        mock_temp.return_value = "The current temperature is 25°C"

        # Access the inner function (callback) of the command
        temp_callback = utility.temp.callback
        await temp_callback(mock_interaction)

        mock_interaction.response.defer.assert_called_once()
        mock_temp.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        call_args = mock_interaction.followup.send.call_args
        assert call_args.args[0] == "The current temperature is 25°C"


async def test_google_search(bot, mock_interaction):
    with patch("services.google.google_search", new_callable=AsyncMock) as mock_google:
        mock_google.return_value = ["http://example.com", "http://example.org"]

        # Access the inner function (callback) of the command
        google_search_callback = utility.google_search.callback
        await google_search_callback(mock_interaction, "test query")

        mock_interaction.response.defer.assert_called_once()
        mock_google.assert_called_once_with("test query")
        mock_interaction.followup.send.assert_called_once()

        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_content = call_args.kwargs.get("content")
        elif call_args.args:
            sent_content = call_args.args[0]
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(sent_content, list), "Sent content should be a list of URLs"
        assert len(sent_content) == 2, "Two URLs should be returned"
        assert (
            "http://example.com" in sent_content
        ), "First URL should be in the results"
        assert (
            "http://example.org" in sent_content
        ), "Second URL should be in the results"

        # Optionally, check for suppress_embeds if your implementation uses it
        if call_args.kwargs:
            assert call_args.kwargs.get(
                "suppress_embeds", False
            ), "Embeds should be suppressed"
