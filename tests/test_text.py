from unittest.mock import AsyncMock, patch, ANY

import discord
import pytest

from commands import text

pytestmark = pytest.mark.asyncio


async def test_anthropic(bot, mock_interaction):
    with patch(
        "services.claude.anthropic_chat_completion", new_callable=AsyncMock
    ) as mock_anthropic:
        mock_anthropic.return_value = "Test response from Anthropic"

        # Access the inner function (callback) of the command
        anthropic_callback = text.anthropic.callback
        await anthropic_callback(
            mock_interaction, "test prompt", 1024, "claude-3-5-sonnet-20240620"
        )

        mock_interaction.response.defer.assert_called_once()
        mock_anthropic.assert_called_once_with(
            "test prompt", 1024, "claude-3-5-sonnet-20240620"
        )
        mock_interaction.followup.send.assert_called_once()

        # Check if the content is correct and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_content = call_args.kwargs.get("content")
        elif call_args.args:
            sent_content = call_args.args[0]
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "test prompt" in sent_content
        assert "Test response from Anthropic" in sent_content

        # Check if the content is formatted as expected
        expected_format = f"""### _TestUser_: \n\n```txt\ntest prompt\n```\n### anthropic:\n\n Test response from Anthropic"""
        assert (
            sent_content.strip() == expected_format.strip()
        ), "Response format is incorrect"

        # Verify that the mock_anthropic function was called with the correct arguments
        mock_anthropic.assert_called_once_with(
            "test prompt", 1024, "claude-3-5-sonnet-20240620"
        )


async def test_gpt(bot, mock_interaction):
    with patch(
        "services.chatgpt.gpt_chat_completion", new_callable=AsyncMock
    ) as mock_gpt:
        mock_response = ("Test response from GPT", 5)
        mock_gpt.return_value = mock_response

        # Access the inner function (callback) of the command
        gpt_callback = text.gpt.callback
        await gpt_callback(
            mock_interaction, "test prompt", seed=-1, model_name="gpt-4o-mini"
        )

        mock_interaction.response.defer.assert_called_once()

        # Use ANY for the first argument to ignore the specific content of the messages
        mock_gpt.assert_called_once_with(ANY, "gpt-4o-mini", -1)

        # Additional checks on the actual argument passed
        args, _ = mock_gpt.call_args
        messages = args[0]
        assert isinstance(messages, list)
        assert len(messages) == 3  # System message, user message, and assistant message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == [{"type": "text", "text": "test prompt"}]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == mock_response

        mock_interaction.followup.send.assert_called_once()

        # Check if the content is correct and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_embed = call_args.kwargs.get("embed")
        elif call_args.args:
            sent_embed = call_args.args[0]
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(
            sent_embed, discord.Embed
        ), "Sent content should be a discord.Embed"

        # Check embed fields
        assert sent_embed.description == str(
            mock_response
        ), "Embed description should be the string representation of the tuple"
        assert any(
            field.name == "Prompt" and field.value == "test prompt"
            for field in sent_embed.fields
        )
        assert any(field.name == "History Size" for field in sent_embed.fields)
        assert any(
            field.name == "Model" and field.value == "gpt-4o-mini"
            for field in sent_embed.fields
        )


async def test_refine(bot, mock_interaction):
    with patch("services.chatgpt.refine_prompt", new_callable=AsyncMock) as mock_refine:
        mock_refine.return_value = "Refined test prompt"

        # Access the inner function (callback) of the command
        refine_callback = text.refine.callback
        await refine_callback(mock_interaction, "test prompt")

        mock_interaction.response.defer.assert_called_once()
        mock_refine.assert_called_once_with("test prompt")
        mock_interaction.followup.send.assert_called_once()

        # Check if the content is correct and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_content = call_args.kwargs.get("content")
        elif call_args.args:
            sent_content = call_args.args[0]
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "You asked for: ```txt\ntest prompt```" in sent_content
        assert (
            "GPT4 upscaled the prompt to: ```txt\nRefined test prompt```"
            in sent_content
        )
        assert (
            "Copy and paste the upscaled prompt into `/imagine` to generate an image."
            in sent_content
        )
