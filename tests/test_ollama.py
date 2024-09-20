import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import discord
from commands import ollama_command


@pytest.mark.asyncio
async def test_ollama():
    mock_interaction = MagicMock(spec=discord.Interaction)
    mock_interaction.response.defer = AsyncMock()
    mock_message = AsyncMock()
    mock_interaction.followup.send = AsyncMock(return_value=mock_message)

    mock_message.edit = AsyncMock()

    ollama_callback = ollama_command.ollama_command

    with patch(
        "services.ollama_service.encode_image", return_value="base64_encoded_image"
    ):

        await ollama_callback.callback(mock_interaction, prompt="Hello world!")

        mock_interaction.response.defer.assert_called_once()
        # mock_interaction.followup.send.assert_called_once_with("Generating response...")
        mock_interaction.followup.send.assert_called_once_with(
            "Generating response...",
            embed=pytest.approx(
                ollama_command.build_response_embed(
                    prompt="Hello world!",
                    model="llama3.1:8b-instruct-q4_K_M",
                    system=None,
                    image=None,
                    temperature=0.7,
                    top_k=100,
                    top_p=0.9,
                    num_predict=512,
                )
            ),
        )

        # Check if edit was called with the full response
        edited_content = mock_message.edit.call_args_list[-1][1]["content"]
        # Should start with "Hello world!"
        assert edited_content.startswith(
            "## Ollama Response"
        )  # TODO verify actual output
