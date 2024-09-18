import base64
import io
import os
import tempfile
from unittest.mock import AsyncMock, patch

import discord
import pytest
from PIL import Image

from commands import image_generation

pytestmark = pytest.mark.asyncio


async def test_flux(bot, mock_interaction):
    with patch(
        "services.fal_ai.generate_flux_image", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "http://example.com/image.png"

        # Access the inner function (callback) of the command
        flux_callback = image_generation.flux.callback
        await flux_callback(mock_interaction, "test prompt")

        mock_interaction.response.defer.assert_called_once()
        mock_generate.assert_called_once_with(
            "test prompt", "fal-ai/flux/schnell", "landscape_16_9", 3.5
        )
        mock_interaction.followup.send.assert_called_once()

        # Check if the content is a string and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.args:
            sent_content = call_args.args[0]
        elif "content" in call_args.kwargs:
            sent_content = call_args.kwargs["content"]
        else:
            pytest.fail("followup.send was called without content")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "test prompt" in sent_content
        assert "http://example.com/image.png" in sent_content
        assert "fal-ai/flux/schnell" in sent_content
        assert "landscape_16_9" in sent_content


async def test_flux_img(bot, mock_interaction):
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
        image = Image.new("RGB", (100, 100), color="red")
        image.save(temp_image, format="PNG")
        temp_image_path = temp_image.name

    with patch(
        "services.comfy_ui.flux_img_to_img", new_callable=AsyncMock
    ) as mock_flux_img:
        # Create a bytes object that represents a valid image
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100), color="blue").save(buffer, format="PNG")
        mock_flux_img.return_value = [buffer.getvalue()]

        # Access the inner function (callback) of the command
        flux_img_callback = image_generation.flux_img.callback
        await flux_img_callback(mock_interaction, "test clip text", temp_image_path)

        mock_interaction.response.defer.assert_called_once()
        mock_flux_img.assert_called_once_with(
            temp_image_path,
            "test clip text",
            "flux1-dev-fp8.safetensors",
            1.0,
            20,
            0.75,
        )
        mock_interaction.followup.send.assert_called_once()

        # Check if the content is correct and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_content = call_args.kwargs.get("content")
            sent_files = call_args.kwargs.get("files")
        elif call_args.args:
            sent_content = call_args.args[0] if len(call_args.args) > 0 else None
            sent_files = call_args.args[1] if len(call_args.args) > 1 else None
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "test clip text" in sent_content
        assert temp_image_path in sent_content

        assert sent_files is not None, "Files should be sent with the message"
        assert len(sent_files) == 1, "One file should be sent"
        assert isinstance(
            sent_files[0], discord.File
        ), "Sent file should be a discord.File object"

    # Clean up the temporary file
    os.unlink(temp_image_path)


async def test_unload_comfy(bot, mock_interaction):
    with patch(
        "services.comfy_ui.unload_comfy_via_api", new_callable=AsyncMock
    ) as mock_unload:
        mock_unload.return_value = "Unloaded successfully"

        # Access the inner function (callback) of the command
        unload_comfy_callback = image_generation.unload_comfy.callback
        await unload_comfy_callback(mock_interaction)

        mock_interaction.response.defer.assert_called_once()
        mock_unload.assert_called_once()
        mock_interaction.followup.send.assert_called_once_with(
            "Response: Unloaded successfully"
        )


async def test_dalle(bot, mock_interaction):
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
        image = Image.new("RGB", (100, 100), color="green")
        image.save(temp_image, format="PNG")
        temp_image.seek(0)
        image_bytes = temp_image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        temp_image_path = temp_image.name

    with patch("services.chatgpt.dalle_callback", new_callable=AsyncMock) as mock_dalle:
        mock_dalle.return_value = ([base64_image], "Refined prompt")

        # Access the inner function (callback) of the command
        dalle_callback = image_generation.dalle.callback
        await dalle_callback(mock_interaction, "test prompt")

        mock_interaction.response.defer.assert_called_once()
        mock_dalle.assert_called_once_with("test prompt", "standard", "1024x1024")
        mock_interaction.followup.send.assert_called_once()

        # Check if the content is correct and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_content = call_args.kwargs.get("content")
            sent_files = call_args.kwargs.get("files")
        elif call_args.args:
            sent_content = call_args.args[0] if len(call_args.args) > 0 else None
            sent_files = call_args.args[1] if len(call_args.args) > 1 else None
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "test prompt" in sent_content
        assert "Refined prompt" in sent_content

        assert sent_files is not None, "Files should be sent with the message"
        assert len(sent_files) == 1, "One file should be sent"
        assert isinstance(
            sent_files[0], discord.File
        ), "Sent file should be a discord.File object"

    # Clean up the temporary file
    os.unlink(temp_image_path)
