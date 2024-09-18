from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from commands import audio

pytestmark = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_wav_with_mocked_file(bot, mock_interaction):
    with patch(
        "utils.audio_utils.load_and_run_sao_model", new_callable=AsyncMock
    ) as mock_sao, patch(
        "utils.video_utils.convert_audio_to_waveform_video",
        return_value="fake_video.mp4",
    ) as mock_convert, patch(
        "discord.File", new_callable=Mock
    ) as mock_file:

        mock_sao.return_value = (b"fake_audio_data", 44100)
        mock_file.return_value = Mock(filename="fake_video.mp4")

        wav_callback = audio.wav.callback
        await wav_callback(mock_interaction, "test prompt")

        mock_interaction.response.defer.assert_called_once()
        mock_sao.assert_called_once_with(
            "test prompt", 10, 100, 7, 0.3, 500, "dpmpp-3m-sde"
        )
        mock_convert.assert_called_once_with(b"fake_audio_data", 44100, 10)
        mock_file.assert_called_once_with("fake_video.mp4")
        mock_interaction.followup.send.assert_called_once()

        call_args = mock_interaction.followup.send.call_args
        sent_content = call_args.kwargs.get("content")
        sent_files = call_args.kwargs.get("files")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "test prompt" in sent_content
        assert isinstance(sent_files, list), "Sent files should be a list"
        assert len(sent_files) == 1, "There should be one file sent"
        assert isinstance(sent_files[0], Mock), "Sent file should be a Mock object"
        assert (
            sent_files[0].filename == "fake_video.mp4"
        ), "Sent file should have the correct filename"


async def test_say(bot, mock_interaction):
    with patch(
        "services.chatgpt.generate_speech", new_callable=AsyncMock
    ) as mock_speech, patch(
        "utils.video_utils.convert_audio_to_waveform_video",
        return_value="fake_video.mp4",
    ) as mock_convert, patch(
        "discord.File", new_callable=Mock
    ) as mock_file:

        mock_speech.return_value = Path("fake_speech.mp3")
        mock_file.return_value = Mock(filename="fake_video.mp4")

        # Access the inner function (callback) of the command
        say_callback = audio.say.callback
        await say_callback(mock_interaction, "test text")

        mock_interaction.response.defer.assert_called_once()
        mock_speech.assert_called_once_with("test text", "onyx", 1.0)
        mock_convert.assert_called_once_with(
            Path("fake_speech.mp3"), Path("fake_speech.mp4")
        )
        mock_file.assert_called_once_with("fake_video.mp4")
        mock_interaction.followup.send.assert_called_once()

        # Check if the content is correct and contains the expected elements
        call_args = mock_interaction.followup.send.call_args
        if call_args.kwargs:
            sent_content = call_args.kwargs.get("content")
            sent_file = call_args.kwargs.get("file")
        elif call_args.args:
            sent_content = call_args.args[0]
            sent_file = call_args.args[1] if len(call_args.args) > 1 else None
        else:
            pytest.fail("followup.send was called without arguments")

        assert isinstance(sent_content, str), "Sent content should be a string"
        assert "onyx" in sent_content
        assert isinstance(sent_file, Mock), "Sent file should be a Mock object"
        assert (
            sent_file.filename == "fake_video.mp4"
        ), "Sent file should have the correct filename"
