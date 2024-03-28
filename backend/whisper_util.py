## whisper_util.py
#  This program will accept a URL to a video or audio file,
#  download the video as an audio file (if needed) with yt-dlp,
#  and then transcribe the audio file with faster-whisper.

import os
import asyncio


MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"




async def dl_and_transcribe(url: str) -> str:
    # download the video as an audio file
    audio_file = await download_youtube_video_as_audio(url)

    # transcribe the audio file
    transcript_filename = await transcribe(audio_file)

    # remove the audio file
    os.remove(audio_file)

    # return the transcript filename
    return transcript_filename


# test
if __name__ == "__main__":
    # test
    url = "https://www.youtube.com/watch?v=0RyInjfgNc4"
    asyncio.run(dl_and_transcribe(url))
