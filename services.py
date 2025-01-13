### Begin OpenAI ###
import os

import fal_client

from typing import Union
from pathlib import Path
import base64
import mimetypes


from pathlib import Path
from typing import Optional

import httpx
import openai
from openai import OpenAI
from anthropic import AsyncAnthropic

import subprocess

from utils import convert_audio_to_waveform_video


async def gpt_chat_completion(
    messages: list,
    model_name: str,
    seed: Optional[int] = None,
) -> str:

    # api args dict
    api_args = {
        "model": model_name,
        "messages": messages,
        # "stream": True, # TODO We are disabling streaming to output the final message and audio at once.
    }
    if seed is not None:
        api_args["seed"] = seed

    # We are no longer streaming output and dont need to edit the message as streaming comes in. We can just send the final message.
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    completion = openai_client.chat.completions.create(**api_args)
    # TODO this f-string will go stale
    print(
        f"Created chat completion with api args: model={model_name}, messages={len(messages)}"
    )

    # Get the assistant response message
    return completion.choices[0].message.content


async def llm_callback(
    messages: list,
    model_name: str,
    model_api_url: str,
    temperature: Optional[float] = None,
    top_p: float = 1,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    # TODO remove or use this

    OPENAI_HEADERS = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }

    data_dict = {
        "model": model_name,
        "messages": messages,
        "top_p": top_p,
    }
    if temperature is not None:
        data_dict["temperature"] = temperature
    if seed is not None:
        data_dict["seed"] = seed
    if max_tokens is not None:
        data_dict["max_tokens"] = max_tokens
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=5, pool=5)
    ) as httpx_client:
        response = await httpx_client.post(
            model_api_url,  # either openai or lm-studio server url
            headers=OPENAI_HEADERS,
            json=data_dict,
        )
        response.raise_for_status()
        print(f"{response.status_code} status code from {model_api_url}")
        return response.json()


async def generate_speech(text: str, voice: str, speed: float) -> str:
    # Save to local .cache folder instead of tempfile

    speech_file_path = Path(".cache") / f"{voice}_{speed}_{text[:100]}.mp3"
    speech_file_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # create the .cache folder if it doesn't exist

    response = openai.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
        speed=float(speed),
        response_format="mp3",  # tts-1 or tts-1-hd
    )
    response.stream_to_file(speech_file_path)

    # Convert speech audio to video
    video_file_path = speech_file_path.with_suffix(".mp4")
    convert_audio_to_waveform_video(  # TODO make this more pure functional
        audio_file=str(speech_file_path), video_file=str(video_file_path)
    )
    return str(video_file_path)


### End OpenAI ###

### Begin Anthropic ###


async def anthropic_chat_completion(
    prompt: str,
    max_tokens: int,
    model: str = "claude-3-5-sonnet-20240620",  # TODO update
) -> str:
    anthropic_client = (
        AsyncAnthropic()
    )  # Automatically detects API key in env var ANTHROPIC_API_KEY

    create_message_args_dict = {
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"},
    }

    message = await anthropic_client.messages.create(**create_message_args_dict)
    message_text = message.content[0].text  # Message -> list[TextBlock] -> str

    return message_text


### End Anthropic ###


### Begin fal.ai ###
def image_to_base64_url(image_path: Path) -> str:
    # Ensure the file exists
    if not image_path.exists():
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    # Guess the MIME type based on the file extension
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file {image_path}")

    # Read the image file as binary data
    with image_path.open("rb") as img_file:
        image_data = img_file.read()

    # Encode the image data to base64
    base64_encoded = base64.b64encode(image_data).decode("utf-8")

    # Create the base64 URL with appropriate prefix
    base64_url = f"data:{mime_type};base64,{base64_encoded}"

    return base64_url


async def generate_flux_image(
    prompt: str,
    model: str,
    image_size: str,
    guidance_scale: float,
) -> str:

    handler = fal_client.submit(
        model,
        arguments={
            "prompt": prompt,
            "image_size": image_size,
            "guidance_scale": guidance_scale,
        },
    )

    result = handler.get()
    return result["images"][0]["url"]


# you can pass your own URL or a Base64 data URI.
async def flux_img_to_img(
    prompt: str,
    image_url: Union[str, Path],
    strength: float = 0.80,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
):
    if isinstance(image_url, Path):
        # convert to bytes
        image_url = image_to_base64_url(image_url)
    elif not image_url.startswith("http"):
        raise ValueError(
            f"Invalid image URL: {image_url}. Must be a URL str or a Path."
        )

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    result = await fal_client.subscribe_async(
        "fal-ai/flux/dev/image-to-image",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "enable_safety_checker": False,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    print(result)
    return result["images"][0]["url"]


### End fal.ai ###

### Begin Google ###


async def google_search(query: str) -> str:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
        )
        results = response.json()
        results = "\n".join(
            [result["link"] for result in results["items"][:3]]
        )  # just want the first three, only the links
        return results


### End Google ###


### Begin weather ###
async def temp_callback() -> str:
    import python_weather

    async with python_weather.Client(unit=python_weather.IMPERIAL) as weather_client:  # type: ignore
        current_weather = await weather_client.get("Fayetteville, AR")
        temperature = current_weather.temperature
        return f"The current temperature in Fayetteville, AR is {temperature}°F."


### End weather ###

### Begin youtube ###


async def get_top_youtube_result_httpx(search_query, api_key):
    """
    Calls the YouTube Search API and fetches the top search result for a given query.

    Parameters:
    search_query (str): The search query string.
    api_key (str): Your YouTube Data API key.

    Returns:
    dict: Information about the top search result, or an error message if the call fails.
    """

    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "maxResults": 1,
        "key": api_key,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            if not items:
                return {"error": "No results found"}
            top_result = items[0]
            return {
                "videoId": top_result["id"]["videoId"],
                "title": top_result["snippet"]["title"],
                "description": top_result["snippet"]["description"],
                "channelTitle": top_result["snippet"]["channelTitle"],
            }
        else:
            return {
                "error": "Failed to fetch results, status code: {}".format(
                    response.status_code
                )
            }


async def download_youtube_video_as_audio(video_url: str):
    """Download a youtube video as an audio file.

    Args:
        video_url (str): The URL of the video to download. Must be a valid youtube URL. Example: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    Raises:
        Exception: A temporary file is created and subsequently uploaded. If the file is not created, an exception is raised.

    Returns:
        Path: The path to the downloaded audio file.
    """

    # scratch_filename = tempfile.NamedTemporaryFile(suffix=".opus", delete=False).name
    # the file is getting deleted before we can use it when using tempfile.NamedTemporaryFile
    # so we will use a /var/tmp/ file instead
    scratch_filename = "/var/tmp/yt_audio.opus"

    # Download the video as an audio file
    yt_dlp_command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "opus",
        "--output",
        scratch_filename,
        video_url,
    ]

    # Run the command
    subprocess.run(yt_dlp_command)

    # Check if the file exists
    if not os.path.exists(scratch_filename):
        raise Exception(f"Failed to download the video as an audio file: {video_url}")

    # Return the filename
    return Path(scratch_filename)


### End youtube ###
