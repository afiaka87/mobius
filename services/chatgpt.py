import os
from pathlib import Path

import httpx
import openai
from openai import OpenAI

from utils.video_utils import convert_audio_to_waveform_video

OPENAI_HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json",
}

import constants

DIFFUSION_MASTER_SYSTEM_PROMPT = constants.DIFFUSION_MASTER_SYSTEM_PROMPT


async def refine_prompt(prompt: str, gpt_model_name: str = "gpt-4o") -> str:

    # TODO use openai library instead of httpx
    # Configure your OpenAI API call using httpx
    url = "https://api.openai.com/v1/chat/completions"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=5, pool=5)
    ) as httpx_client:
        response = await httpx_client.post(
            url,
            headers=OPENAI_HEADERS,
            json={
                "model": gpt_model_name,
                "messages": [
                    {"role": "system", "content": DIFFUSION_MASTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        # If there was a 400, When the OpenAI API returns a 400 status code, it indicates a Bad Request error. This error typically means there was something wrong with your request's syntax or the parameters provided. For more detailed information about what's specifically wrong, you should look at the response body returned by the API. It usually contains a "error" field providing a more detailed message about what went wrong. To retrieve and understand the detailed error message, ensure your request handling code reads the response body and logs it or displays it in a way you can inspect.

        # handle 400
        if response.status_code == 400:
            # get the error message
            error_message = response.json()["error"]
            # send the error message
            return f"Error: {error_message}. If it's 400 your prompt was too fucked up. Please try again."
        # raise for status but print the response first
        response.raise_for_status()
        chatgpt_answer = response.json()["choices"][0]["message"]["content"]
        # the upscaled prompt is always inside of a code block (with a potentially random/empty programming language specified)
        print(f"chatgpt_answer: {chatgpt_answer}")
        chatgpt_answer = chatgpt_answer.split("```")[1]
        chatgpt_answer = chatgpt_answer.replace("vbnet", "")
        chatgpt_answer = chatgpt_answer.replace("{", "")
        chatgpt_answer = chatgpt_answer.replace("}", "")
        chatgpt_answer = chatgpt_answer.strip()
        chatgpt_answer = chatgpt_answer.strip('"')

    return chatgpt_answer


async def gpt_chat_completion(
    messages: list,
    model_name: str,
    seed: int = None,
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
    temperature: float = None,
    top_p: float = 1,
    seed: int = None,
    max_tokens: int = None,
) -> dict:
    # TODO remove or use this
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


async def dalle2_callback(
    prompt: str, refine_prompt: bool, refine: bool, resolution: str, num_images: int
) -> tuple:
    # TODO use `openai` library instead of `httpx`

    refined_prompt = None
    if refine_prompt:
        refined_prompt = await refine_prompt_callback(prompt)

    url = "https://api.openai.com/v1/images/generations"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=5, pool=5)
    ) as httpx_client:
        response = await httpx_client.post(
            url,
            headers=OPENAI_HEADERS,
            json={
                "model": "dall-e-2",
                "prompt": refined_prompt if refine else prompt,
                "size": resolution,
                "n": num_images,
                "response_format": "b64_json",  # needed to get the base64 image data instead of a URL
            },
        )
        response.raise_for_status()
        output_image_data = response.json()["data"]
        return output_image_data, refined_prompt


import enum


class DalleQuality(enum.Enum):
    STANDARD = "standard"
    HD = "hd"


class DalleModel(enum.Enum):
    DALLE_3 = "dall-e-3"
    DALLE_2 = "dall-e-2"


async def dalle_callback(
    prompt: str,
    quality: str = "standard",
    resolution: str = "1024x1024",
    model: DalleModel = DalleModel.DALLE_3,
) -> tuple[list, str]:
    url = "https://api.openai.com/v1/images/generations"

    openai_headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }

    # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024 for dall-e-2. Must be one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3 models.
    # https://platform.openai.com/docs/api-reference/images/create
    if model == DalleModel.DALLE_2:
        assert resolution in ["256x256", "512x512", "1024x1024"]
    elif model == DalleModel.DALLE_3:
        assert resolution in ["1024x1024", "1792x1024", "1024x1792"]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=5, pool=5)
    ) as httpx_client:
        response = await httpx_client.post(
            url,
            headers=openai_headers,
            json={
                "model": model.value,
                "prompt": prompt,
                "size": resolution,
                "n": 1,
                "response_format": "b64_json",
                "style": "vivid",
                "quality": quality,
            },
        )
        image_data = response.json()["data"]
        image_b64_list = []
        refined_prompt = ""  # TODO this is only for DALL-E 3
        for img in image_data:
            generation_b64 = img["b64_json"]
            # TODO this only needs to be returned once, also only works for DALL-E 3
            if model == DalleModel.DALLE_3:
                refined_prompt = img["revised_prompt"]
            image_b64_list.append(generation_b64)
        return image_b64_list, refined_prompt


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
        audio_file=speech_file_path, video_file=video_file_path
    )
    return video_file_path
