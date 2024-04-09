# File should contain one fucntion per each discord app command, but only represents non-discord/frontend logic.
# This is specifically for the backend of the discord bot.
# Meant to decouple the discord bot from the backend logic.

# Path: claxy/backend.py

import os
import random

import httpx
import python_weather

from backend.comfy_ui import comfyui_sdxl_callback
from bot.constants import DEFAULT_NEGATIVE_PROMPT, DIFFUSION_MASTER_SYSTEM_PROMPT

OPENAI_HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json",
}


async def refine_prompt_callback(prompt: str) -> str:
    gpt_model_name: str = "gpt-4-0125-preview"

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


DEFAULT_LLM_SYSTEM_PROMPT = {
    "role": "system",
    "content": f"""Your name is Mobius. Do not offer unprompted advice or clarifications. Be concise. Speak in specific, topic relevant terminology. Do NOT hedge or qualify. Do not waffle. Speak directly and be willing to make creative guesses. Be willing to reference less reputable sources for ideas. Never apologize.  Now - you will see a user prompt/question. REMEMBER - REPEAT their prompt first before responding. Your reply should always start in the format of "User: " and the user's question/prompt, then your response as "Mobius: " and your response. Be willing to reference less reputable sources for ideas. Never apologize.""",
}

async def llm_callback(messages: list, temperature: float = None, top_p: float = 1, seed: int = None, max_tokens: int = None) -> dict:
    url = "https://api.openai.com/v1/chat/completions"

    data_dict = {
        "model": "gpt-4-vision-preview",  # TODO
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
            url,
            headers=OPENAI_HEADERS,
            json=data_dict,
        )
        response.raise_for_status()
        return response.json()


async def dalle2_callback(
    prompt: str, refine_prompt: bool, refine: bool, resolution: str, num_images: int
) -> tuple:
    # Now we need a "refined" prompt using the `gob` command.
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


async def dalle3_callback(prompt: str) -> tuple:
    url = "https://api.openai.com/v1/images/generations"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=5, pool=5)
    ) as httpx_client:
        response = await httpx_client.post(
            url,
            headers=OPENAI_HEADERS,
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "size": "1024x1024",  # TODO support landscape, portrait, square
                "n": 1,
                "response_format": "b64_json",
                "style": "natural",
                "quality": "hd"
            },
        )
        image_data = response.json()["data"][0]

        generation_b64 = image_data["b64_json"]
        refined_prompt = image_data["revised_prompt"]

        return generation_b64, refined_prompt


async def dream_callback(
    prompt: str,
    refine: bool = True,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    cfg: float = 7.5,
    height: int = 1024,
    width: int = 1024,
    steps: int = 35,
    sampler: str = "dpmpp_2m_sde",
    scheduler: str = "karras",
    batch_size: int = 1,  # TODO
    seed: int = 0,
) -> tuple:
    # if refine_prompt is true, refine the prompt
    refined_prompt = None
    if refine:
        refined_prompt = await refine_prompt_callback(prompt)

    if seed == 0:
        seed = random.randint(0, 1000000)

    output_image_data = await comfyui_sdxl_callback(
        positive_prompt=refined_prompt if refine else prompt,
        negative_prompt=negative_prompt,
        cfg=cfg,
        height=height,
        width=width,
        sampler=sampler,
        scheduler=scheduler,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )

    return output_image_data, refined_prompt, seed


async def gpustat_callback() -> str:
    import subprocess

    gpustat_output = subprocess.check_output(["gpustat", "--no-header"]).decode("utf-8")
    # the used vs total vram is the third column (split by | character)
    gpu_usage = gpustat_output.split("|")[2].strip()
    gpu_usage = f"VRAM usage: {gpu_usage}"
    return gpu_usage


async def temp_callback() -> str:
    async with python_weather.Client(unit=python_weather.IMPERIAL) as weather_client:  # type: ignore
        current_weather = await weather_client.get("Fayetteville, AR")
        temperature = current_weather.current.temperature
        return f"The current temperature in Fayetteville, AR is {temperature}°F."


