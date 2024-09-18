import base64
import imghdr
import io
import os
import tempfile
from pathlib import Path
from typing import Union

import discord
import httpx
from PIL import Image


def convert_base64_images_to_discord_files(base64_images) -> list[discord.File]:
    discord_files = []
    for base64_image in base64_images:
        image_stub = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        os.makedirs(os.path.dirname(image_stub), exist_ok=True)

        # if image is base64 str, convert to bytes
        if isinstance(base64_image, str):
            base64_image = base64.b64decode(base64_image)

        pil_image = Image.open(io.BytesIO(base64_image))
        pil_image.save(image_stub, "JPEG", quality=95)
        discord_files.append(discord.File(image_stub))
    return discord_files


def convert_base64_gifs_to_discord_files(base64_images) -> list[discord.File]:
    discord_files = []
    for base64_image in base64_images:
        image_stub = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
        os.makedirs(os.path.dirname(image_stub), exist_ok=True)
        pil_gif = Image.open(io.BytesIO(base64_image))
        pil_gif.save(image_stub, "GIF", quality=95)
        discord_files.append(discord.File(image_stub))
    return discord_files


def download_image(image_url: str, save_path: Union[str, Path]) -> Path:
    """
    Download an image from the given URL and save it to the specified path.

    Args:
        image_url (str): The URL of the image to download.
        save_path (Union[str, Path]): The path where the image will be saved.
    """
    # Send a GET request to the image URL
    response = httpx.get(image_url)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Get the image content from the response
    image_content = response.content

    # Determine the image file type/extension
    image_type = imghdr.what(None, h=image_content)

    if image_type is None:
        raise ValueError("Unable to determine the image file type.")

    # Create a Path object for the save path
    save_path = Path(save_path)

    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the image with the correct extension
    with open(save_path.with_suffix(f".{image_type}"), "wb") as file:
        file.write(image_content)

    print(f"Image downloaded and saved as {save_path.with_suffix(f'.{image_type}')}")

    # Return (full) save path with the correct extension

    return save_path.with_suffix(f".{image_type}")


async def upload_image_to_imgbb(
    image: Union[bytes, Path], expiration: int = None
) -> str:
    """Upload an image to the ImgBB service and return the URL.

    Args:
        image (Union[bytes, Path]): The image to upload. Can be a Path object or bytes.
        expiration (int, optional): The expiration time for the image. Defaults to None.

    Returns:
        str: The URL of the uploaded image.
    """
    url = "https://api.imgbb.com/1/upload"

    # Determine if the image is a Path or bytes and prepare the payload accordingly
    if isinstance(image, Path):
        with open(image, "rb") as img_file:
            image_data = img_file.read()
    else:
        image_data = image

    # Prepare the payload
    payload = {"key": os.environ["IMGBB_API_KEY"], "image": image_data}

    if expiration:
        payload["expiration"] = expiration

    async with httpx.AsyncClient() as client:
        response = await client.post(url, files={"image": image_data}, data=payload)
        response.raise_for_status()
        return response.json()["data"]["url"]


async def download_image_as_b64(
    image_url: str,
) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        image = response.content
        image_base64 = base64.b64encode(image).decode("utf-8")
        # needs to be in form data:image/jpeg;base64,{image}
        return f"data:image/jpeg;base64,{image_base64}"
