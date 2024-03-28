# util.py - Utilities and helpers for discord bot
import time
import io
from PIL import Image
import datetime
import httpx
import discord
import os


def convert_base64_images_to_discord_files(
        base64_images) -> list[discord.File]:
    import time
    import io
    from PIL import Image
    import base64
    import datetime
    discord_files = []
    for base64_image in base64_images:
        stub = str(time.time()).replace(".", "")
        # datetime str for folder
        current_datetime = datetime.datetime.now().strftime("%Y_%m_%d")
        image_stub = f"images/{current_datetime}/{stub}.jpg"
        os.makedirs(os.path.dirname(image_stub), exist_ok=True)
        # if image is base64 str, convert to bytes
        # if image is bytes, do nothing
        if isinstance(base64_image, str):
            base64_image = base64.b64decode(base64_image)

        pil_image = Image.open(io.BytesIO(base64_image))
        pil_image.save(image_stub, "JPEG", quality=95)
        discord_files.append(discord.File(image_stub))
    return discord_files


def convert_base64_gifs_to_discord_files(base64_images) -> list[discord.File]:
    discord_files = []
    for base64_image in base64_images:
        stub = str(time.time()).replace(".", "")
        # datetime str for folder
        current_datetime = datetime.datetime.now().strftime("%Y_%m_%d")
        image_stub = f"images/{current_datetime}/{stub}.gif"
        os.makedirs(os.path.dirname(image_stub), exist_ok=True)
        # save as gif using PIL
        # does PIL support gif?
        #
        pil_gif = Image.open(io.BytesIO(base64_image))
        pil_gif.save(image_stub, "GIF", quality=95)
        discord_files.append(discord.File(image_stub))
    return discord_files