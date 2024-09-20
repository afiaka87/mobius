import discord
import httpx
import base64
import re
from ollama import AsyncClient
from discord import app_commands
from discord.ext import commands
from typing import Optional
from utils.image_utils import download_image
import os
import base64
import io


def markdown_header(title: str, text: str, level: int = 2) -> str:
    """
    Surround the text with markdown headers.
    """
    hashes = "#" * level
    return f"{hashes} {title}\n\n{text}"


def build_response_embed(
    prompt: str,
    model: str,
    system: str,
    image: str,
    temperature: float,
    top_k: int,
    top_p: float,
    num_predict: int,
) -> discord.Embed:
    """
    Build a discord.Embed object from the full_response.
    """
    embed = discord.Embed()
    embed.title = "Parameters"
    embed.add_field(name="Prompt", value=prompt)
    embed.add_field(name="Model", value=model)
    embed.add_field(name="System", value=system)
    embed.add_field(name="Image", value=image)
    embed.add_field(name="Temperature", value=temperature)
    embed.add_field(name="Top K", value=top_k)
    embed.add_field(name="Top P", value=top_p)
    embed.add_field(name="Num Predict", value=num_predict)
    return embed


@app_commands.command(
    description="Generate a response using an Ollama model.",
    name="ollama",
)
@app_commands.choices(
    model=[
        app_commands.Choice(
            name="llama3.1:8b-text-q4_K_M", value="llama3.1:8b-text-q4_K_M"
        ),
        app_commands.Choice(
            name="llama3.1:8b-instruct-q4_K_M", value="llama3.1:8b-instruct-q4_K_M"
        ),
        app_commands.Choice(
            name="llava:7b-v1.6-vicuna-q4_K_M", value="llava:7b-v1.6-vicuna-q4_K_M"
        ),
    ]
)
async def ollama_command(
    interaction: discord.Interaction,
    prompt: str,
    model: str = "llama3.1:8b-instruct-q4_K_M",
    system: Optional[str] = None,
    image: Optional[discord.Attachment] = None,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 100,
    top_p: Optional[float] = 0.9,
    num_predict: Optional[int] = 512,
):
    """
    Generate a response using an Ollama model.
    """
    await interaction.response.defer()

    try:

        # Prepare the messages
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        # Handle image for multimodal models
        if image:
            image_base64 = await image.read()
            messages[0]["images"] = [image_base64]

        # Prepare the options
        options = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": num_predict,
        }

        message = await interaction.followup.send(
            "Generating response...",
            embed=build_response_embed(
                prompt, model, system, image, temperature, top_k, top_p, num_predict
            ),
        )
        full_response = (
            prompt if "text" in model else ""
        )  # user prompt prepended on text completion models

        async for part in await AsyncClient().chat(
            model=model, messages=messages, stream=True, options=options
        ):
            full_response += part["message"]["content"]
            if len(full_response) % 25 == 0:  # Update every 25 characters
                await message.edit(
                    content=markdown_header("Ollama Response", full_response)
                )  # surround with code block backticks
        await message.edit(
            content=markdown_header("Ollama Response", full_response)
        )  # surround with code block backticks

    except Exception as e:
        await interaction.followup.send(f"An error occurred: {str(e)}")
        print(e)
        raise e


@app_commands.context_menu(name="Describe Image")
async def analyze_image_context_menu(
    interaction: discord.Interaction, message: discord.Message
):
    prompt: str = "Describe the image."
    model: str = "llava:7b-v1.6-vicuna-q4_K_M"
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 100
    top_p: Optional[float] = 0.9
    num_predict: Optional[int] = 512

    await interaction.response.defer()
    await interaction.followup.send(
        content="Generating response.",
    )  # surround with code block backticks

    try:
        # Prepare the messages
        messages = [{"role": "user", "content": prompt}]

        # Check if the message has an image in the content
        match = re.search(r"https?://\S+\.(png|jpg|jpeg|webp)", message.content)
        if match:
            image_url = match.group(0)

            # Download and encode the image
            os.makedirs(".cache", exist_ok=True)
            image_path = download_image(
                image_url, save_path=".cache/describe_image.png"
            )

            # load bytes to base64
            image_data = open(image_path, "rb").read()
            image_bytes = io.BytesIO(image_data)
            image_base64 = base64.b64encode(image_bytes.read()).decode("utf-8")
        elif message.attachments:
            image_url = message.attachments[0].url
            image_path = download_image(
                image_url, save_path=".cache/describe_image.png"
            )
            image_data = open(image_path, "rb").read()
            image_bytes = io.BytesIO(image_data)
            image_base64 = base64.b64encode(image_bytes.read()).decode("utf-8")
        else:
            raise ValueError("No image found in message.")

        messages[0]["images"] = [image_base64]

        # Prepare the options
        options = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": num_predict,
        }

        reply = await message.reply(
            "Generating response...",
        )
        full_response = (
            prompt if "text" in model else ""
        )  # user prompt prepended on text completion models

        async for part in await AsyncClient().chat(
            model=model, messages=messages, stream=True, options=options
        ):
            full_response += part["message"]["content"]
            if len(full_response) % 25 == 0:  # Update every 25 characters
                await reply.edit(
                    content=markdown_header("Ollama Response", full_response)
                )  # surround with code block backticks
        await reply.edit(
            content=markdown_header("Ollama Response", full_response)
        )  # surround with code block backticks

    except Exception as e:
        await interaction.followup.send(f"An error occurred: {str(e)}")
        print(e)
        raise e


def setup(bot: commands.Bot):
    bot.tree.add_command(ollama_command)
    bot.tree.add_command(analyze_image_context_menu)
