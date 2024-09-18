import discord
from discord import app_commands
from discord.ext import commands
from services import chatgpt, comfy_ui, fal_ai
from utils.image_utils import convert_base64_images_to_discord_files

@app_commands.command(name="flux", description="Run txt to img w FLUX SCHNELL")
@app_commands.choices(
    model=[
        app_commands.Choice(name="fal-ai/flux/dev", value="fal-ai/flux/dev"),
        app_commands.Choice(name="fal-ai/flux/schnell", value="fal-ai/flux/schnell"),
    ],
    image_size=[
        app_commands.Choice(name="landscape_4_3", value="landscape_4_3"),
        app_commands.Choice(name="landscape_16_9", value="landscape_16_9"),
        app_commands.Choice(name="portrait_3_4", value="portrait_3_4"),
        app_commands.Choice(name="portrait_9_16", value="portrait_9_16"),
        app_commands.Choice(name="square", value="square"),
        app_commands.Choice(name="square_hd", value="square_hd"),
    ]
)
async def flux(
    interaction: discord.Interaction,
    prompt: str,
    model: str = "fal-ai/flux/schnell",
    image_size: str = "landscape_16_9",
    guidance_scale: float = 3.5,
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    image_url = await fal_ai.generate_flux_image(prompt, model, image_size, guidance_scale)
    image_urls = [image_url]
    output = f"Prompt: **`{prompt}`** \nModel: **`{model}`**\nImage Size: **`{image_size}`**\n\n"
    output += "\n".join(image_urls)
    await interaction.followup.send(content=output)

@app_commands.command(name="flux-img", description="Run img to img w FLUX DEV")
@app_commands.choices(
    model_name=[
        app_commands.Choice(name="flux1-dev-fp8.safetensors", value="flux1-dev-fp8.safetensors"),
        app_commands.Choice(name="flux1-schnell.sft", value="flux1-schnell.sft"),
    ]
)
async def flux_img(
    interaction: discord.Interaction,
    clip_text: str,
    image_url: str = "",
    model_name: str = "flux1-dev-fp8.safetensors",
    megapixels: float = 1.0,
    steps: int = 20,
    denoise: float = 0.75,
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    image_bytes_list = await comfy_ui.flux_img_to_img(
        image_url, clip_text, model_name, megapixels, steps, denoise
    )
    discord_images = convert_base64_images_to_discord_files(image_bytes_list)
    await interaction.followup.send(
        content=f"Prompt: **`{clip_text}`**\nImage URL: **`{image_url}`**",
        files=discord_images,
    )

@app_commands.command(name="unload_comfy", description="Unload models from vRAM on the ComfyUI server.")
async def unload_comfy(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=False, thinking=True)
    response = await comfy_ui.unload_comfy_via_api()
    await interaction.followup.send(f"Response: {response}")

@app_commands.command(name="dalle", description="Generate images using DALL-E 3")
@app_commands.choices(
    quality=[
        app_commands.Choice(name="standard", value="standard"),
        app_commands.Choice(name="hd", value="hd"),
    ],
    size=[
        app_commands.Choice(name="1024x1024", value="1024x1024"),
        app_commands.Choice(name="1792x1024", value="1792x1024"),
        app_commands.Choice(name="1024x1792", value="1024x1792"),
    ],
)
async def dalle(
    interaction: discord.Interaction,
    prompt: str,
    quality: str = "standard",
    size: str = "1024x1024",
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    image_b64_list, refined_prompt = await chatgpt.dalle_callback(prompt, quality, size)
    discord_images = convert_base64_images_to_discord_files(image_b64_list)
    await interaction.followup.send(
        content=f"Prompt: **`{prompt}`**\nRefined Prompt: **`{refined_prompt}`**",
        files=discord_images,
    )

def setup(bot: commands.Bot):
    bot.add_command(fal_ai)
    bot.add_command(flux_img)
    bot.add_command(unload_comfy)
    bot.add_command(dalle)