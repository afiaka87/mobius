import csv
import random
import tempfile
import datetime
from pathlib import Path
import subprocess
import logging
import os
import time
import discord
from backend import comfy_ui as comfy_client
from faster_whisper import WhisperModel
import openai
from discord.app_commands import choices, Choice
from backend.local_api import (
    dalle2_callback,
    dalle3_callback,
    dream_callback,
    gpustat_callback,
    llm_callback,
    refine_prompt_callback,
    temp_callback,
)
from bot.constants import DEFAULT_NEGATIVE_PROMPT


from bot.util import (
    convert_base64_images_to_discord_files,
)
from backend.whisper_util import COMPUTE_TYPE, DEVICE, MODEL_SIZE
from discord.ext.commands import Bot

logging.basicConfig(level=logging.INFO)

openai.api_key = os.environ["OPENAI_API_KEY"]

intents = discord.Intents.default()
intents.message_content = True

discord_client = Bot("!", intents=intents)
cmd_tree = discord_client.tree
guild = discord.Object(id=os.environ["DISCORD_GUILD_ID"])

filename = "midjprompts.csv"

# Read the first column values into a list
with open(filename, "r", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    column_values = [row[0] for row in reader]


@cmd_tree.command(
    name="dream",
    description="Text -> Image generation using Stable Diffusion XL.",
    guild=guild,
)
@choices(
    scheduler=[  # 256x256, 512x512, or 1024x1024
        Choice(name="normal", value="normal"),
        Choice(name="karras", value="karras"),
        Choice(name="exponential", value="exponential"),
        Choice(name="sgm_uniform", value="sgm_uniform"),
        Choice(name="simple", value="simple"),
        Choice(name="ddim_uniform", value="ddim_uniform"),
    ],
    sampler=[
        Choice(name="euler", value="euler"),
        Choice(name="euler_ancestral", value="euler_ancestral"),
        Choice(name="heun", value="heun"),
        Choice(name="heunpp2", value="heunpp2"),
        Choice(name="dpm_2", value="dpm_2"),
        Choice(name="dpm_2_ancestral", value="dpm_2_ancestral"),
        Choice(name="lms", value="lms"),
        Choice(name="dpm_fast", value="dpm_fast"),
        Choice(name="dpm_adaptive", value="dpm_adaptive"),
        Choice(name="dpmpp_2s_ancestral", value="dpmpp_2s_ancestral"),
        Choice(name="dpmpp_sde", value="dpmpp_sde"),
        Choice(name="dpmpp_sde_gpu", value="dpmpp_sde_gpu"),
        Choice(name="dpmpp_2m", value="dpmpp_2m"),
        Choice(name="dpmpp_2m_sde", value="dpmpp_2m_sde"),
        Choice(name="dpmpp_2m_sde_gpu", value="dpmpp_2m_sde_gpu"),
        Choice(name="dpmpp_3m_sde", value="dpmpp_3m_sde"),
        Choice(name="dpmpp_3m_sde_gpu", value="dpmpp_3m_sde_gpu"),
        Choice(name="ddpm", value="ddpm"),
        Choice(name="lcm", value="lcm"),
    ],
)
async def dream(
    interaction: discord.Interaction,
    prompt: str,
    refine: bool = False,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    cfg: float = 7.0,
    height: int = 1024,
    width: int = 1024,
    steps: int = 30,
    sampler: str = "dpmpp_2m_sde",
    scheduler: str = "karras",
    num_images: int = 1,
    seed: int = 0,
):
    """
    Generates dream-like images based on the given prompt using the Stable Diffusion XL model.

    Parameters:
        - interaction (discord.Interaction): The interaction object representing the user's command.
        - prompt (str): The prompt for generating the dream-like images.
        - refine (bool, optional): Whether to refine the prompt using negative prompts. Defaults to True.
        - negative_prompt (str, optional): The negative prompt for refining the prompt. Defaults to DEFAULT_NEGATIVE_PROMPT.
        - height (int, optional): The height of the generated images. Defaults to 1024.
        - width (int, optional): The width of the generated images. Defaults to 1024.
        - steps (int, optional): The number of steps for generating the images. Defaults to 20.
        - sampler (str, optional): The sampler to use for generating the images. Defaults to "dpmpp_2m_sde".
        - scheduler (str, optional): The scheduler to use for generating the images. Defaults to "karras".
        - batch_size (int, optional): The number of images to generate in each batch. Defaults to 1.
        - seed (int, optional): The seed value for generating the images. Defaults to 0 for a random seed.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)
    current_time = time.time()

    # convert sampler and scheduler to strings
    sampler = str(sampler)
    scheduler = str(scheduler)

    if prompt == "random":
        # Select a random value from the list
        random_value = random.choice(column_values)

        print(random_value)
        prompt = random_value
        # let the user know what the random prompt is (in progress still though)
        await interaction.followup.send(
            f"Detected `random` as prompt, Chose prompt: \n```txt\n{prompt}\n```\nGenerating image..."
        )

    try:
        output_image_data, refined_prompt, updated_seed = await dream_callback(
            prompt=prompt,
            refine=refine,
            negative_prompt=negative_prompt,
            cfg=cfg,
            height=height,
            width=width,
            steps=steps,
            sampler=sampler,
            scheduler=scheduler,
            batch_size=num_images,
            seed=seed,
        )

        # convert the base64 images to discord files and send them in a message
        discord_files = convert_base64_images_to_discord_files(output_image_data)
        discord_files = discord_files[:num_images]

        embed = discord.Embed()
        embed.add_field(name="Resolution", value=f"{width}x{height}", inline=True)
        embed.add_field(name="Steps", value=steps, inline=True)
        embed.add_field(name="Sampler", value=sampler, inline=True)
        embed.add_field(name="Scheduler", value=scheduler, inline=True)
        embed.add_field(name="Seed", value=updated_seed, inline=True)
        embed.add_field(name="Classifier-free Guidance", value=cfg, inline=True)
        embed.add_field(name="Prompt", value=prompt[:1000] + ".", inline=False)
        if negative_prompt != DEFAULT_NEGATIVE_PROMPT:
            embed.add_field(
                name="Negative prompt",
                value=negative_prompt[:1024].strip(),
                inline=False,
            )
        if refine and refined_prompt:  # refined_prompt may be None
            embed.add_field(
                name="Refined prompt",
                value=refined_prompt[:1024].strip(),
                inline=False,
            )
    except Exception as e:
        # tell them to use /dalle bc /dream is down
        await interaction.followup.send(
            f"`/dream` is down. Clay is prolly batin'. Try again once more, or use `/dalle2` or `/dalle3` instead.\n"
        )
        print(f"Error: {e}")
        return

    elapsed_time = time.time() - current_time

    # send the message
    await interaction.followup.send(
        content=f"`/dream` finished in`{elapsed_time:.2f} seconds`",
        embed=embed,
        files=discord_files,
    )


async def download_youtube_video_as_audio(video_url: str):
    """Download a youtube video as an audio file.

    Args:
        video_url (str): The URL of the video to download. Must be a valid youtube URL. Example: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    Raises:
        Exception: A temporary file is created and subsequently uploaded. If the file is not created, an exception is raised.

    Returns:
        Path: The path to the downloaded audio file.
    """
    if os.path.exists("scratch.opus"):
        os.remove("scratch.opus")

    # download the video as an audio file
    yt_dlp_command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "opus",
        "--output",
        "scratch.opus",
        video_url,
    ]

    # run the command
    subprocess.run(yt_dlp_command)

    # check if the file exists
    if not os.path.exists("scratch.opus"):
        raise Exception(
            "File scratch.opus does not exist. yt-dlp failed to download the video."
        )

    # return the filename
    return Path("scratch.opus")


@cmd_tree.command(
    name="transcribe",
    description="Transcribe a youtube video using OpenAI Whisper Large-V3 (8-bit quantized)",
    guild=guild,
)
async def transcribe(
    interaction: discord.Interaction,
    youtube_url: str,
):
    """Transcribe a youtube video using OpenAI Whisper Large-V3 (8-bit quantized).

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        youtube_url (str): The URL of the youtube video to transcribe.

    Raises:
        Exception: If the audio file does not exist or is not a valid format, an exception is raised.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)
    # # audiofile can be path or str
    audio_file = await download_youtube_video_as_audio(youtube_url)

    # async def transcribe(discord_interaction: discord.Interaction, audio_file: Union[Path, str]) -> str:
    if isinstance(audio_file, str):
        audio_file = Path(audio_file)

    # make sure the audio file exists
    if not audio_file.exists():
        raise Exception("Audio file '%s' does not exist" % audio_file)

    # make sure format is correct
    if not audio_file.suffix.lower() in [
        ".wav",
        ".mp3",
        ".ogg",
        ".flac",
        ".m4a",
        ".opus",
    ]:
        raise Exception("Audio file '%s' is not a valid format" % audio_file)

    # load whisper model
    # tell the user we're loading the model
    message = await interaction.followup.send("Loading model...")
    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    await message.edit(content="Model loaded!")  # type: ignore

    # initialize transcriber. won't transcribe until enumerated
    await message.edit(content="Transcribing...")  # type: ignore
    segments, info = whisper_model.transcribe(
        str(audio_file), beam_size=5, vad_filter=True
    )
    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    await message.edit(  # type: ignore
        content=f"Determined language: {info.language} with probability {info.language_probability}"
    )

    # filename should be timestamped and unique, ends with .txt
    now = datetime.datetime.now()
    transcript_filename = now.strftime("%Y-%m-%d-%H-%M-%S") + ".txt"

    # save the segments to the file
    await message.edit(content="Transcribing...")  # type: ignore
    with open(transcript_filename, "w") as f:
        segment_history = []  # for storing groups of segments to edit message with
        for idx, segment in enumerate(segments):
            segment_as_timestamp_line = "[%.2fs -> %.2fs] %s\n" % (
                segment.start,
                segment.end,
                segment.text,
            )
            f.write(segment_as_timestamp_line)  # write to file
            segment_history.append(segment_as_timestamp_line)  # add to segment history
            # every four segments, edit message with group of four segments
            if idx % 8 == 0:
                grouping = "".join(segment_history)
                # grouping should be less than 2000 characters
                if len(grouping) > 2000:
                    grouping = grouping[:1964] + "\n..."

                grouping = "\n```txt\n" + grouping + "```"
                try:
                    await message.edit(content=grouping)  # type: ignore
                except Exception as e:
                    await message.edit(content=f"Error: {e}")  # type: ignore
                # clear segment history
                segment_history = []

    await message.edit(content="Transcription complete!")  # type: ignore

    # send the message
    await message.reply(  # type: ignore
        f"Transcribed video: {youtube_url}",
        files=[discord.File(transcript_filename)],
    )


@cmd_tree.command(name="turbo", description="Generate with SDXL-Turbo", guild=guild)
async def turbo(
    interaction: discord.Interaction,
    prompt: str,
    refine: bool = True,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    height: int = 512,
    width: int = 512,
    batch_size: int = 4,
    steps: int = 1,
    seed: int = 0,
):
    """
    Generate turbo images based on the given prompt.

    Parameters:
    - interaction: The Discord interaction object.
    - prompt: The prompt for generating turbo images.
    - refine: Whether to refine the prompt or not. Default is True.
    - negative_prompt: The negative prompt for generating turbo images. Default is DEFAULT_NEGATIVE_PROMPT.
    - height: The height of the generated images. Default is 512.
    - width: The width of the generated images. Default is 512.
    - batch_size: The batch size for generating turbo images. Default is 4.
    - steps: The number of steps for generating turbo images. Default is 1.
    - seed: The seed for generating turbo images. Default is 0.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)

    # if refine_prompt is true, refine the prompt
    refined_prompt = None
    if refine:
        refined_prompt = await refine_prompt_callback(prompt)

    output_image_data = await comfy_sdxl_turbo_callback(
        positive_prompt=refined_prompt if refine else prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
    )

    # convert the base64 images to discord files and send them in a message
    discord_files = convert_base64_images_to_discord_files(output_image_data)

    embed = discord.Embed()
    embed.add_field(name="Resolution", value=f"{width}x{height}", inline=True)
    embed.add_field(name="Steps", value=steps, inline=True)
    embed.add_field(name="Seed", value=seed, inline=True)
    embed.add_field(name="Prompt", value=prompt[:1000] + "...", inline=False)
    embed.add_field(
        name="Negative prompt",
        value=negative_prompt[:160].strip() + "...",
        inline=False,
    )
    if refine:
        embed.add_field(name="Refined prompt", value=refined_prompt, inline=False)

    # send the message
    await interaction.followup.send(
        content=f"Thanks for using `/turbo`! The quality is worse than `/dream`, but the speed makes up for it. Like `/dream`, it may go down from time to time. Sorry. ",
        embed=embed,
        files=discord_files,
    )


@cmd_tree.command(name="svd", description="Generate with SVD", guild=guild)
async def svd(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 1024,
    height: int = 576,
    motion_bucket_id: int = 31,
    fps: int = 6,
    augmentation_level: float = 0.0,
    pingpong: bool = False,
):
    """
    Perform Singular Value Decomposition (SVD) on an image based on the given prompt.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        prompt (str): The positive prompt for generating the image.
        negative_prompt (str, optional): The negative prompt for generating the image. Defaults to DEFAULT_NEGATIVE_PROMPT.
        width (int, optional): The width of the generated image. Defaults to 1024.
        height (int, optional): The height of the generated image. Defaults to 576.
        motion_bucket_id (int, optional): The motion bucket ID for generating the image. Defaults to 31.
        fps (int, optional): The frames per second for generating the image. Defaults to 6.
        augmentation_level (float, optional): The augmentation level for generating the image. Defaults to 0.0.
        pingpong (bool, optional): Whether to generate the image in pingpong mode. Defaults to False.
    """
    # Defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)

    start_time = time.time()

    output_image_path = await comfy_client.comfyui_svd_callback(
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        motion_bucket_id=motion_bucket_id,
        fps=fps,
        augmentation_level=augmentation_level,
        pingpong=pingpong,
    )

    discord_files = [discord.File(output_image_path)]

    # stop the timer
    duration_in_seconds = time.time() - start_time

    # send the message
    await interaction.followup.send(
        f"Generated prompt: `{prompt}` in `{duration_in_seconds:.2f} seconds`",
        files=discord_files,
    )


async def get_top_youtube_result_httpx(search_query, api_key):
    """
    Calls the YouTube Search API and fetches the top search result for a given query.

    Parameters:
    search_query (str): The search query string.
    api_key (str): Your YouTube Data API key.

    Returns:
    dict: Information about the top search result, or an error message if the call fails.
    """
    import httpx

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


@cmd_tree.command(
    name="youtube", description="Search youtube. Returns top result.", guild=guild
)
async def youtube(
    interaction: discord.Interaction,
    query: str,
):
    """Search youtube and return the top result.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        query (str): The search query for youtube.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)
    key = os.environ["YOUTUBE_API_KEY"]

    result = await get_top_youtube_result_httpx(query, key)

    if "error" in result:
        await interaction.followup.send(result["error"])
    else:
        # format url
        url = f"https://www.youtube.com/watch?v={result['videoId']}"
        await interaction.followup.send(url)


@cmd_tree.command(
    name="anim", description="Generate with AnimateDiff. Work in progress.", guild=guild
)
async def anim(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 512,
    height: int = 512,
    cfg: float = 7.5,
    steps: int = 20,
    num_frames: int = 16,  # has to be less than 32
    pingpong: bool = False,
    fps: int = 16,
    seed: int = 0,
):
    """Generate an animated video based on the given prompt using AnimateDiff.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        prompt (str): The positive prompt for generating the video.
        negative_prompt (str, optional): The negative prompt for generating the video. Defaults to DEFAULT_NEGATIVE_PROMPT.
        width (int, optional): The width of the generated video. Defaults to 512.
        height (int, optional): The height of the generated video. Defaults to 512.
        cfg (float, optional): The classifier-free guidance for generating the video. Defaults to 7.5.
        steps (int, optional): The number of steps for generating the video. Defaults to 20.
        num_frames (int, optional): The number of frames for generating the video. Defaults to 16.
        fps (int, optional): The frames per second for generating the video. Defaults to 16.
        seed (int, optional): The seed for generating the video. Defaults to 0 for a random seed.
    """
    try:
        if seed == 0:
            seed = random.randint(0, 1000000)
        # defer w/ thinking to show the bot is working
        await interaction.response.defer(ephemeral=False, thinking=True)

        start_time = time.time()
        output_image_path = await comfy_client.comfyui_animatediff_txt2vid_callback(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            cfg=cfg,
            steps=steps,
            num_frames=num_frames,
            pingpong=pingpong,
            fps=fps,
            seed=seed,
        )

        discord_files = [discord.File(output_image_path)]

        # stop the timer
        duration_in_seconds = time.time() - start_time

        # send the message
        await interaction.followup.send(
            f"Generated video from prompt: \n```txt\n{prompt}\n```\n in `{duration_in_seconds:.2f} seconds`",
            files=discord_files,
        )
    except Exception as e:
        await interaction.followup.send(f"Error: {e}")


@cmd_tree.command(
    name="img2vid", description="Generate a video from an image (WIP)", guild=guild
)
async def img2vid(
    interaction: discord.Interaction,
    image: discord.Attachment,
    width: int = 576,
    height: int = 576,
    motion_bucket_id: int = 31,
    fps: int = 6,
    augmentation_level: float = 0.0,
    pingpong: bool = False,
):
    """Generate a video from an image using AnimateDiff.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        image (discord.Attachment): The image to generate the video from.
        width (int, optional): The width of the generated video. Defaults to 576.
        height (int, optional): The height of the generated video. Defaults to 576.
        motion_bucket_id (int, optional): The motion bucket ID for generating the video. Defaults to 31.
        fps (int, optional): The frames per second for generating the video. Defaults to 6.
        augmentation_level (float, optional): The augmentation level for generating the video. Defaults to 0.0.
        pingpong (bool, optional): Whether to generate the video in pingpong mode. Defaults to False.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)

    start_time = time.time()

    uploaded_image_path = Path(f"uploaded_image_{time.time()}.png")
    await image.save(uploaded_image_path)

    output_image_path = await comfy_client.comfyui_animatediff_img2vid_callback(
        image_path=str(uploaded_image_path.resolve()),
        width=width,
        height=height,
        motion_bucket_id=motion_bucket_id,
        fps=fps,
        augmentation_level=augmentation_level,
        pingpong=pingpong,
    )

    discord_files = [discord.File(uploaded_image_path), discord.File(output_image_path)]

    # stop the timer
    duration_in_seconds = time.time() - start_time

    # send the message
    await interaction.followup.send(
        f"Generated video from image in `{duration_in_seconds:.2f} seconds. It's probably bad. Sorry.`",
        files=discord_files,
    )


@cmd_tree.command(name="temp", description="Get the temperature.", guild=guild)
async def temp(interaction: discord.Interaction):
    """Get the temperature in Fayetteville, AR.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
    """
    await interaction.response.defer(ephemeral=False, thinking=True)
    friendly_current_temperature = await temp_callback()
    await interaction.followup.send(friendly_current_temperature)


@cmd_tree.command(name="gpustat", description="Get the GPU stats.", guild=guild)
async def gpustat(interaction: discord.Interaction):
    """Get the GPU vram.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
    """
    await interaction.response.defer(ephemeral=False, thinking=True)
    gpu_usage = await gpustat_callback()
    await interaction.followup.send(gpu_usage)


@cmd_tree.command(name="llm", description="Chat with GPT-4-turbo.", guild=guild)
async def llm(
    interaction: discord.Interaction,
    prompt: str,
    system_prompt: str = """Your name is Mobius. Do not offer unprompted advice or clarifications. Be concise. Speak in specific, topic relevant terminology. Do NOT hedge or qualify. Do not waffle. Speak directly and be willing to make creative guesses. Be willing to reference less reputable sources for ideas. Never apologize.  Now - you will see a user prompt/question. REMEMBER - REPEAT their prompt first before responding. Your reply should always start in the format of "User: " and the user's question/prompt, then your response as "Mobius: " and your response. Be willing to reference less reputable sources for ideas. Never apologize.""",
):
    """Chat with GPT-4-turbo.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        prompt (str): The prompt for the chat.
        system_prompt (str): The system prompt for the chat. Set to the string "None" for no system prompt. Defaults to the default system prompt.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)

    if system_prompt.lower() == "none":
        messages = (
            []
        )  # messages list may start with nothing if user provided None for system prompt
    else:
        messages = [{"role": "system", "content": system_prompt}]

    # append the user prompt to messages
    messages.append({"role": "user", "content": prompt})

    # call the OpenAI API
    print(f"Calling API with messages: {messages}")
    chatgpt_answer = await llm_callback(messages=messages)
    chatgpt_answer = chatgpt_answer["choices"][0]["message"]["content"]

    # if the prompt is too long, save it to a file and send the file
    if len(chatgpt_answer) > 2000:
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(chatgpt_answer)
        await interaction.followup.send(
            f"The response was too long for discord. Here's the response in a file.",
            file=discord.File(f.name),
        )
    else:
        await interaction.followup.send(chatgpt_answer)


@cmd_tree.command(
    name="dalle2",
    description="Generate an image from text using DALL-E 2. Much cheaper than DALL-E3, but worse quality.",  # TODO remove alpha when done
    guild=guild,
)
@choices(
    resolution=[  # 256x256, 512x512, or 1024x1024
        Choice(name="small", value="256x256"),
        Choice(name="medium", value="512x512"),
        Choice(name="large", value="1024x1024"),
    ]
)
async def dalle2(
    interaction: discord.Interaction,
    prompt: str,
    resolution: Choice[str] = "512x512",
    num_images: int = 1,
    refine: bool = False,
):
    """Generate an image from text using DALL-E 2.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        prompt (str): The prompt for generating the image.
        resolution (Choice[str], optional): The resolution of the generated image. Defaults to "512x512". Choices are "256x256", "512x512", or "1024x1024".
        num_images (int, optional): The number of images to generate. Defaults to 1.
        refine (bool, optional): Whether to refine the prompt or not. Defaults to True.
    """
    # defer w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)

    # if resolution is str, convert to Choice
    if isinstance(resolution, str):
        resolution = Choice(name=resolution, value=resolution)

    # call the backend
    try:
        output_image_data, refined_prompt = await dalle2_callback(
            prompt=prompt,
            refine_prompt=refine,
            refine=refine,
            resolution=resolution.value,
            num_images=num_images,
        )
    except Exception as e:
        await interaction.followup.send(
            f"Error occurred. Yell at clay. If it's a 400 error your prompt is too naughty: {e}"
        )
        return

    discord_files = []
    for image_data in output_image_data:
        generation_b64 = image_data["b64_json"]
        discord_files.extend(convert_base64_images_to_discord_files([generation_b64]))

    embed = discord.Embed()
    embed.add_field(name="Prompt", value=prompt, inline=True)
    embed.add_field(name="Resolution", value=resolution.value, inline=True)

    # footer for revised prompt
    if refined_prompt:
        embed.set_footer(text=f"{refined_prompt}")

    # send the message
    await interaction.followup.send(
        content=f"Thanks for using `/dalle2`! It's much cheaper than `/dalle3` but the quality is worse. Sorry.",
        embed=embed,
        files=discord_files,
    )


@cmd_tree.command(
    name="dalle3",
    description="Generate an image from text using DALL-E 3. Please use sparingly.",  # TODO remove alpha when done
    guild=guild,
)
async def dalle3(
    interaction: discord.Interaction,
    prompt: str,
):
    """Generate an image from text using DALL-E 3.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        prompt (str): The prompt for generating the image.
    """
    # Defer ("[bot] is thinking...") w/ thinking to show the bot is working
    await interaction.response.defer(ephemeral=False, thinking=True)
    try:
        generation_b64, refined_prompt = await dalle3_callback(prompt=prompt)
    except Exception as e:
        await interaction.followup.send(
            f"Error occurred. Yell at clay. If it's a 400 error your prompt is too naughty: {e}"
        )
        return

    discord_files = convert_base64_images_to_discord_files([generation_b64])

    # Create a `discord.Embed` object with the prompt and resolution
    embed = discord.Embed()
    embed.add_field(name="Prompt", value=prompt, inline=True)
    embed.add_field(name="Resolution", value="1024x1024", inline=True)
    embed.set_footer(text=refined_prompt)

    # Send the message with the embed and the image
    await interaction.followup.send(
        content=f"Please use `/dalle3` sparingly. Try using `/dream` (free), or `/dalle2` (much cheaper).",
        embed=embed,
        files=discord_files,
    )


@cmd_tree.command(
    name="refine",
    description="Refine a prompt using GPT-4-turbo.",
    guild=guild,
)
async def refine(
    interaction: discord.Interaction,
    prompt: str,
):
    """Refine a prompt using GPT-4-turbo.

    Args:
        interaction (discord.Interaction): The Discord interaction object.
        prompt (str): The prompt to refine.
    """
    await interaction.response.defer(ephemeral=False, thinking=True)
    chatgpt_answer = await refine_prompt_callback(prompt)

    # send the message
    await interaction.followup.send(
        f"You asked for: ```txt\n{prompt}```\nGPT4 upscaled the prompt to: ```txt\n{chatgpt_answer}```\nCopy and paste the upscaled prompt into `/dream`, `/turbo`, or `/svd` to generate an image.",
    )


@discord_client.event
async def on_ready():
    print(f"Logged in as {discord_client.user}!")
    await discord_client.wait_until_ready()

    # trick to speed up syncing
    cmd_tree.copy_global_to(guild=guild)

    # we have to sync again to get the commands to show up
    await cmd_tree.sync(guild=guild)
    # register llm manually
    print("Slash commands synced! Ready to go!")


if __name__ == "__main__":
    discord_client.run(os.environ["DISCORD_API_TOKEN"])
