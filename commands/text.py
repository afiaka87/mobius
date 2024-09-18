import json
import os

import discord
from discord import app_commands
from discord.ext import commands

import constants
from services import chatgpt, claude


def create_temp_file(message_text: str) -> str:
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    temp_file.write(message_text.encode("utf-8"))
    temp_file.flush()
    temp_file.seek(0)
    temp_path = temp_file.name
    return temp_path


@app_commands.command(
    name="anthropic", description="Chat completion with Anthropic LLM models."
)
@app_commands.choices(
    model=[
        app_commands.Choice(name=model, value=model)
        for model in [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
    ]
)
async def anthropic(
    interaction: discord.Interaction,
    prompt: str,
    max_tokens: int = 1024,
    model: str = "claude-3-5-sonnet-20240620",
):
    await interaction.response.defer(ephemeral=False, thinking=True)
    message_text = await claude.anthropic_chat_completion(prompt, max_tokens, model)

    if len(message_text) >= 2000:
        await interaction.followup.send(
            content="Response too long, sending as a file.",
            file=discord.File(create_temp_file(message_text), filename="response.txt"),
        )
    else:
        username = interaction.user.name
        formatted_response = f"""### _{username}_: \n\n```txt\n{prompt}\n```\n### anthropic:\n\n {message_text}"""
        await interaction.followup.send(content=formatted_response)


@app_commands.command(
    name="gpt",
    description="Chat with GPT-4o. Supports history. Outputs as a discord embed.",
)
async def gpt(
    interaction: discord.Interaction,
    prompt: str,
    seed: int = -1,
    model_name: str = "gpt-4o-mini",
):
    await interaction.response.defer(ephemeral=False, thinking=True)

    # Load history from file (history.json)
    history = []
    if not os.path.exists("history.json"):
        with open("history.json", "a+") as f:
            json.dump([], f)  # Write an empty array to the file
            print("Created history file with empty array.")
    else:
        with open("history.json", "r") as f:
            history = json.load(f)
            print(f"Loaded history {len(history)} messages from file")

    # Check if the history is too long, trim
    max_history_size = 30
    if len(history) > max_history_size:
        history = history[
            -max_history_size:
        ]  # Trim the history to the last max_history_size messages
        print(f"Trimmed history to {max_history_size} messages")

    # if there is no system message, add the default system prompt
    # if the first message is not a system message, add the default system prompt
    if len(history) == 0 or history[0]["role"] != "system":
        history.insert(
            0,
            {
                "role": "system",
                "content": constants.LLM_DEFAULT_SYSTEM_PROMPT,
            },
        )
        print("Added default system prompt to history")

    # Append the user prompt to messages.
    content_array = [
        {"type": "text", "text": prompt}
    ]  # (we do it this way in case we want to use image_urls later)

    #  `66: openai.BadRequestError: Error code: 400 - {'error': 'Invalid 'content': 'image_url' field must be an object in the form { image_url: { url: "...base64 encoded image here..." } } Got 'object'.'}
    history.append({"role": "user", "content": content_array})
    print(f"Appended user prompt to history: {prompt}")

    assistant_response_message = await chatgpt.gpt_chat_completion(
        history,
        model_name,
        seed,
    )

    # Add the assistant response to the history
    history.append({"role": "assistant", "content": assistant_response_message})

    embed = discord.Embed(description=assistant_response_message)
    embed.add_field(name="Prompt", value=prompt[:1000], inline=False)
    embed.add_field(name="History Size", value=len(history))
    embed.add_field(name="Model", value=model_name)

    await interaction.followup.send(embed=embed)


@app_commands.command(name="refine", description="Refine a prompt using GPT-4-turbo.")
async def refine(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(ephemeral=False, thinking=True)
    refined_prompt = await chatgpt.refine_prompt(prompt)
    await interaction.followup.send(
        f"You asked for: ```txt\n{prompt}```\nGPT4 upscaled the prompt to: ```txt\n{refined_prompt}```\n"
        f"Copy and paste the upscaled prompt into `/imagine` to generate an image."
    )


def setup(bot: commands.Bot):
    bot.add_command(anthropic)
    bot.add_command(gpt)
    bot.add_command(refine)
