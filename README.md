# mobius
<img src="/logo.jpg" style="max-width:100px;">

A discord bot by/for Clay. Currently a work-in-progress. Not really meant for public consumption.

# installation

```bash
git clone https://github.com/afiaka87/mobius.git
cd mobius
pip install -r requirements.txt
```

Fill out the `.env` file with the necessary environment variables:
```bash
export DISCORD_API_TOKEN=
export DISCORD_GUILD_ID=
export HUGGINGFACE_TOKEN=
export INVOKEAI_TAG=
export INVOKEAI_GIT=
export OPENAI_API_KEY=
export YOUTUBE_API_KEY=
```

Then run the bot:
```bash
bash run.sh
```


# usage

## slash commands

- `/dream` - generate dream-like images
- `/turbo` - generate turbo images
- `/svd` - generate a video from text using SVD
- `/transcribe` - transcribe a youtube video
- `/youtube` - search youtube
- `/temp` - get the temperature
- `/gpustat` - get the GPU stats
- `/llm` - chat with GPT-4-turbo
- `/dalle2` - generate an image from text using DALL-E 2
- `/dalle3` - generate an image from text using DALL-E 3
- `/refine` - refine a prompt using GPT-4-turbo
- `/anim` - generate with AnimateDiff
- `/img2vid` - generate a video from an image
- `/midj` - generate a prompt from a list of prompts