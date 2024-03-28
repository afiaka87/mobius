# mobius
![mobius logo](/logo.jpg)

A discord bot by/for Clay. Currently a work-in-progress. Not really meant for public consumption.

# installation

```bash
git clone https://github.com/afiaka87/mobius.git
cd mobius
pip install -r requirements.txt
```

# usage

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