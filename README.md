# mobius
<img src="/logo.jpg" width="200" height="200" />

A discord bot by/for Clay. Currently a work-in-progress. Not really meant for public consumption.

# installation

```bash
git clone https://github.com/afiaka87/mobius.git
cd mobius
python3 -m venv .venv
source .venv/bin/activate
(.venv) pip install -r requirements.txt
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
(venv) python3 main.py
```


# usage

## slash commands

Here's a list of available slash commands:

### Audio Commands
- `/wav`: Generate a song or sound from text using stable audio open.
- `/say`: Generate speech from text using OpenAI's TTS API.

### Text Commands
- `/anthropic`: Chat with Claude AI using the Anthropic API.
- `/gpt`: Chat with GPT models using the OpenAI API.
- `/refine`: Refine a prompt using GPT-4-turbo.

Note: This list may not be exhaustive and could change as the bot is developed further. For the most up-to-date list of commands and their descriptions, use the Discord built-in slash command interface or check the files in the `commands` directory.

# daemon

Systemd is used to run the bot as a daemon. To install the daemon, write the following to `/etc/systemd/system/mobius.service` (be sure to change the working directory to the correct path):

```
[Unit]
Description=Clays discord bot mobius.
After=network.target

[Service]
ExecStart=/bin/bash -c '/home/change/me/mobius/.venv/bin/python /home/change/me/mobius/main.py'

User=root

[Install]
WantedBy=multi-user.target
```

Then, daemon-reload:
```bash
sudo systemctl daemon-reload
```

And enable the service:
```bash
sudo systemctl enable mobius.service
```

And start the service:
```bash
sudo systemctl start mobius.service
```

And check the status:
```bash
sudo systemctl status mobius.service
```
