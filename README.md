# mobius
<img src="/logo.png" width="128" height="128" />

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

## Image Commands
- `/flux`: Generate images using Flux.
- `/dalle`: Generate images using DALL-E 3.
- `/unload_comfy`: Unload models from vRAM on the ComfyUI server.

## Utility Commands
- `/youtube`: Search youtube. Returns top result.
- `/temp`: Get the temperature.
- `/google`: Uses the google custom search api to get results from the web.

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


# tests

To run the tests, use `pytest`:
```bash
pytest
```

# license

```
MIT License

Copyright (c) 2024 Clayton Mullis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```