# Stream Status Monitor

Polls Cloudflare Stream API and sends Discord notifications when a stream goes live.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Discord bot token with appropriate permissions

## Environment Variables

Set these in a `.env` file or your environment:

```
DISCORD_API_TOKEN=your_discord_bot_token
DISCORD_MENTION_USER_ID=your_discord_user_id
CLOUDFLARE_LIVE_INPUT_KEY=your_cloudflare_api_key
```

## Running

```bash
uv run python send_status.py
```

## systemd User Service Setup

1. Copy the service file:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp stream-status.service ~/.config/systemd/user/
   ```

2. Edit the service file and update the paths:
   ```bash
   nano ~/.config/systemd/user/stream-status.service
   ```
   - Set `WorkingDirectory` to this directory
   - Set `ExecStart` to your `uv` binary path (find with `which uv`)

3. Reload systemd and enable:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable stream-status
   systemctl --user start stream-status
   ```

4. Check status:
   ```bash
   systemctl --user status stream-status
   journalctl --user -u stream-status -f
   ```
