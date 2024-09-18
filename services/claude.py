from anthropic import AsyncAnthropic


async def anthropic_chat_completion(
    prompt: str,
    max_tokens: int,
    model: str = "claude-3-5-sonnet-20240620",
) -> str:
    anthropic_client = (
        AsyncAnthropic()
    )  # Automatically detects API key in env var ANTHROPIC_API_KEY

    create_message_args_dict = {
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"},
    }

    message = await anthropic_client.messages.create(**create_message_args_dict)
    message_text = message.content[0].text  # Message -> list[TextBlock] -> str

    return message_text
