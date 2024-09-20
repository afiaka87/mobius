from typing import List, Optional, Dict, Any, AsyncGenerator
import base64
import ollama

from PIL import Image


async def chat_with_ollama(
    model: str,
    messages: List[Dict[str, Any]],
    stream: bool = True,
    options: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    client = ollama.AsyncClient()

    try:
        response = await client.chat(
            model=model, messages=messages, stream=stream, options=options
        )
        yield response["message"]["content"]
    except ollama.ResponseError as e:
        yield f"Error: {e.error}"


async def list_models() -> List[Dict[str, Any]]:
    client = ollama.AsyncClient()
    return await client.list()


async def pull_model(model: str) -> AsyncGenerator[str, None]:
    client = ollama.AsyncClient()
    async for progress in await client.pull(model, stream=True):
        yield progress["status"]


def encode_image(image_path: str) -> str:
    """
    Open the image file with Pillow and encode it to base64
    """
    with Image.open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
