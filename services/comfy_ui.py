import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import websockets
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.image_utils import download_image

# Configure logging
logger = logging.getLogger(__name__)


HOST = "archbox"
PORT = 2211
server_address = f"{HOST}:{PORT}"
client_id = str(uuid.uuid4())


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
async def queue_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        payload = {"prompt": workflow, "client_id": client_id}
        response = await client.post(f"http://{server_address}/prompt", json=payload)
        response.raise_for_status()
        return response.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
async def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    async with httpx.AsyncClient() as client:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = await client.get(f"http://{server_address}/view", params=data)
        response.raise_for_status()
        return response.content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
async def get_history(workflow_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://{server_address}/history/{workflow_id}")
        response.raise_for_status()
        return response.json()


async def get_images(
    ws: websockets.WebSocketClientProtocol, workflow: Dict[str, Any]
) -> Dict[str, Union[bytes, Any]]:
    workflow_id = (await queue_workflow(workflow))["prompt_id"]
    output_images, current_node = {}, ""

    try:
        async for message in ws:
            if isinstance(message, str):
                data = json.loads(message)
                if (
                    data["type"] == "executing"
                    and data["data"]["prompt_id"] == workflow_id
                ):
                    current_node = data["data"]["node"]
                    if current_node is None:
                        break
            elif current_node == SAVE_IMAGE_WEBSOCKET_NODE:
                output_images.setdefault(current_node, []).append(message[8:])
    except websockets.exceptions.ConnectionClosed as exc:
        logger.error(f"WebSocket connection closed: {exc}")
        raise
    except json.JSONDecodeError as exc:
        logger.error(f"Error decoding WebSocket message: {exc}")
        raise

    return output_images


def prepare_workflow(
    workflow_text: str, positive_text: str, seed: Optional[int] = None
) -> Dict[str, Any]:
    workflow = json.loads(workflow_text)
    workflow[POSITIVE_CLIP_TEXT_ENCODE_NODE]["inputs"]["text"] = positive_text
    workflow[KSAMPLER_NODE]["inputs"]["seed"] = seed or random.randint(0, 10000000)
    return workflow


def process_images(images: Dict[str, Union[bytes, Any]]) -> List[bytes]:
    image_bytes_list = []
    for node_id, image_data_list in images.items():
        for image_data in image_data_list:
            image_bytes_list.append(image_data)
    return image_bytes_list


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (
            websockets.exceptions.ConnectionClosed,
            ConnectionRefusedError,
        )
    ),
)
async def connect_websocket(uri: str, workflow: Dict[str, Any]) -> List[bytes]:
    async with websockets.connect(
        uri, max_size=100000000
    ) as ws:  # we have to increase the max_size to 100MB because the images are too large
        return process_images(await get_images(ws, workflow))


flux_img_to_img_workflow = """
{
  "6": {
    "inputs": {
      "text": "anime style",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI_FLUX_IMG_TO_IMG",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "FLUX1/ae.sft"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-schnell.sft",
      "weight_dtype": "fp8_e4m3fn"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "30",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "dpmpp_2m"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "simple",
      "steps": 20,
      "denoise": 0.75,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "12",
        0
      ],
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": 909703098000590
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "26": {
    "inputs": {
      "image": "Screenshot 2024-07-31 at 10.00.56 PM.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "29": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1,
      "image": [
        "26",
        0
      ]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {
      "title": "ImageScaleToTotalPixels"
    }
  },
  "30": {
    "inputs": {
      "pixels": [
        "29",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  }
}
"""

OBS_CACHE_DIR = Path("/home/sam/.cache/mobius/obs_images/")
OBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def flux_img_to_img(
    image_url: str,
    clip_text: str = "",
    model_name: str = "flux1-dev-fp8.safetensors",
    megapixels: float = 1.0,
    steps=20,
    denoise=0.75,
) -> List[bytes]:

    if model_name not in ["flux1-dev-fp8.safetensors", "flux1-schnell.sft"]:
        raise ValueError("Invalid model name")

    # create a temporary directory to store the image
    # with tempfile.TemporaryDirectory() as temp_dir:
    # save the image to the temporary directory

    # we need to know the image format to save it with the correct extension
    #
    # we can use the `imghdr` module to determine the image format
    # https://docs.python.org/3/library/imghdr.html

    # output_path needs to be ~/.cache/flux/ create if not exists

    if image_url == "" or image_url is None:
        # use the existing saved image, if it exists
        output_path = OBS_CACHE_DIR.joinpath(f"obs_image.png")
    else:
        save_path = Path("~/.cache/flux/").expanduser() / "input_image.png"
        output_path = download_image(image_url, save_path=save_path)

    # load the workflow
    workflow = json.loads(flux_img_to_img_workflow)

    # set the image path (26th node)
    output_path = str(output_path.resolve())  # use resolve to get the full path
    print(f"output_path: {output_path}")
    workflow["26"]["inputs"]["image"] = str(output_path)

    # set the clip text (6th node)
    workflow["6"]["inputs"]["text"] = clip_text

    # set the unet model name (12th node)
    workflow["12"]["inputs"]["unet_name"] = model_name

    # dev uses 20 steps, schnell uses just 4
    if model_name == "flux1-dev-fp8.safetensors":
        workflow["17"]["inputs"]["steps"] = steps
    elif model_name == "flux1-schnell.sft":
        print()
        workflow["17"]["inputs"]["steps"] = 4

    # set the megapixels (29th node)
    workflow["29"]["inputs"]["megapixels"] = megapixels

    # set the denoise ratio (17th node)
    workflow["17"]["inputs"]["denoise"] = denoise

    # connect to the websocket
    uri = f"ws://{server_address}/ws?clientId={client_id}"
    image_bytes_list = await connect_websocket(uri, workflow)

    return image_bytes_list


async def unload_comfy_via_api():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://archbox:2211/api/free",
            headers={
                "Accept": "*/*",
                "Referer": "http://archbox:2211/",
                "Content-Type": "application/json",
                "Origin": "http://archbox:2211",
            },
            json={"unload_models": True},
        )
        return response
