import glob
import os
import math
import random
from typing import List
import websockets  #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import websocket
import uuid
import json
import urllib.request
from pathlib import Path
import urllib.parse

# Comfy UI workflow descriptions

# get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# workloads are in project_root/backend/comfy_workloads
WORKLOADS_DIR = PROJECT_ROOT.joinpath("backend/comfy_workloads")

COMFY_UI_SDXL_AND_REFINER_PATH = WORKLOADS_DIR.joinpath(
    "comfy_ui_sdxl_and_refiner.json")
SDXL_TURBO_COMFY_PROMPT_PATH = WORKLOADS_DIR.joinpath(
    "sdxl_turbo_comfy_prompt.json")
SDXL_TO_SVD_PROMPT_PATH = WORKLOADS_DIR.joinpath("sdxl_to_svd_prompt.json")
IMAGE_TO_VIDEO_WORKFLOW_PATH = WORKLOADS_DIR.joinpath(
    "img_to_vid_interp_workflow_api.json")
# TEXT_TO_VID_ANIM_DIFF_WORKFLOW_PATH = WORKLOADS_DIR.joinpath( "animate_diff_txt_to_vid.json")
TEXT_TO_VID_ANIM_DIFF_WORKFLOW_PATH = WORKLOADS_DIR.joinpath(
    "anim_api.json")

HOST = "100.83.37.105"
PORT = 2211

server_address = f"{HOST}:{PORT}"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address),
                                 data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    print(f"Getting image: {data}")
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(
            server_address, url_values)) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(
            server_address, prompt_id)) as response:
        return json.loads(response.read())


async def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()

        if isinstance(out, str):
            print(f"Received: {out}")
            message = json.loads(out)

            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  #Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'],
                                           image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


async def get_images_clean(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    while True:
        out = await ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  #Execution is done
        else:
            continue  #previews are binary data


#Commented out code to display the output images:
async def comfyui_sdxl_callback(
        positive_prompt,
        negative_prompt="",
        cfg=7.0,
        sampler="dpmpp_2m_sde",  # DPM++ 3M SDE Karras
        scheduler="karras",
        height=1024,
        width=1024,
        steps=20,
        batch_size=1,
        seed=0) -> List[bytes]:
    comfy_ui_workflow_description = json.load(
        open(COMFY_UI_SDXL_AND_REFINER_PATH))

    #set the text prompt for our positive CLIPTextEncode
    comfy_ui_workflow_description["6"]["inputs"]["text"] = positive_prompt

    # set negative prompt to empty string
    comfy_ui_workflow_description["7"]["inputs"]["text"] = negative_prompt

    # "sampler_name": "dpmpp_2s_ancestral",
    # set the sampler for our KSampler node
    comfy_ui_workflow_description["3"]["inputs"]["sampler_name"] = sampler

    # "scheduler": "karras",
    # set the scheduler for our KSampler node
    comfy_ui_workflow_description["3"]["inputs"]["scheduler"] = scheduler

    #set the seed for our KSampler node
    comfy_ui_workflow_description["3"]["inputs"]["seed"] = seed

    # set the cfg
    comfy_ui_workflow_description["3"]["inputs"]["cfg"] = cfg

    # set the batch size
    comfy_ui_workflow_description["5"]["inputs"]["batch_size"] = batch_size

    # set the height and width for our EmptyLatentImage node
    comfy_ui_workflow_description["5"]["inputs"]["height"] = height
    comfy_ui_workflow_description["5"]["inputs"]["width"] = width

    # set the height and width for our EmptyLatentImage node
    comfy_ui_workflow_description["5"]["inputs"]["height"] = height
    comfy_ui_workflow_description["5"]["inputs"]["width"] = width

    #set the number of steps for our KSampler node
    comfy_ui_workflow_description["3"]["inputs"]["steps"] = steps

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = await get_images(ws, comfy_ui_workflow_description)

    all_image_data = []
    for node_id in images:
        for image_data in images[node_id]:
            all_image_data.append(image_data)
    return all_image_data


async def comfy_sdxl_turbo_callback(
        positive_prompt,
        negative_prompt="",
        steps=1,
        height=512,
        width=512,
        batch_size=1,
        seed=0):

    if seed == 0:
        seed = random.randint(0, int(math.pow(2, 32)))

    comfy_ui_workflow_description = json.load(
        open(SDXL_TURBO_COMFY_PROMPT_PATH))

    # set the text prompt for our negative CLIPTextEncode
    comfy_ui_workflow_description["2"]["inputs"]["text"] = negative_prompt

    #set the text prompt for our positive CLIPTextEncode
    comfy_ui_workflow_description["3"]["inputs"]["text"] = positive_prompt

    #set the number of steps for our SDXLTurboScheduler node
    comfy_ui_workflow_description["7"]["inputs"]["steps"] = steps

    #set the seed for our KSampler node
    comfy_ui_workflow_description["4"]["inputs"]["noise_seed"] = seed

    #set the height and width for our EmptyLatentImage node
    comfy_ui_workflow_description["8"]["inputs"]["height"] = height
    comfy_ui_workflow_description["8"]["inputs"]["width"] = width

    # set the batch size for our EmptyLatentImage node
    comfy_ui_workflow_description["8"]["inputs"]["batch_size"] = batch_size

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = await get_images(ws, comfy_ui_workflow_description)

    all_image_data = []
    for node_id in images:
        for image_data in images[node_id]:
            all_image_data.append(image_data)
    return all_image_data


async def comfyui_svd_callback(
    positive_prompt,
    negative_prompt="",
    height=1024,
    width=576,
    motion_bucket_id=31,
    fps=6,
    augmentation_level=0.0,
    pingpong=False,
):

    comfy_ui_workflow_description = json.load(open(SDXL_TO_SVD_PROMPT_PATH))

    # set the text prompt for our negative CLIPTextEncode
    comfy_ui_workflow_description["19"]["inputs"]["text"] = negative_prompt

    #set the text prompt for our positive CLIPTextEncode
    comfy_ui_workflow_description["18"]["inputs"]["text"] = positive_prompt

    #set the height and width for our EmptyLatentImage node and our SVD_img2vid_Conditioning node
    comfy_ui_workflow_description["12"]["inputs"]["height"] = height
    comfy_ui_workflow_description["12"]["inputs"]["width"] = width

    comfy_ui_workflow_description["22"]["inputs"]["height"] = height
    comfy_ui_workflow_description["22"]["inputs"]["width"] = width

    #set the motion bucket id for our SVD_img2vid_Conditioning node
    comfy_ui_workflow_description["12"]["inputs"][
        "motion_bucket_id"] = motion_bucket_id

    #set the fps for our SVD_img2vid_Conditioning node
    comfy_ui_workflow_description["12"]["inputs"]["fps"] = fps

    #set the augmentation level for our SVD_img2vid_Conditioning node
    comfy_ui_workflow_description["12"]["inputs"][
        "augmentation_level"] = augmentation_level

    #set the pingpong for our VHS_VideoCombine node
    comfy_ui_workflow_description["23"]["inputs"]["pingpong"] = pingpong

    # connect asynchonously
    async with websockets.connect("ws://{}/ws?clientId={}".format(
        server_address, client_id)) as ws:
        # ws = websocket.WebSocket()
        # ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
        images = await get_images_clean(ws, comfy_ui_workflow_description)

        # get the saved gif from the SVD node
        # gif_image_path = get_images(ws, comfy_ui_workflow_description)["23"][0]
        # images = get_images(ws, comfy_ui_workflow_description)

        # get the saved mp4 from the SVD node
        video_path = "/home/sam/Projects/ComfyUI/output/svd/"

        # most recently updated gif in the output folder

        # start with "svd" and ending with ".gif"
        list_of_files = glob.glob(video_path + '*.gif')
        latest_file = max(list_of_files, key=os.path.getctime)

        print(f"Latest file: {latest_file}")

        return latest_file


async def comfyui_animatediff_img2vid_callback(
    image_path: str,
    height: int = 576,
    width: int = 576,  # TODO guess this from image
    motion_bucket_id: int = 31,
    fps: int = 6,
    augmentation_level: float = 0.0,
    pingpong: bool = False,
):
    # load json in one line
    image_to_video_interp_workflow = json.load(
        open(IMAGE_TO_VIDEO_WORKFLOW_PATH))
    # set the image path (25th node)
    image_to_video_interp_workflow["25"]["inputs"]["image"] = image_path
    # set the height and width for our EmptyLatentImage node and our SVD_img2vid_Conditioning node
    image_to_video_interp_workflow["12"]["inputs"]["height"] = height
    image_to_video_interp_workflow["12"]["inputs"]["width"] = width
    #set the motion bucket id for our SVD_img2vid_Conditioning node
    image_to_video_interp_workflow["12"]["inputs"][
        "motion_bucket_id"] = motion_bucket_id
    #set the fps for our SVD_img2vid_Conditioning node
    image_to_video_interp_workflow["12"]["inputs"]["fps"] = fps
    #set the augmentation level for our SVD_img2vid_Conditioning node
    image_to_video_interp_workflow["12"]["inputs"][
        "augmentation_level"] = augmentation_level
    #set the pingpong for our VHS_VideoCombine node
    image_to_video_interp_workflow["23"]["inputs"]["pingpong"] = pingpong
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    # get the saved gif from the SVD node

    # get the saved mp4 from the SVD node
    video_path = "/home/sam/Projects/ComfyUI/output/"

    # most recently updated gif in the output folder

    # get the saved gif from the SVD node
    # gif_image_path = get_images(ws, comfy_ui_workflow_description)["23"][0]
    # images = get_images(ws, comfy_ui_workflow_description)
    images = await get_images(ws, image_to_video_interp_workflow)

    list_of_files = glob.glob(video_path + '*.mp4')
    latest_file = max(list_of_files, key=os.path.getctime)

    print(f"Latest file: {latest_file}")

    return latest_file


async def comfyui_animatediff_txt2vid_callback(
    positive_prompt: str,
    negative_prompt:
    str = "",
    width: int = 512,
    height: int = 512,
    cfg: float = 7.5,
    steps: int = 20,
    num_frames: int = 16,  # has to be less than 32
    pingpong: bool = False,
    fps: int = 16,
    seed: int = 0,
):
    if seed == 0:
        seed = random.randint(0, int(math.pow(2, 32)))

    # load json in one line
    txt_to_vid_anim_diff_workflow = json.load(
        open(TEXT_TO_VID_ANIM_DIFF_WORKFLOW_PATH))

    # set the text prompt for our positive CLIPTextEncode (3rd node, inputs text)
    txt_to_vid_anim_diff_workflow["3"]["inputs"]["text"] = positive_prompt

    # set the text prompt for our negative CLIPTextEncode (6th node, inputs text)
    txt_to_vid_anim_diff_workflow["6"]["inputs"]["text"] = negative_prompt

    # set the height and width for our EmptyLatentImage node (node 9)
    txt_to_vid_anim_diff_workflow["9"]["inputs"]["height"] = height
    txt_to_vid_anim_diff_workflow["9"]["inputs"]["width"] = width

    # set the num_frames for our SVD_img2vid_Conditioning node
    txt_to_vid_anim_diff_workflow["9"]["inputs"]["batch_size"] = num_frames

    # set the cfg for our KSampler node (7th node, inputs cfg)
    txt_to_vid_anim_diff_workflow["7"]["inputs"]["cfg"] = cfg

    # set the steps for our KSampler node (7th node, inputs steps)
    txt_to_vid_anim_diff_workflow["7"]["inputs"]["steps"] = steps

    # set the seed for our KSampler node (7th node, inputs seed)
    txt_to_vid_anim_diff_workflow["7"]["inputs"]["seed"] = seed

    # set the pingpong for our VHS_VideoCombine node (35th node, inputs pingpong)
    txt_to_vid_anim_diff_workflow["35"]["inputs"]["pingpong"] = pingpong

    # set the fps for our VHS_VideoCombine node (35th node, inputs frame_rate)
    txt_to_vid_anim_diff_workflow["35"]["inputs"]["frame_rate"] = fps

    # connect asynchonously
    async with websockets.connect("ws://{}/ws?clientId={}".format(
        server_address, client_id)) as ws:
        # get the saved gif from the SVD node
        # gif_image_path = get_images(ws, comfy_ui_workflow_description)["23"][0]
        # images = get_images(ws, comfy_ui_workflow_description)
        await get_images_clean(ws, txt_to_vid_anim_diff_workflow)

    # get the saved gif from the animate_diff node
    video_path = "/home/sam/Projects/ComfyUI/output/"

    # most recently updated gif in the output folder starting with "aaa_readme_" and ending with ".gif"
    list_of_files = glob.glob(video_path + 'aaa_readme_*.gif')
    latest_file = max(list_of_files, key=os.path.getctime)

    print(f"Latest file: {latest_file}")

    return latest_file