import gradio as gr
from services import fal_ai, comfy_ui, chatgpt


async def flux(prompt, model, image_size, guidance_scale):
    image_url = await fal_ai.generate_flux_image(
        prompt, model, image_size, guidance_scale
    )
    return image_url


async def flux_img(clip_text, image_url, model_name, megapixels, steps, denoise):
    image_bytes_list = await comfy_ui.flux_img_to_img(
        image_url, clip_text, model_name, megapixels, steps, denoise
    )
    return image_bytes_list[0] if image_bytes_list else None


async def unload_comfy():
    response = await comfy_ui.unload_comfy_via_api()
    return f"Response: {response}"


async def dalle(prompt, quality, size):
    image_b64_list, refined_prompt = await chatgpt.dalle_callback(prompt, quality, size)
    return image_b64_list[0] if image_b64_list else None, refined_prompt


def create_interfaces():
    with gr.Group():
        gr.Markdown("## Flux")
        flux_interface = gr.Interface(
            fn=flux,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Dropdown(
                    ["fal-ai/flux/dev", "fal-ai/flux/schnell"],
                    label="Model",
                    value="fal-ai/flux/schnell",
                ),
                gr.Dropdown(
                    [
                        "landscape_4_3",
                        "landscape_16_9",
                        "portrait_3_4",
                        "portrait_9_16",
                        "square",
                        "square_hd",
                    ],
                    label="Image Size",
                    value="landscape_16_9",
                ),
                gr.Slider(0.1, 10.0, value=3.5, label="Guidance Scale"),
            ],
            outputs=gr.Image(type="filepath", label="Generated Image"),
        )

    with gr.Group():
        gr.Markdown("## Flux Image-to-Image")
        flux_img_interface = gr.Interface(
            fn=flux_img,
            inputs=[
                gr.Textbox(label="CLIP Text"),
                gr.Image(type="filepath", label="Input Image"),
                gr.Dropdown(
                    ["flux1-dev-fp8.safetensors", "flux1-schnell.sft"],
                    label="Model Name",
                    value="flux1-dev-fp8.safetensors",
                ),
                gr.Slider(0.1, 5.0, value=1.0, label="Megapixels"),
                gr.Slider(1, 100, value=20, step=1, label="Steps"),
                gr.Slider(0.1, 1.0, value=0.75, label="Denoise"),
            ],
            outputs=gr.Image(type="numpy", label="Generated Image"),
        )

    with gr.Group():
        gr.Markdown("## Unload ComfyUI")
        unload_comfy_interface = gr.Interface(
            fn=unload_comfy, inputs=[], outputs=gr.Textbox(label="Response")
        )

    with gr.Group():
        gr.Markdown("## DALL-E")
        dalle_interface = gr.Interface(
            fn=dalle,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Radio(["standard", "hd"], label="Quality", value="standard"),
                gr.Dropdown(
                    ["1024x1024", "1792x1024", "1024x1792"],
                    label="Size",
                    value="1024x1024",
                ),
            ],
            outputs=[
                gr.Image(type="numpy", label="Generated Image"),
                gr.Textbox(label="Refined Prompt"),
            ],
        )
