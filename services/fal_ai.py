import fal_client


async def generate_flux_image(
    prompt: str,
    model: str,
    image_size: str,
    guidance_scale: float,
) -> str:

    handler = fal_client.submit(
        model,
        arguments={
            "prompt": prompt,
            "image_size": image_size,
            "guidance_scale": guidance_scale,
        },
    )

    result = handler.get()
    return result["images"][0]["url"]
