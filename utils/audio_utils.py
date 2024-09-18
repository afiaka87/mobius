import asyncio
import gc

import torch
from einops import rearrange


@torch.inference_mode()
async def load_and_run_sao_model(
    prompt, duration, steps, cfg_scale, sigma_min, sigma_max, sampler_type
):
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    # Function to run blocking code in a separate thread
    def run_model():
        # Model, config setup
        # sao_model, sao_model_config = load_sao_model()
        sao_model, sao_model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        sample_rate = sao_model_config["sample_rate"]
        sample_size = sao_model_config["sample_size"]

        # Move model to gpu if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sao_model = sao_model.to(device)

        # Set up text and timing conditioning
        conditioning = [
            {
                "prompt": prompt,
                "seconds_start": 0,  # Start time in seconds
                "seconds_total": duration,  # Total time in seconds
            }
        ]

        # Generate stereo audio autocast to fp16
        output = generate_diffusion_cond(
            sao_model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=sigma_min,  # Minimum noise level
            sigma_max=sigma_max,  # Maximum noise level
            sampler_type=sampler_type,
            device=device,
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output_npy = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        # Audio should only be duration seconds long, so clip it
        output_npy = output_npy[:, : sample_rate * duration]

        # Move model to cpu and clean up
        sao_model = sao_model.to("cpu")
        output = output.to("cpu")

        # Clean up variables
        del sao_model
        del sao_model_config
        del output

        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Garbage collect
        gc.collect()
        # Synchronize CUDA
        torch.cuda.synchronize()

        # Return the output and sample rate
        return output_npy, sample_rate

    # Run the blocking code in a separate thread
    return await asyncio.to_thread(run_model)
