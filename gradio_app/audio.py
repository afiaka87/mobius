import gradio as gr
from utils import audio_utils, video_utils
from services import chatgpt


async def wav(prompt, duration, steps, cfg_scale, sigma_min, sigma_max, sampler_type):
    output, sample_rate = await audio_utils.load_and_run_sao_model(
        prompt, duration, steps, cfg_scale, sigma_min, sigma_max, sampler_type
    )
    waveform_video_filename = video_utils.convert_audio_to_waveform_video(
        output, sample_rate, duration
    )
    return waveform_video_filename


async def say(text, voice, speed):
    speech_file_path = await chatgpt.generate_speech(text, voice, float(speed))
    video_file_path = speech_file_path.with_suffix(".mp4")
    video_file_path = video_utils.convert_audio_to_waveform_video(
        speech_file_path, video_file_path
    )
    return video_file_path


def create_interfaces():
    with gr.Group():
        gr.Markdown("## Generate WAV")
        wav_interface = gr.Interface(
            fn=wav,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Slider(1, 60, value=10, step=1, label="Duration (seconds)"),
                gr.Slider(10, 200, value=100, step=1, label="Steps"),
                gr.Slider(1, 20, value=7, step=1, label="CFG Scale"),
                gr.Slider(0.1, 1.0, value=0.3, label="Sigma Min"),
                gr.Slider(100, 1000, value=500, step=1, label="Sigma Max"),
                gr.Dropdown(
                    ["dpmpp-3m-sde"], label="Sampler Type", value="dpmpp-3m-sde"
                ),
            ],
            outputs=gr.Video(label="Generated Audio Waveform"),
        )

    with gr.Group():
        gr.Markdown("## Text-to-Speech")
        say_interface = gr.Interface(
            fn=say,
            inputs=[
                gr.Textbox(label="Text to Speak"),
                gr.Dropdown(
                    ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    label="Voice",
                    value="onyx",
                ),
                gr.Dropdown(
                    ["0.5", "1.0", "1.25", "1.5", "2.0"], label="Speed", value="1.0"
                ),
            ],
            outputs=gr.Video(label="Generated Speech Waveform"),
        )
