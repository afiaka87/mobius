import gradio as gr
from dotenv import load_dotenv

# Import Gradio interfaces from other files
from gradio_app import image_generation, audio, text, utility, video

# Load environment variables
load_dotenv()


def create_gradio_app():
    # Create blocks for each category
    with gr.Blocks(title="Discord Bot Web UI") as app:
        gr.Markdown("# Discord Bot Web UI")

        with gr.Tab("Image Generation"):
            image_generation.create_interfaces()

        with gr.Tab("Audio"):
            audio.create_interfaces()

        with gr.Tab("Text"):
            text.create_interfaces()

        with gr.Tab("Utility"):
            utility.create_interfaces()

        with gr.Tab("Video"):
            video.create_interfaces()

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
