import gradio as gr
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Import Gradio interfaces from other files
from gradio_app import image_generation, audio, text, utility, video


def create_gradio_app():
    # Create blocks for each category
    with gr.Blocks(title="Lame-othy") as app:
        gr.Markdown("Lame-othy")

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
    parser = argparse.ArgumentParser(description="Run the Discord Bot Web UI")
    parser.add_argument(
        "--listen",
        default="127.0.0.1",
        help="IP address to listen on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    args = parser.parse_args()

    app = create_gradio_app()
    app.queue()  # Enable queueing for handling concurrent requests
    app.launch(server_name=args.listen, server_port=args.port, share=True)
