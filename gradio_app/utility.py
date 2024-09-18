import gradio as gr
from services import youtube, weather, google
import os


async def youtube_search(query):
    result = await youtube.get_top_youtube_result_httpx(
        query, api_key=os.getenv("YOUTUBE_API_KEY")
    )
    if "error" in result:
        return result["error"]
    else:
        return f"https://www.youtube.com/watch?v={result['videoId']}"


async def get_temperature():
    return await weather.temp_callback()


async def google_search(query):
    return await google.google_search(query)


def create_interfaces():
    with gr.Group():
        gr.Markdown("## YouTube Search")
        youtube_interface = gr.Interface(
            fn=youtube_search,
            inputs=gr.Textbox(label="Search Query"),
            outputs=gr.Textbox(label="Top Result URL"),
        )

    with gr.Group():
        gr.Markdown("## Get Temperature")
        temp_interface = gr.Interface(
            fn=get_temperature,
            inputs=[],
            outputs=gr.Textbox(label="Current Temperature"),
        )

    with gr.Group():
        gr.Markdown("## Google Search")
        google_interface = gr.Interface(
            fn=google_search,
            inputs=gr.Textbox(label="Search Query"),
            outputs=gr.Textbox(label="Search Results"),
        )
