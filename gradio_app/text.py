from services import claude, chatgpt, ollama_service
import gradio as gr
from ollama import AsyncClient

OLLAMA_API_BASE = "http://localhost:11434"


async def anthropic(prompt, max_tokens, model="claude-3-5-sonnet-20240620"):
    message_text = await claude.anthropic_chat_completion(prompt, max_tokens, model)
    return message_text


async def gpt(prompt, seed, model_name="gpt-4o-mini"):
    history = [{"role": "system", "content": "You are a helpful AI assistant."}]
    history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    assistant_response_message = await chatgpt.gpt_chat_completion(
        history, model_name, seed
    )
    return assistant_response_message


async def refine(prompt):
    refined_prompt = await chatgpt.refine_prompt(prompt)
    return refined_prompt


async def ollama_chat(
    model, prompt, system, image, temperature, top_k, top_p, num_predict, stream
):
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    if image:
        # image may come from discord or gradio, so we check if it's a discord attachment
        if hasattr(image, "read"):
            image_data = image.read()
            image_base64 = ollama_service.encode_image(image_data)
        else:  # already in base64
            image_data = image
            image_base64 = image_data
        messages[0]["images"] = [image_base64]

    options = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_predict": num_predict,
    }

    full_response = ""
    async for part in await AsyncClient().chat(
        model=model, messages=messages, stream=True, options=options
    ):
        full_response += part["message"]["content"]
        if len(full_response) % 25 == 0:  # Update every 25 characters
            yield full_response

    yield full_response


def create_interfaces_ollama():
    with gr.Group():
        gr.Markdown("# Ollama Chat")
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=[
                        "llama3.1:8b-text-q4_K_M",
                        "llama3.1:8b-instruct-q4_K_M",
                        "llava:7b-v1.6-vicuna-q4_K_M",
                    ],
                    value="llama3.1:8b-instruct-q4_K_M",
                )
                prompt = gr.Textbox(label="Prompt", lines=3)
                system = gr.Textbox(label="System Message (Optional)", lines=2)
                image = gr.Image(
                    label="Image (Optional, for multimodal models)", type="filepath"
                )
            with gr.Column():
                temperature = gr.Slider(
                    label="Temperature", minimum=0, maximum=2, value=0.7, step=0.1
                )
                top_k = gr.Slider(
                    label="Top K", minimum=1, maximum=100, value=40, step=1
                )
                top_p = gr.Slider(
                    label="Top P", minimum=0, maximum=1, value=0.9, step=0.05
                )
                num_predict = gr.Slider(
                    label="Number of Tokens to Predict",
                    minimum=1,
                    maximum=1000,
                    value=128,
                    step=1,
                )
                stream = gr.Checkbox(label="Stream Output", value=True)

        generate_button = gr.Button("Generate")
        output = gr.Textbox(label="Output", lines=10)

        generate_button.click(
            fn=ollama_chat,
            inputs=[
                model,
                prompt,
                system,
                image,
                temperature,
                top_k,
                top_p,
                num_predict,
                stream,
            ],
            outputs=output,
        )

    # return gr.Interface(
    #     fn=ollama_chat,
    #     inputs=[
    #         model,
    #         prompt,
    #         system,
    #         image,
    #         temperature,
    #         top_k,
    #         top_p,
    #         num_predict,
    #         stream,
    #     ],
    #     outputs=output,
    #     title="Ollama Chat",
    #     description="Chat with Ollama models",
    #     allow_flagging="never",
    # )


def create_interfaces():
    with gr.Group():
        gr.Markdown("## Anthropic Chat")
        anthropic_interface = gr.Interface(
            fn=anthropic,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Slider(1, 4096, value=1024, step=1, label="Max Tokens"),
                gr.Dropdown(
                    [
                        "claude-3-5-sonnet-20240620",
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                        "claude-2.1",
                        "claude-2.0",
                        "claude-instant-1.2",
                    ],
                    label="Model",
                    value="claude-3-5-sonnet-20240620",
                ),
            ],
            outputs=gr.Textbox(label="Response"),
        )

    with gr.Group():
        gr.Markdown("## GPT Chat")
        gpt_interface = gr.Interface(
            fn=gpt,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Number(value=-1, label="Seed"),
                gr.Dropdown(["gpt-4o-mini"], label="Model Name", value="gpt-4o-mini"),
            ],
            outputs=gr.Textbox(label="Response"),
        )

    with gr.Group():
        gr.Markdown("## Refine Prompt")
        refine_interface = gr.Interface(
            fn=refine,
            inputs=gr.Textbox(label="Prompt"),
            outputs=gr.Textbox(label="Refined Prompt"),
        )

    with gr.Group():
        gr.Markdown("## Ollama Chat")
        ollama_interface = create_interfaces_ollama()
