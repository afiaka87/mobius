import gradio as gr
from services import claude, chatgpt


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
