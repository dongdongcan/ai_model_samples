import gradio as gr
import os
from openai import OpenAI


def check_key():
    open_api_key = os.getenv("OPENAI_API_KEY")
    if open_api_key is None:
        print("Error: Environment variable 'OPENAI_API_KEY' is not set. Exiting the program.")
        return False
    return True


def generate_response(usr_input: str):
    if check_key() == False:
        return "You should set OPENAI_API_KEY first."
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": usr_input},
        ],
    )
    response = completion.choices[0].message.content
    return response


UI = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5, placeholder="Input your question here"),
    outputs="text",
    title="Assistant by openai API ",
    description="",
    examples=[
        ["introduce what is llm?"],
    ],
)
UI.launch()
