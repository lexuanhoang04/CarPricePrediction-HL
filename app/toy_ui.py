# toy_ui.py
import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000"  # your FastAPI server

def call_api(a, b):
    payload = {"a": float(a), "b": float(b)}

    r = requests.post(
        f"{API_URL}/add",
        json=payload,
        timeout=10
    )
    r.raise_for_status()

    data = r.json()
    return data["sum"], data   # result + raw json

with gr.Blocks() as demo:
    gr.Markdown("## Toy Gradio → FastAPI (/add)")

    with gr.Row():
        a_in = gr.Number(label="a", value=1)
        b_in = gr.Number(label="b", value=2)

    btn = gr.Button("Add")

    sum_out = gr.Number(label="sum")
    json_out = gr.JSON(label="raw response")

    btn.click(
        fn=call_api,
        inputs=[a_in, b_in],
        outputs=[sum_out, json_out]
    )

if __name__ == "__main__":
    demo.launch()
