import requests
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

API_URL = "https://lexuanhoang1904-carprice.hf.space"

app = FastAPI()

class AddRequest(BaseModel):
    a: float
    b: float

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/add")
def add(req: AddRequest):
    return {"a": req.a, "b": req.b, "sum": req.a + req.b}


# ---- Gradio UI ----

def send_request(a, b):
    r = requests.post(
        f"{API_URL}/add",
        json={"a": a, "b": b},
        timeout=10
    )
    return r.json()

with gr.Blocks() as demo:
    gr.Markdown("# ➕ Add API")

    a = gr.Number(label="a")
    b = gr.Number(label="b")

    btn = gr.Button("Send")
    out = gr.JSON()

    btn.click(send_request, [a, b], out)

app = gr.mount_gradio_app(app, demo, path="/")
