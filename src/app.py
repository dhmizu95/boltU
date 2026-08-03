"""Gradio demo for a trained checkpoint. Streams tokens as they're generated.
`share=True` needs enable_internet: true in kernel-metadata.json; the tunnel dies with the
session — this is a demo path, not a deployment (plans/boltu-plan.md §9)."""
import glob
import os
import sys
import time

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sample import decode_stream, enc, load_model

_model_cache = {}  # checkpoint path -> (model, config); loaded lazily, kept warm


def scan_checkpoints(ckpt_dir="checkpoints"):
    return sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))


def get_model(ckpt_path, device):
    if ckpt_path not in _model_cache:
        _model_cache[ckpt_path] = load_model(ckpt_path, device)
    return _model_cache[ckpt_path]


def stream_generate(prompt, checkpoint, temperature, top_k, top_p, max_new_tokens):
    if not checkpoint:
        yield "", "no checkpoint found in checkpoints/"
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = get_model(checkpoint, device)
    top_k = int(top_k) if top_k > 0 else None
    top_p = top_p if top_p < 1.0 else None

    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    gen = model.generate(idx, int(max_new_tokens), temperature, top_k, top_p)

    n = 0

    def token_stream():
        nonlocal n
        for next_id, _ in gen:
            n += 1
            yield next_id.item()

    out_text = ""
    start = time.time()
    for chunk in decode_stream(token_stream()):
        out_text += chunk
        elapsed = time.time() - start
        yield out_text, f"{n / elapsed:.1f} tok/s, {elapsed:.1f}s total"


def build_demo():
    checkpoints = scan_checkpoints()
    with gr.Blocks(title="boltU") as demo:
        gr.Markdown("# boltU")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4)
                checkpoint = gr.Dropdown(
                    choices=checkpoints, value=checkpoints[0] if checkpoints else None,
                    label="Checkpoint",
                )
                temperature = gr.Slider(0.1, 2.0, value=0.9, label="Temperature")
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = disabled)")
                top_p = gr.Slider(0.1, 1.0, value=1.0, label="Top-p (1.0 = disabled)")
                max_new_tokens = gr.Slider(1, 500, value=200, step=1, label="Max new tokens")
                btn = gr.Button("Generate")
            with gr.Column():
                output = gr.Textbox(label="Output", lines=14)
                stats = gr.Markdown()
        btn.click(
            stream_generate,
            inputs=[prompt, checkpoint, temperature, top_k, top_p, max_new_tokens],
            outputs=[output, stats],
        )
    return demo


if __name__ == "__main__":
    build_demo().queue().launch(share=True, server_name="0.0.0.0")
