"""ChatGPT-style Streamlit UI for testing a boltU checkpoint. Streams tokens as generated.

Run: streamlit run src/streamlit_app.py

boltU is a base pretraining model, not instruction-tuned (see plan §10) — there's no real chat
template. Turns are framed as a plain "User: ...\\nAssistant: ..." transcript, which nudges a
well-trained base model toward turn-taking but isn't guaranteed, especially on a small model.
"""
import glob
import os
import sys

import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sample import EOT_ID, decode_stream, enc, load_model

st.set_page_config(page_title="boltU chat", page_icon="\U0001f4ac")


def scan_checkpoints(ckpt_dir="checkpoints"):
    return sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))


@st.cache_resource(show_spinner="Loading model...")
def get_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt_path, device)
    return model, cfg, device


def build_prompt(messages):
    lines = [("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines)


def stream_reply(model, device, prompt, max_new_tokens, temperature, top_k):
    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    gen = model.generate(idx, max_new_tokens, temperature, top_k, None)

    def token_stream():
        for next_id, _ in gen:
            tok = next_id.item()
            if tok == EOT_ID:
                return
            yield tok

    yield from decode_stream(token_stream())


st.title("boltU chat")

checkpoints = scan_checkpoints()
if not checkpoints:
    st.error("No checkpoints found in checkpoints/. Train a model first.")
    st.stop()

if len(checkpoints) > 1:
    checkpoint = st.sidebar.selectbox("Model", checkpoints, index=len(checkpoints) - 1)
else:
    checkpoint = checkpoints[0]
    st.sidebar.caption(f"Model: {checkpoint}")

temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.9)
top_k = st.sidebar.slider("Top-k", 0, 200, 50)
max_new_tokens = st.sidebar.slider("Max new tokens", 10, 500, 150)
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            # plain text, not markdown -- the model's raw hyphens/numbers would otherwise get
            # reinterpreted as list syntax, misrepresenting what it actually generated
            st.text(m["content"])
        else:
            st.markdown(m["content"])

if user_input := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    model, cfg, device = get_model(checkpoint)
    prompt = build_prompt(st.session_state.messages)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        for chunk in stream_reply(model, device, prompt, max_new_tokens, temperature, top_k or None):
            reply += chunk
            placeholder.text(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
