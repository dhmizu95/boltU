"""Load a checkpoint and generate text. Also: --benchmark (val perplexity) and
--compare (same prompt at a few temperatures)."""
import argparse
import math
import os
import sys

import numpy as np
import tiktoken
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT

enc = tiktoken.get_encoding("gpt2")
EOT_ID = 50256


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def decode_stream(token_ids):
    """Yield UTF-8-safe decoded text chunks for a stream of token ids. A single BPE token doesn't
    always align to a full UTF-8 character (multi-byte characters can split across tokens), so
    decoding tokens one at a time can emit invalid byte sequences -- shows up as U+FFFD ("<27>")."""
    buffer = b""
    for tok in token_ids:
        buffer += enc.decode_single_token_bytes(tok)
        try:
            yield buffer.decode("utf-8")
            buffer = b""
        except UnicodeDecodeError:
            continue
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


def generate(model, prompt, max_new_tokens, temperature, top_k, top_p, device, stream=False):
    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    gen = model.generate(idx, max_new_tokens, temperature, top_k, top_p)
    out_ids = []

    def token_stream():
        for next_id, _ in gen:
            tok = next_id.item()
            out_ids.append(tok)
            yield tok

    for chunk in decode_stream(token_stream()):
        if stream:
            print(chunk, end="", flush=True)
    if stream:
        print()
    return prompt + enc.decode(out_ids)


def benchmark(model, cfg, device, data_dir="data"):
    """Validation perplexity over the FULL held-out set (not a sampled subset)."""
    arr = np.fromfile(os.path.join(data_dir, "val_0000.bin"), dtype=np.uint16).astype(np.int64)
    block_size = cfg["block_size"]
    n_windows = (len(arr) - 1) // block_size
    total_loss = 0.0
    with torch.no_grad():
        for i in tqdm(range(n_windows), desc="benchmark"):
            s = i * block_size
            x = torch.tensor(arr[s : s + block_size], device=device).unsqueeze(0)
            y = torch.tensor(arr[s + 1 : s + 1 + block_size], device=device).unsqueeze(0)
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / n_windows
    print(f"val loss: {avg_loss:.4f}  perplexity: {math.exp(avg_loss):.2f}  ({n_windows} windows)")


def compare(model, prompt, max_new_tokens, top_k, top_p, device):
    for temp in (0.7, 0.9, 1.2):
        text = generate(model, prompt, max_new_tokens, temp, top_k, top_p, device)
        print(f"\n--- temperature {temp} ---\n{text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ckpt_best.pt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    if args.benchmark:
        benchmark(model, cfg, device, args.data_dir)
    elif args.compare:
        compare(model, args.prompt, args.max_new_tokens, args.top_k, args.top_p, device)
    else:
        text = generate(
            model, args.prompt, args.max_new_tokens, args.temperature,
            args.top_k, args.top_p, device, stream=args.stream,
        )
        if not args.stream:
            print(text)


if __name__ == "__main__":
    main()
