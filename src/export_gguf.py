"""Export a boltU checkpoint to GGUF (F16, "gpt2" architecture) for llama.cpp.

Layout note: our attn/mlp projections are nn.Linear ((out_features, in_features), row-major),
which is already the layout GGUF/ggml expects for these tensor roles. HF's GPT-2 stores the
same weights transposed in a Conv1D layer ((in_features, out_features)), so *HF* converters
transpose before writing — ours must NOT be transposed again, or the exported model would
silently produce wrong output despite loading fine.

Tokenizer: pulled from the canonical `transformers` "gpt2" tokenizer (same BPE vocab/merges as
our tiktoken "gpt2" encoding) rather than hand-reconstructed from tiktoken's rank dict, since
recovering the exact merge order from tiktoken's internal representation is a nontrivial,
error-prone reverse-engineering problem — a bad tokenizer export would look fine on load and
degrade output quality invisibly, which defeats "without any loss".
"""
import argparse
import json
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import GPT2Tokenizer

GPT2_EOT_ID = 50256


def read_merges(base_path):
    # newer transformers only writes the unified tokenizer.json (no standalone merges.txt);
    # its merges are [left, right] pairs, not the "left right" strings GGUF wants.
    data = json.loads((base_path / "tokenizer.json").read_text(encoding="utf-8"))
    return [f"{left} {right}" for left, right in data["model"]["merges"]]


def add_tokenizer(writer):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(base_path)
        vocab = gguf.BpeVocab(base_path)
        tokens, scores, toktypes = zip(*vocab.all_tokens())
        merges = read_merges(base_path)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt2")
    writer.add_token_list(list(tokens))
    writer.add_token_scores(list(scores))
    writer.add_token_types(list(toktypes))
    writer.add_token_merges(merges)
    writer.add_bos_token_id(GPT2_EOT_ID)
    writer.add_eos_token_id(GPT2_EOT_ID)


def add_tensors(writer, sd, cfg, np_dtype):
    def t(name):
        return sd[name].float().numpy().astype(np_dtype)

    writer.add_tensor("token_embd.weight", t("transformer.wte.weight"))
    writer.add_tensor("position_embd.weight", t("transformer.wpe.weight"))
    writer.add_tensor("output_norm.weight", t("transformer.ln_f.weight"))
    writer.add_tensor("output_norm.bias", t("transformer.ln_f.bias"))
    writer.add_tensor("output.weight", t("lm_head.weight"))  # tied to token_embd

    for i in range(cfg["n_layer"]):
        p = f"transformer.h.{i}."
        writer.add_tensor(f"blk.{i}.attn_norm.weight", t(p + "ln_1.weight"))
        writer.add_tensor(f"blk.{i}.attn_norm.bias", t(p + "ln_1.bias"))
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", t(p + "attn.c_attn.weight"))
        writer.add_tensor(f"blk.{i}.attn_qkv.bias", t(p + "attn.c_attn.bias"))
        writer.add_tensor(f"blk.{i}.attn_output.weight", t(p + "attn.c_proj.weight"))
        writer.add_tensor(f"blk.{i}.attn_output.bias", t(p + "attn.c_proj.bias"))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", t(p + "ln_2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias", t(p + "ln_2.bias"))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", t(p + "mlp.c_fc.weight"))
        writer.add_tensor(f"blk.{i}.ffn_up.bias", t(p + "mlp.c_fc.bias"))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", t(p + "mlp.c_proj.weight"))
        writer.add_tensor(f"blk.{i}.ffn_down.bias", t(p + "mlp.c_proj.bias"))


DTYPES = {
    "f16": (np.float16, gguf.LlamaFileType.MOSTLY_F16),
    "f32": (np.float32, gguf.LlamaFileType.ALL_F32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16",
                     help="f16 (default, GGUF's standard unquantized size) or f32 (bit-exact)")
    args = ap.parse_args()
    np_dtype, file_type = DTYPES[args.dtype]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    writer = gguf.GGUFWriter(args.out, "gpt2")
    writer.add_name(Path(args.checkpoint).stem)
    writer.add_context_length(cfg["block_size"])
    writer.add_embedding_length(cfg["n_embd"])
    writer.add_block_count(cfg["n_layer"])
    writer.add_feed_forward_length(4 * cfg["n_embd"])
    writer.add_head_count(cfg["n_head"])
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(file_type)

    add_tokenizer(writer)
    add_tensors(writer, ckpt["model"], cfg, np_dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
