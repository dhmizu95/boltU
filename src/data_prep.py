"""Stream a HF dataset, tokenize with GPT-2 BPE, write flat uint16 .bin shards.

Val holdout is cut on a document boundary (see plans/boltu-plan.md §4), not an 80/20 split:
whole documents accumulate into val_0000.bin until the next document would push it over
val_tokens; that overflowing document is dropped (not written to either split) and training
shards start from the document after it. This guarantees zero leakage between splits.
"""
import argparse
import json
import multiprocessing as mp
import os
import random

import numpy as np
import tiktoken
import yaml
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    tokens = enc.encode_ordinary(doc["text"])
    tokens.append(EOT)
    arr = np.array(tokens, dtype=np.uint32)
    assert (arr < 2**16).all(), "token id out of uint16 range"
    return arr.astype(np.uint16)


def write_shard(path, tokens):
    tmp = path + ".tmp"
    tokens.tofile(tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from datasets import load_dataset

    ds = load_dataset(
        cfg["dataset_name"], name=cfg.get("dataset_subset"), split="train", streaming=True
    )

    os.makedirs(args.out_dir, exist_ok=True)
    val_cap = cfg["val_tokens"]
    shard_size = cfg["shard_tokens"]
    total_target = cfg["total_tokens"]

    nprocs = max(1, (os.cpu_count() or 2) - 1)
    shard_filenames = []
    total_written = 0
    raw_text_samples = []  # for the round-trip assert, collected as we go

    with mp.Pool(nprocs) as pool:
        doc_iter = pool.imap(tokenize, ds, chunksize=16)

        # --- val shard: whole documents up to val_cap, cut on a doc boundary ---
        val_buf = np.empty(val_cap, dtype=np.uint16)
        val_pos = 0
        pbar = tqdm(total=total_target, unit="tok", desc="tokenizing")
        for tokens in doc_iter:
            if len(raw_text_samples) < 20:
                raw_text_samples.append(tokens)
            if val_pos + len(tokens) > val_cap:
                break  # this doc dropped; train starts at the *next* doc
            val_buf[val_pos : val_pos + len(tokens)] = tokens
            val_pos += len(tokens)
        val_path = os.path.join(args.out_dir, "val_0000.bin")
        write_shard(val_path, val_buf[:val_pos])
        shard_filenames.append(os.path.basename(val_path))
        total_written += val_pos
        pbar.update(val_pos)

        # --- train shards: normal fixed-size shards, documents may span shard boundaries ---
        shard_idx = 1
        buf = np.empty(shard_size, dtype=np.uint16)
        buf_pos = 0
        for tokens in doc_iter:
            if total_written >= total_target:
                break
            i = 0
            while i < len(tokens):
                space = shard_size - buf_pos
                take = min(space, len(tokens) - i)
                buf[buf_pos : buf_pos + take] = tokens[i : i + take]
                buf_pos += take
                i += take
                total_written += take
                pbar.update(take)
                if buf_pos == shard_size:
                    path = os.path.join(args.out_dir, f"train_{shard_idx:04d}.bin")
                    write_shard(path, buf)
                    shard_filenames.append(os.path.basename(path))
                    shard_idx += 1
                    buf_pos = 0
            if total_written >= total_target:
                break
        if buf_pos > 0:
            path = os.path.join(args.out_dir, f"train_{shard_idx:04d}.bin")
            write_shard(path, buf[:buf_pos])
            shard_filenames.append(os.path.basename(path))
        pbar.close()

    meta = {
        "total_tokens": int(total_written),
        "val_tokens": int(val_pos),
        "vocab_size": enc.n_vocab,
        "shard_filenames": shard_filenames,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- round-trip assert: decode(encode(s)) == s on a random 500-char raw sample ---
    all_raw_text = "".join(
        enc.decode(t.tolist()) for t in raw_text_samples
    )
    if len(all_raw_text) > 500:
        start = random.randint(0, len(all_raw_text) - 500)
        sample = all_raw_text[start : start + 500]
        assert enc.decode(enc.encode(sample, allowed_special="all")) == sample, (
            "tokenizer round-trip failed"
        )

    print(f"total tokens written: {total_written:,} (val: {val_pos:,})")

    # decode a random 200-token window from the first train shard as a sanity check
    train_shards = [s for s in shard_filenames if s.startswith("train_")]
    if train_shards:
        arr = np.fromfile(os.path.join(args.out_dir, train_shards[0]), dtype=np.uint16)
        if len(arr) > 200:
            start = random.randint(0, len(arr) - 200)
            window = arr[start : start + 200].tolist()
            print("--- sample decoded window ---")
            print(enc.decode(window))


if __name__ == "__main__":
    main()
