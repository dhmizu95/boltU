"""In-RAM (default) or memmap (large corpora) batch sampler over the uint16 shards.

No DataLoader, no worker processes — see plans/boltu-plan.md §11.2. get_batch() samples
random offsets directly from an in-RAM array and pins the resulting tensor.
"""
import json
import os

import numpy as np
import torch

DATA_DIR = "data"
MEMMAP_THRESHOLD_TOKENS = 4_000_000_000  # ~8GB at uint16; see plans/boltu-plan.md §11.1

_meta_cache = {}
_split_cache = {}


def _load_meta(data_dir):
    if data_dir not in _meta_cache:
        with open(os.path.join(data_dir, "meta.json")) as f:
            _meta_cache[data_dir] = json.load(f)
    return _meta_cache[data_dir]


def _shard_paths(split, data_dir):
    meta = _load_meta(data_dir)
    prefix = "val_" if split == "val" else "train_"
    return [os.path.join(data_dir, f) for f in meta["shard_filenames"] if f.startswith(prefix)]


def _get_split(split, data_dir):
    key = (split, data_dir)
    if key not in _split_cache:
        paths = _shard_paths(split, data_dir)
        total_tokens = sum(os.path.getsize(p) // 2 for p in paths)
        if total_tokens > MEMMAP_THRESHOLD_TOKENS:
            _split_cache[key] = ("memmap", paths)
        else:
            _split_cache[key] = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths])
    return _split_cache[key]


def _sample_ram(arr, batch_size, block_size):
    ix = np.random.randint(0, len(arr) - block_size, size=batch_size)
    x = np.stack([arr[i : i + block_size] for i in ix])
    y = np.stack([arr[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


def _sample_memmap(paths, batch_size, block_size):
    # Re-open a fresh memmap per call (not cached) — a long-lived memmap object leaks. Each row
    # picks a shard weighted by its token count so sampling stays uniform over tokens overall.
    sizes = np.array([os.path.getsize(p) // 2 for p in paths], dtype=np.float64)
    shard_choice = np.random.choice(len(paths), size=batch_size, p=sizes / sizes.sum())
    x = np.empty((batch_size, block_size), dtype=np.uint16)
    y = np.empty((batch_size, block_size), dtype=np.uint16)
    for row, shard_i in enumerate(shard_choice):
        arr = np.memmap(paths[shard_i], dtype=np.uint16, mode="r")
        i = np.random.randint(0, len(arr) - block_size)
        x[row] = arr[i : i + block_size]
        y[row] = arr[i + 1 : i + 1 + block_size]
    return x, y


def get_batch(split, batch_size, block_size, device, data_dir=DATA_DIR):
    data = _get_split(split, data_dir)
    if isinstance(data, tuple):
        x, y = _sample_memmap(data[1], batch_size, block_size)
    else:
        x, y = _sample_ram(data, batch_size, block_size)

    # cast once on the stacked batch, not per window (§11.2) — uint16 on disk, int64 for nn.Embedding
    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = get_batch("train", batch_size=8, block_size=64, device=device)
    print("x:", x.shape, x.dtype, x.device)
    print("y:", y.shape, y.dtype, y.device)
