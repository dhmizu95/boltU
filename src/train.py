"""Training loop hardened for Kaggle: interruption-proof checkpointing, exact resume,
wall-clock self-termination. See plans/boltu-plan.md §6 and §11.

DDP (2xT4) is deferred — Tier A/B fit single-GPU; add when scaling to the Tier B/C marathon
(plan §11.7): divide grad_accum_steps by world_size, wrap model in DDP, gate rank-0-only I/O.
telemetry.py (MFU, throttle reasons, pynvml) is likewise deferred — plan scopes it Tier B/C only.
"""
import argparse
import csv
import math
import os
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_batch
from model import GPT

CKPT_DIR = "checkpoints"
LOG_CSV = "checkpoints/metrics.csv"
HEARTBEAT_PATH = "checkpoints/heartbeat.json"
KAGGLE_INPUT_CKPT = "/kaggle/input/boltu-checkpoints/ckpt_latest.pt"

_should_stop = False


def _handle_sigterm(signum, frame):
    global _should_stop
    _should_stop = True


signal.signal(signal.SIGTERM, _handle_sigterm)


def get_lr(step, max_steps, warmup_steps, peak_lr, min_lr_frac):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return peak_lr * min_lr_frac
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * min_lr_frac + coeff * (peak_lr - peak_lr * min_lr_frac)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def find_resume_path():
    if os.path.exists(KAGGLE_INPUT_CKPT):
        return KAGGLE_INPUT_CKPT
    local = os.path.join(CKPT_DIR, "ckpt_latest.pt")
    if os.path.exists(local):
        return local
    return None


def save_checkpoint(model, optimizer, scaler, step, best_val_loss, cfg, use_scaler, tag="latest"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if use_scaler else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": cfg,
        "rng": rng_state(),
    }
    atomic_save(ckpt, os.path.join(CKPT_DIR, f"ckpt_{tag}.pt"))
    if tag == "latest":
        numbered = os.path.join(CKPT_DIR, f"ckpt_step_{step:07d}.pt")
        atomic_save(ckpt, numbered)
        _prune_numbered_checkpoints(keep=2)


def _prune_numbered_checkpoints(keep):
    files = sorted(
        f for f in os.listdir(CKPT_DIR) if f.startswith("ckpt_step_") and f.endswith(".pt")
    )
    for f in files[:-keep]:
        os.remove(os.path.join(CKPT_DIR, f))


def push_checkpoint_dataset(dataset_slug, message, retries=2):
    """Best-effort: retry with backoff, log on failure, never raise. See plan §11.3."""
    if not dataset_slug:
        return
    for attempt in range(retries + 1):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", CKPT_DIR, "-m", message, "--dir-mode", "zip"],
                check=True, capture_output=True, text=True, timeout=600,
            )
            print(f"[checkpoint] pushed dataset version: {message}")
            return
        except Exception as e:
            wait = 2 ** attempt
            print(f"ERROR: dataset version push failed (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(wait)
    print("ERROR: all dataset version push attempts failed; checkpoint left on local disk only",
          file=sys.stderr)


def write_heartbeat(step, val_loss):
    tmp = HEARTBEAT_PATH + ".tmp"
    import json
    with open(tmp, "w") as f:
        json.dump({"step": step, "timestamp": time.time(), "val_loss": val_loss}, f)
    os.replace(tmp, HEARTBEAT_PATH)


def append_csv_row(row, fieldnames):
    new_file = not os.path.exists(LOG_CSV)
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def estimate_loss(model, cfg, device, autocast_ctx, eval_iters, eval_seed):
    model.eval()
    saved_np_state = np.random.get_state()
    np.random.seed(eval_seed)
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split, cfg["micro_batch"], cfg["block_size"], device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    np.random.set_state(saved_np_state)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--checkpoint-dataset", default=None, help="kaggle dataset slug to push checkpoints to")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_hours = args.max_hours if args.max_hours is not None else cfg["max_hours"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # --- derive step counts from token budget (§6/§11.4), never hardcode ---
    tokens_per_micro = cfg["micro_batch"] * cfg["block_size"]
    assert cfg["tokens_per_step"] % tokens_per_micro == 0, (
        f"tokens_per_step {cfg['tokens_per_step']} not divisible by micro_batch*block_size "
        f"{tokens_per_micro}; adjust micro_batch or tokens_per_step"
    )
    grad_accum_steps = cfg["tokens_per_step"] // tokens_per_micro
    effective_batch_tokens = grad_accum_steps * tokens_per_micro
    assert effective_batch_tokens == cfg["tokens_per_step"]
    print(f"grad_accum_steps={grad_accum_steps}, effective batch = {effective_batch_tokens:,} tokens/step")

    max_steps = args.max_steps or (cfg["total_tokens"] // cfg["tokens_per_step"])
    warmup_steps = max(50, int(cfg["warmup_frac"] * max_steps))
    print(f"max_steps={max_steps}, warmup_steps={warmup_steps}")

    # --- precision: bf16 on Ampere+, else fp16 + GradScaler (T4/P100 have no bf16) ---
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype, use_scaler = torch.bfloat16, False
    elif device.type == "cuda":
        amp_dtype, use_scaler = torch.float16, True
    else:
        amp_dtype, use_scaler = torch.float32, False
    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda"))
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler)
    print(f"amp dtype: {amp_dtype}, grad scaler: {use_scaler}")

    raw_model = GPT(cfg).to(device)
    optimizer = raw_model.configure_optimizers(
        cfg["weight_decay"], cfg["learning_rate"], tuple(cfg["betas"]), device.type
    )

    step = 0
    best_val_loss = float("inf")
    if args.resume:
        path = find_resume_path()
        assert path, "no checkpoint found to resume from"
        print(f"resuming from {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)  # trusted, self-produced checkpoint; contains RNG/optimizer state, not just weights
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        restore_rng_state(ckpt["rng"])
        step = ckpt["step"] + 1  # checkpoint stores the last *completed* step; continue after it
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed at step {step}, best_val_loss {best_val_loss:.4f}")

    compile_enabled = cfg.get("compile", True) and not args.no_compile
    model = torch.compile(raw_model) if compile_enabled else raw_model

    fieldnames = ["step", "train_loss", "val_loss", "lr", "tokens_per_sec", "elapsed_sec", "grad_norm"]
    start_time = time.time()
    budget_seconds = max_hours * 3600
    exit_reason = "error"

    try:
        pbar_range = range(step, max_steps)
        step_times = []
        for step in pbar_range:
            if _should_stop:
                exit_reason = "signal received"
                break

            step_start = time.time()
            lr = get_lr(step, max_steps, warmup_steps, cfg["learning_rate"], cfg["min_lr_frac"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(grad_accum_steps):
                x, y = get_batch("train", cfg["micro_batch"], cfg["block_size"], device)
                with autocast_ctx:
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps
                loss_accum += loss.item()
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.time() - step_start
            step_times.append(step_time)
            if len(step_times) > 200:
                step_times.pop(0)
            tokens_per_sec = effective_batch_tokens / step_time
            elapsed = time.time() - start_time

            val_loss = None
            if step % cfg["eval_interval"] == 0 or step == max_steps - 1:
                losses = estimate_loss(model, cfg, device, autocast_ctx, cfg["eval_iters"], cfg["seed"])
                val_loss = losses["val"]
                print(f"step {step}: train {losses['train']:.4f} val {losses['val']:.4f} lr {lr:.2e}")
                write_heartbeat(step, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(raw_model, optimizer, scaler, step, best_val_loss, cfg, use_scaler, tag="best")

            append_csv_row({
                "step": step, "train_loss": loss_accum, "val_loss": val_loss,
                "lr": lr, "tokens_per_sec": tokens_per_sec, "elapsed_sec": elapsed,
                "grad_norm": grad_norm.item(),
            }, fieldnames)

            if step % cfg["checkpoint_interval"] == 0 and step > 0:
                save_checkpoint(raw_model, optimizer, scaler, step, best_val_loss, cfg, use_scaler, tag="latest")

            # never start work we can't persist: bail early if the *next* checkpoint interval would overrun
            avg_step_time = sum(step_times) / len(step_times)
            projected = avg_step_time * cfg["checkpoint_interval"]
            if elapsed + projected > budget_seconds:
                exit_reason = "time budget reached"
                break
        else:
            exit_reason = "max_steps reached"

    except KeyboardInterrupt:
        exit_reason = "signal received"
    except Exception:
        save_checkpoint(raw_model, optimizer, scaler, step, best_val_loss, cfg, use_scaler, tag="latest")
        raise

    save_checkpoint(raw_model, optimizer, scaler, step, best_val_loss, cfg, use_scaler, tag="latest")
    push_checkpoint_dataset(args.checkpoint_dataset, f"step {step}, exit: {exit_reason}")
    print(f"exit reason: {exit_reason}")
    sys.exit(0)


if __name__ == "__main__":
    main()
