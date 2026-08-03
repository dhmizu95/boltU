# boltU — Building an LLM From Scratch on Colab and Kaggle

Adapted from `guideline.md`, restructured for Colab (notebook **and** Colab CLI) instead of a local NVIDIA GPU.

---

## 0. Read this before you start

The original guideline is a good skeleton, but three of its assumptions will sink you on Colab. Fix them up front.

| Guideline says | Problem | What to do instead |
|---|---|---|
| "split into 80/20 training/validation" | Standard for supervised ML, wrong for LM pretraining. You'd burn 20% of your token budget on a validation set 1000× larger than needed. | Hold out a **fixed 5–10M tokens** for validation (~0.5–2%). Everything else trains. |
| "500 MB to 1 GB" of text for a 100–200M param model | 1 GB of text ≈ 250M tokens. A 124M-param model wants ~2.5B tokens (Chinchilla: ~20 tokens/param). You'd be **10× under-trained** — the model will produce fluent-ish gibberish and you'll blame the code. | Either shrink the model to ~30M params, or get 10 GB+ of text. See §1. |
| "use mixed precision **and** gradient checkpointing" | Mixed precision: yes, always. Gradient checkpointing: costs 25–35% throughput. It's an OOM remedy, not a default. | Enable AMP always. Add gradient checkpointing **only** if you OOM, and try reducing micro-batch + raising grad-accum first. |

Two Colab-specific realities:

- **Sessions die.** Free tier: ~12h ceiling, idle disconnects, preemption. Checkpoint/resume isn't a nice-to-have — it's the core architectural requirement. Design for "process is killed at a random step" from day one.
- **`/content` is ephemeral, Drive is slow.** Never train reading directly from Drive and never write checkpoints straight to Drive mid-step. Stage data to local disk at session start; write checkpoints locally, then sync.

---

## 1. Phase 0 — Choose your scope (do this first, on paper)

Pick one row. This single decision determines everything downstream.

| Tier | Layers / d_model / heads | Ctx | Params | Token budget (20×) | Text needed | Realistic on |
|---|---|---|---|---|---|---|
| **A — Learning run** | 6 / 384 / 6 | 512 | ~30M | 600M | ~2.5 GB | T4 (free), overnight |
| **B — Middle** | 8 / 512 / 8 | 1024 | ~51M | 1.0B | ~4 GB | L4, ~1 day |
| **C — GPT-2 small clone** (guideline's target) | 12 / 768 / 12 | 1024 | ~124M | 2.5B | ~10 GB | A100/H100, or a multi-day L4 marathon |

Rough throughput to sanity-check your timeline (tokens/sec, 124M params, ctx 1024, AMP, no gradient checkpointing — **measure your own, these vary 2× by dataloader and kernel**):

| GPU | tok/s | Time for 2.5B tokens |
|---|---|---|
| T4 (fp16, no bf16) | ~8–15k | 2–4 days of pure compute |
| L4 | ~30–45k | 15–23 h |
| A100 40GB | ~120–180k | 4–6 h |
| H100 | ~250–350k | 2–3 h |

**Two things the table above assumes.** It is measured at 124M params / ctx 1024 — do not apply it to Tier A directly. Most of Tier A's speedup comes from the model being ~4× smaller (expect ~3–4× the tok/s, less than 4× because small models are more overhead-bound), *not* from the shorter context. Dropping ctx 1024→512 is worth only ~7% here, since attention is roughly 13% of FLOPs/token at this scale.

**And: free tier is T4-only.** A100 and H100 require a paid Colab plan, and even there allocation isn't guaranteed. If you pick Tier C on the free tier you are signing up for a multi-day T4 marathon. Check what you actually got with `torch.cuda.get_device_name()` before committing to a tier.

**Recommendation:** do Tier A end-to-end first. Get a working pipeline that trains, checkpoints, resumes, and generates text. *Then* scale the config to Tier C and re-run. Debugging a 3-day run is miserable; debugging a 3-hour run is fine. The code is identical — only a config file changes.

**Deliverable:** a filled-in `configs/base.yaml` with your chosen numbers.

---

## 2. Phase 1 — Repo scaffold (local machine)

The guideline builds a local project folder; that still applies, but the repo lives locally/on GitHub and **executes** on Colab. This is what makes the Colab CLI worth using — you keep a real repo with real files instead of copy-pasting into cells.

**Naming convention.** The project is branded **boltU**, but every *identifier* is lowercase `boltu`: the Python package, the Colab session name, and all Kaggle slugs. Two reasons this isn't fussiness. Kaggle lowercases dataset and kernel slugs regardless of what you type, so `boltU-tokens` silently becomes `boltu-tokens` and any hardcoded `/kaggle/input/boltU-tokens` path fails at runtime. And if you develop on macOS (case-insensitive filesystem) with a `boltU/` package, `import boltU` works locally and then dies on Colab's Linux box. Keep `boltU` in prose, READMEs, and the UI title; keep `boltu` everywhere a machine reads it.

```
boltU/
├── configs/
│   ├── base.yaml            # model + training hyperparams
│   └── tiny.yaml            # 2-min smoke-test config
├── data/                    # gitignored; token shards live here
├── src/
│   ├── data_prep.py         # download → tokenize → .bin shards
│   ├── dataset.py           # memmap loader, batch sampling
│   ├── model.py             # GPT implementation
│   ├── train.py             # loop, checkpointing, resume
│   ├── sample.py            # inference / generation
│   └── app.py               # Gradio UI
├── checkpoints/             # gitignored
├── requirements.txt
└── README.md
```

**Prompt to give Claude Code:**

> Scaffold a Python project for training a GPT-style language model from scratch. Create this directory structure: configs/, data/, src/, checkpoints/. Add a requirements.txt with torch, tiktoken, datasets, numpy, matplotlib, tqdm, gradio, pyyaml. Add a .gitignore excluding data/, checkpoints/, *.bin, and __pycache__. Create configs/base.yaml with a documented schema covering model dims (n_layer, n_head, n_embd, block_size, vocab_size, dropout) and training params (batch_size, grad_accum_steps, learning_rate, warmup_steps, max_steps, weight_decay, grad_clip, eval_interval, checkpoint_interval). Do not write model or training logic yet — structure and config only.

**Push it to GitHub before Phase 2.** This is the sync channel — Colab pulls the repo, you never
copy-paste code into cells. A private repo is fine; put the token in a Colab Secret, not inline.

```bash
gh repo create boltu --private --source=. --remote=origin --push
```

**Done when:** `tree` shows the structure, `configs/base.yaml` parses with `yaml.safe_load`, and
the repo is pushed.

---

## 3. Phase 2 — Colab environment (replaces guideline Step 1)

The guideline's Step 1 installs CUDA PyTorch locally. On Colab, PyTorch + CUDA are **pre-installed** — don't reinstall torch, you'll break the CUDA build and waste 10 minutes. Only add what's missing.

### Option A — Notebook

`Runtime → Change runtime type → GPU`, then in the first cell:

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/boltu/{data,checkpoints,logs}

%cd /content
![ -d boltU ] || git clone https://{userdata.get('GH_TOKEN')}@github.com/<you>/boltu.git boltU
%cd /content/boltU
!git pull --ff-only

!pip install -q tiktoken datasets gradio pyyaml
import torch, subprocess
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

Same cell at the top of every notebook. It's idempotent — clone on first run, `git pull` after.

### Option B — Colab CLI (recommended; keeps your repo as the source of truth)

Google released the official Colab CLI in June 2026 — it connects a local terminal to a remote
Colab runtime. Package `google-colab-cli`, binary `colab`, Python ≥3.12, **Linux/macOS only**.

**On Windows, this needs WSL2.** There is no native Windows build. One-time cost:

```powershell
wsl --install -d Ubuntu-24.04    # reboot if this is your first distro
```

Then, inside WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install google-colab-cli          # or: pip install google-colab-cli

# Auth is a prerequisite, not optional — the CLI rides on application-default credentials
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab --auth=adc whoami

colab new -s boltu --gpu L4            # or T4 / A100 / H100
colab install -s boltu -r requirements.txt
colab drivemount -s boltu              # mounts Drive at /content/drive
echo "import torch; print(torch.cuda.get_device_name(0))" | colab exec -s boltu
```

Keep the repo on the Windows side and reach it from WSL at `/mnt/d/...`, or clone it again inside
the WSL filesystem — the latter is meaningfully faster for git operations.

Core commands you'll use throughout:

| Command | Use |
|---|---|
| `colab new -s NAME --gpu L4` | provision a session |
| `colab exec -s NAME -f script.py` | run a **local** file remotely (contents are shipped; no upload step) |
| `colab upload/download -s NAME SRC DST` | move data and checkpoints |
| `colab drivemount -s NAME` | mount Drive for persistence |
| `colab log -s NAME -o run.ipynb` | export the session history as a notebook |
| `colab status` / `colab stop -s NAME` | monitor / release the VM |

The CLI has a built-in keep-alive daemon, which is exactly what you want for long training runs — no browser tab babysitting. It removes the ~90 min idle disconnect; it does **not** remove the ~12 h session ceiling, and the VM keep-alive itself caps at 24 h.

These invocations match the published command reference as of this writing, but the tool shipped in June 2026 and flags may shift — `colab <cmd> --help` is authoritative if something errors.

### Option B.1 — `exec` vs `run` vs `console` (read before Phase 5)

The three execution verbs are not interchangeable, and picking the wrong one is the most likely way to waste a session.

| Verb | Signature | Fits |
|---|---|---|
| `colab exec` | `exec [-s NAME] [-f FILE] [--output-image PATH]` | short, arg-free snippets against a live session |
| `colab run` | `run [--gpu GPU] [--keep] SCRIPT [ARGS...]` | one-shot: provision → run → release |
| `colab console` | `console [-s NAME]` — raw TTY (tmux) | **long training runs** |

Two consequences:

1. **`exec` takes no script arguments.** Look at the signatures: only `run` has `[ARGS...]`. So
   `colab exec -f src/train.py --config configs/base.yaml` does not do what it reads like — the
   flags are parsed by `colab`, not forwarded to your script. Either use `colab run` (which
   forwards, but provisions a *fresh* VM each time), or make `train.py` read its config from an
   env var / a fixed path, or drive it from `console`.
2. **Neither `exec` nor `run` is built for multi-hour jobs.** A third-party writeup reports a
   default 30-second execution timeout on both; that figure is *not* in the official command
   reference, so treat it as unverified and check `colab exec --help`. Either way, streaming a
   12-hour job through a synchronous RPC is the wrong shape. Use the TTY:

```bash
colab console -s boltu
# ...now inside the VM's tmux session:
cd /content/boltU && git pull --ff-only
nohup python src/train.py --config configs/base.yaml --resume > logs/train.out 2>&1 &
# Ctrl-b d to detach, then Ctrl-d to drop the local TTY. Training keeps running.
```

Poll it from outside without reattaching:

```bash
colab download -s boltu boltU/logs/metrics.csv /tmp/metrics.csv
```

which is exactly what the §11.7 dashboard consumes.

3. **Stop what you provision.** Compute units bill for as long as the VM is alive, and the
   keep-alive daemon is *designed* to keep it alive. `colab stop -s boltu` is not optional
   housekeeping.

**On precision:** prefer bf16 wherever it's available. Support starts at Ampere and continues in every later architecture — A100 (Ampere), L4 (Ada Lovelace), and H100 (Hopper) all have it. bf16 shares fp32's exponent range, so it needs no loss scaling and won't silently overflow. T4 (Turing, pre-Ampere) and Kaggle's P100 (Pascal) are the common free-tier GPUs *without* bf16, which is why the fp16 + `GradScaler` path in §5 exists at all.

**Checklist before moving on:**
- [ ] `torch.cuda.is_available()` is `True`
- [ ] You know your GPU name and VRAM (`nvidia-smi`)
- [ ] You know whether you have bf16 (Ampere+: A100/L4/H100 yes; **T4 no** — T4 is fp16-only, which needs a `GradScaler`)
- [ ] Drive is mounted and `/content/drive/MyDrive/boltu/` exists

---

## 4. Phase 3 — Data pipeline (guideline Step 2, corrected)

Tokenize **once**, save as flat `uint16` binary shards, then memory-map them. GPT-2's vocab is 50257 < 65536, so `uint16` halves your disk and I/O versus `int32`. Never re-tokenize at training time.

### Dataset options

| Dataset | Size | Notes |
|---|---|---|
| `HuggingFaceFW/fineweb-edu`, `sample-10BT` | slice to taste | Best quality/effort ratio. Stream it; take as many tokens as your tier needs. |
| `wikitext-103-raw-v1` | ~500 MB | Clean, small, fine for Tier A. |
| `openwebtext` | ~38 GB | The GPT-2 reproduction target; overkill unless Tier C. |
| Project Gutenberg | varies | Public domain, but archaic English — model will sound like 1890. |

**Prompt:**

> Write `src/data_prep.py`. It should stream a Hugging Face dataset (default: HuggingFaceFW/fineweb-edu, sample-10BT) up to a target token count from config. Tokenize with tiktoken's GPT-2 BPE encoding, appending the `<|endoftext|>` token after each document. Write tokens to flat uint16 `.bin` shards of 100M tokens each in `data/`, using multiprocessing over documents with a tqdm progress bar. Hold out the first shard as validation and cap it at 10M tokens — do NOT use an 80/20 split. Save a `data/meta.json` with total token count, vocab size, and shard filenames. At the end, print the total token count and decode a random 200-token window back to text so I can verify the round-trip is correct.

Then `src/dataset.py`:

> Write `src/dataset.py` with a `get_batch(split, batch_size, block_size, device)` function that memory-maps the uint16 shards with `np.memmap`, samples random offsets, builds x and y where y is x shifted by one position, and transfers to GPU with `pin_memory()` and `non_blocking=True`. Re-open the memmap each call to avoid a known memory-leak pattern. Include a `__main__` block that prints the shapes and dtypes of one batch.

**Cut the val holdout on document boundaries.** "First shard, capped at 10M tokens" is right in
spirit, but if you truncate mid-document the val set's final window shares a document with
`train_0001.bin` — a small, real leak that makes val loss read slightly optimistic. Stop the val
shard at the last complete document before 10M tokens and start training data at the next one.

### Persist to Drive

Tokenizing 2.5B tokens takes real time. Do it once, save the `.bin` files to Drive, and at the start of every future session copy them back to local disk (fast) rather than reading from Drive (slow):

```bash
# once
cp data/*.bin /content/drive/MyDrive/boltu/data/
# every session thereafter
cp /content/drive/MyDrive/boltu/data/*.bin /content/data/
```

**Done when:** `meta.json` reports your target token count, and the decoded sample window reads as coherent natural language (not mojibake, not `<|endoftext|>` spam).

---

## 5. Phase 4 — Model architecture (guideline Step 3)

**Prompt:**

> Write `src/model.py`: a decoder-only GPT in pure PyTorch, no Hugging Face modeling classes. Include token + learned positional embeddings, N transformer blocks each with pre-LayerNorm, causal multi-head self-attention using `F.scaled_dot_product_attention(..., is_causal=True)` so it uses Flash Attention kernels, an MLP with 4× expansion and GELU, and residual connections. Tie the token embedding and output head weights. Initialize weights with normal std 0.02, but scale residual-projection weights by `0.02/sqrt(2*n_layer)`. Read all dims from the YAML config. Add a `configure_optimizers` method returning AdamW with betas (0.9, 0.95), applying weight decay 0.1 only to 2D parameters (not biases or LayerNorms), and using the fused implementation when CUDA is available. Add a `generate` method with temperature, top-k, and top-p sampling. Add a `__main__` block that instantiates the model, prints a layer summary and total parameter count, and runs one forward pass on random input to verify output shape is (B, T, vocab_size).

**Done when:** parameter count is within ~5% of your tier's target and the forward pass returns the right shape. If you're targeting 124M and see 160M, you probably forgot weight tying.

---

## 6. Phase 5 — Training loop (guideline Step 4, hardened for Colab)

This is where the plan diverges most from the guideline. The loop must be **interruption-proof**.

**Prompt:**

> Write `src/train.py` implementing a training loop with these requirements:
>
> **Core:** gradient accumulation to reach an effective batch of ~0.5M tokens per optimizer step; AMP autocast using bfloat16 on Ampere+ and float16 with a GradScaler on older GPUs like the T4; gradient clipping at 1.0; cosine learning-rate decay to 10% of peak with linear warmup; `torch.compile` on the model with a flag to disable it.
>
> **Checkpointing (critical):** every N steps, save model state, optimizer state, scaler state, step number, RNG states, and config to `checkpoints/ckpt_latest.pt`. Write to a temp file and atomically rename so a mid-write crash can't corrupt it. Keep only the last 2 checkpoints plus the best-validation checkpoint. After each save, copy to Google Drive in a background thread so training doesn't block on Drive I/O.
>
> **Resume:** a `--resume` flag that loads the latest checkpoint and continues from the exact step with optimizer and scheduler state intact. This must be exact, not approximate.
>
> **Monitoring:** estimate validation loss over ~50 batches every eval_interval steps, using a **fixed RNG seed so the same 50 windows are scored at every eval** — otherwise val loss wiggles for reasons unrelated to training and the curve is unreadable; log step, train loss, val loss, learning rate, tokens/sec, and elapsed time to a CSV; tqdm progress bar; append-mode logging so resumed runs extend the same CSV.
>
> **Safety:** wrap the loop so SIGTERM/KeyboardInterrupt triggers a final checkpoint save before exit.

Starting hyperparameters (124M; scale LR up for smaller models):

```yaml
learning_rate: 6.0e-4      # peak; try 1e-3 for <50M params
min_lr_frac: 0.1           # cosine floor = 10% of peak
weight_decay: 0.1
grad_clip: 1.0
betas: [0.9, 0.95]
warmup_frac: 0.02          # 2% of max_steps — DERIVED, not hardcoded
```

**`tokens_per_step` is per-tier, not a constant.** nanoGPT's 0.5M is calibrated for 124M params. Applying it to Tier A gives 600M / 0.5M = **1,200 total steps**, which is far too few for a cosine schedule to do anything sensible — a fixed 700-step warmup would consume more than half the run.

| Tier | tokens_per_step | Total steps | Warmup @ 2% |
|---|---|---|---|
| A (30M, 600M tok) | 131072 (0.125M) | ~4,600 | ~90 |
| B (51M, 1.0B tok) | 262144 (0.25M) | ~3,800 | ~75 |
| C (124M, 2.5B tok) | 524288 (0.5M) | ~4,800 | ~95 |

Compute both from config at startup and log them:

```python
max_steps    = total_tokens // tokens_per_step
warmup_steps = max(50, int(warmup_frac * max_steps))
```

If `max_steps` comes out under ~2000, your `tokens_per_step` is too large for the tier.

### Run it

```bash
# Smoke test FIRST — 50 steps on the tiny config.
# `run` forwards ARGS and tears the VM down after; ideal for a short throwaway check.
colab run --gpu T4 src/train.py --config configs/tiny.yaml --max_steps 50

# Real run — detached inside the session's tmux, NOT through exec (see §3, Option B.1)
colab console -s boltu
#   cd /content/boltU && git pull --ff-only
#   nohup python src/train.py --config configs/base.yaml --resume > logs/train.out 2>&1 &
#   Ctrl-b d, Ctrl-d
```

Do not use `colab exec -f src/train.py --config ...` — `exec` has no `[ARGS...]` in its signature,
so those flags are eaten by the CLI and never reach your script.

Log `grad_norm` (pre-clip) at step 1 and step 50 of the smoke test. If it has exploded past ~5 by step 50, your LR or warmup is wrong — find that out in two minutes, not two days.

**Loss sanity checks.** With GPT-2's 50257-token vocab, initial loss should be ≈ `ln(50257)` ≈ **10.8**. If it isn't, your init or your targets are wrong. Then:

| Loss | Meaning |
|---|---|
| ~10.8 | step 0, correct |
| ~6.0 | learned unigram frequencies (fast, first few hundred steps) |
| ~4.5 | basic syntax, plausible word order |
| ~3.5 | coherent short passages |
| ~3.0 | ballpark of GPT-2 small on OpenWebText |
| flat or NaN | LR too high, or fp16 overflow — lower LR, verify GradScaler |

**Done when:** you can kill the process mid-run, relaunch with `--resume`, and see loss continue from where it stopped with no discontinuity in the curve. **Test this deliberately before the long run.** It is the single most important test in this project.

---

## 7. Phase 6 — The multi-session marathon

The guideline notes a usable model needs days of training. On Colab that means N sessions, not one. Per-session routine:

1. `colab new -s boltu --gpu L4` and `colab drivemount -s boltu`
2. `git pull` the repo on the VM; run `scripts/prune_drive.sh` (fail on quota now, not at hour six)
3. Copy data shards Drive → `/content/data/`
4. Copy `ckpt_latest.pt` Drive → `/content/checkpoints/`
5. `colab console -s boltu`, then `nohup python src/train.py --resume &`, detach
6. Let it run; the CLI keep-alive daemon handles the idle timeout (not the 12 h cap)
7. Confirm checkpoints are landing in Drive, then `colab stop -s boltu` — **the keep-alive daemon
   will otherwise hold the VM and bill compute units after training has finished**

Wrap steps 1–4 in a `scripts/session_start.sh` so restarting is one command — you'll do it a dozen times.

**Enforce the retention policy on Drive, not just locally.** §5 keeps 2 + best on the VM, but the VM is wiped every session while Drive accumulates forever. At ~1.5 GB per Tier C checkpoint (weights + optimizer state), orphaned files from interrupted syncs will silently eat the 15 GB free quota mid-week — and a full Drive makes every subsequent sync fail. Add `scripts/prune_drive.sh` that lists checkpoints in the Drive folder by mtime, keeps the newest 2 plus `ckpt_best.pt`, deletes the rest, and prints remaining free space. Run it from `session_start.sh` **before** launching training, so you fail on a quota problem at minute one rather than hour six.

Watch out for: Drive's 15 GB free quota (checkpoints are ~1.5 GB each for 124M with optimizer state — keep 2, not 20); free-tier GPU quota exhaustion after heavy use; and silent downgrade to a slower GPU on reconnect (log `torch.cuda.get_device_name()` every session so you can explain throughput changes).

---

## 8. Phase 7 — Evaluation and sampling

Loss alone won't tell you if the thing is usable.

**Prompt:**

> Write `src/sample.py`: load a checkpoint, accept a prompt string, and generate text with configurable temperature, top-k, top-p, and max tokens. Support streaming token-by-token to stdout. Add a `--benchmark` mode that computes validation perplexity over the full held-out set. Add a `--compare` mode that generates from the same prompt at temperatures 0.7, 0.9, and 1.2 side by side. Also write `src/plot_curves.py` that reads the training CSV and plots train and validation loss on a log-x axis, saving to `plots/loss.png`.

Judge output on: does it stay on topic for 50+ tokens, is the grammar consistent, does it loop or repeat, and does validation loss track train loss (divergence = overfitting; on a 20:1 token/param budget you should be nowhere near it).

---

## 9. Phase 8 — The interface (guideline Step 5, adapted)

The guideline's Tkinter/PyQt desktop GUI **cannot run in Colab** — no display server. Use Gradio, which renders inline in the notebook and gives you a shareable link.

**Prompt:**

> Write `src/app.py`: a Gradio interface for the trained model. Include a multiline prompt textbox, sliders for temperature (0.1–2.0), top-k (0–200), top-p (0.1–1.0), and max new tokens, plus a model-checkpoint dropdown that scans the checkpoints directory. Stream generated tokens to the output box as they're produced using a generator function. Show tokens/sec and total time under the output. Load the model once at startup, not per request. Launch with `share=True` and `server_name="0.0.0.0"` so it works from a Colab runtime.

```python
!python src/app.py    # in a notebook cell — the UI renders inline
```

**Optional local desktop GUI:** if you want the guideline's original Tkinter app, download the checkpoint (`colab download -s boltu checkpoints/ckpt_best.pt ./`) and run it locally. A 124M model runs fine on CPU for inference — a few tokens/sec, adequate for testing.

---

## 10. Phase 9 — Optional: make it follow instructions

A base model completes text; it doesn't answer questions. If you want chat-like behavior, add a fine-tune stage: format a small instruction dataset into a consistent prompt template, mask the loss so it's computed only on response tokens, and train 2–3 epochs. This is a separate script (`src/finetune.py`) that loads the pretrained checkpoint — do not mix it into `train.py`.

Scale reference: Alpaca is ~52k examples (~10M tokens), Dolly-15k ~3M tokens, OASST1 ~20M. Against Tier C's 2.5B-token pretraining budget this is under 1% — SFT reshapes output *format*, it does not add knowledge. At 10–20% of a 6e-4 peak LR you're at roughly **6e-5 to 1.2e-4**, with a short warmup and no restart of the cosine schedule.

Two things to expect. Loss numbers are not comparable to pretraining (you're scoring only response tokens over a much narrower distribution, so it will look lower and means nothing across stages). And SFT alone, without preference data, commonly costs you some base capability — the alignment tax. Keep the pretrained checkpoint and compare generations side by side; if the tuned model is more polite but noticeably dumber, that's the tax, and the fix is fewer epochs or a lower LR, not more data.

---

## 11. Phase 5b — Robustness and telemetry (do before any long run)

> **Ordering note.** Sections 1–10 follow the original guideline's build order. §11 and §12 are documented last but **executed earlier**: §11 belongs between Phase 5 and Phase 6, and §12 replaces the Phase 2 / Phase 6 platform layer if you train on Kaggle. The master checklist is in execution order; follow that, not the section numbers.

### 11.1 Data loading: prefer RAM over memmap

At 2 bytes/token, token budgets are smaller than they look:

| Tier | Tokens | On-disk | Fits in free-tier RAM (~12.7 GB)? |
|---|---|---|---|
| A | 600M | 1.2 GB | Easily |
| B | 1.0B | 2.0 GB | Easily |
| C | 2.5B | 5.0 GB | Yes, with headroom |
| beyond | 5B+ | 10 GB+ | No — memmap required |

**Default path:** `np.fromfile(shard, dtype=np.uint16)` once at startup, hold in RAM, slice from it. No page-cache thrashing, no read amplification, no IOPS pressure. Memmap is the fallback for corpora above ~4B tokens, not the default.

If you do need memmap: re-open it inside `get_batch()` each call (holding a long-lived memmap object leaks), and log `psutil.Process().memory_info().rss` alongside `/proc/meminfo` `Cached` so you can distinguish real RSS growth from reclaimable page cache. Only the former is dangerous.

### 11.2 No DataLoader, no workers

**Batch construction detail.** Slicing `arr[i:i+block_size]` from a 1-D contiguous buffer already returns a contiguous view — no `.contiguous()` needed. But token IDs are `uint16` on disk and `nn.Embedding` requires int64 indices, so a cast is unavoidable. Do it **once on the stacked batch**, not per window:

```python
ix = np.random.randint(0, len(arr) - block_size, size=micro_batch)
x = np.stack([arr[i   : i+block_size]   for i in ix]).astype(np.int64)
y = np.stack([arr[i+1 : i+1+block_size] for i in ix]).astype(np.int64)
```

If profiling later shows batch prep is non-trivial, the win is preallocating pinned buffers and copying into them — not adding `.contiguous()` calls.

`get_batch()` samples random offsets from an in-RAM array (or memmap) and calls `.pin_memory()` on the resulting tensor — that is the *tensor method*, unrelated to `DataLoader(pin_memory=True)`. There is no `num_workers` in this design and adding one is not possible without restructuring.

Verify this is fine rather than assuming: log `t_data / t_step`. If data prep exceeds ~2% of step time, revisit. On 2-core Colab instances, worker processes typically cost more in context switching than they recover.

Only if you later stream a corpus too large for local disk does an `IterableDataset` become necessary; then start at `num_workers=2, prefetch_factor=2, persistent_workers=True` and measure.

### 11.3 Checkpoint sync must fail loudly

The background Drive copy is the highest-risk silent failure in the whole pipeline: mount disconnects and API rate limits kill the thread invisibly, and you discover it only when the VM dies with nothing synced.

**Prompt:**

> Rewrite the checkpoint-sync logic in `src/train.py` to use a single-worker `ThreadPoolExecutor`. Store the `Future` from each submitted sync. Before submitting the next sync, call `.result(timeout=0)` on the previous future inside a try/except so any exception raised in the worker surfaces on the main thread. After each successful copy, verify the Drive-side file by comparing its size to the local file and confirming its mtime advanced. Maintain a consecutive-failure counter: log a warning on the first failure, and on the second consecutive failure raise and halt training rather than continuing to produce checkpoints that never leave the VM. Also write a `heartbeat.json` to Drive every eval interval containing step number, wall-clock timestamp, and latest val loss, so a dead session is detectable from outside.

**Do not rely on atomic rename for the heartbeat.** The Drive FUSE mount is not POSIX-compliant and `os.replace` gives no atomicity guarantee there — which is why checkpoints are verified by size and mtime *after* the copy rather than trusted on rename. Make the heartbeat robust by not depending on its contents: **file mtime is the liveness signal**, and it's readable without parsing anything. Wrap the body read in try/except and fall back to the last successful parse. Append-only JSONL is a fine alternative, since a torn final line is simply discarded.

### 11.4 Derive grad accumulation, never hardcode it

```python
tokens_per_micro = micro_batch * block_size
assert target_tokens_per_step % tokens_per_micro == 0, (
    f"{target_tokens_per_step} not divisible by {tokens_per_micro}; "
    f"adjust micro_batch or target"
)
grad_accum_steps = target_tokens_per_step // tokens_per_micro
```

Recompute on every launch, including after an OOM forces `micro_batch` down. Log the resulting effective batch size at startup and assert it matches the config target — a silent drift from 0.5M to 0.43M tokens/step distorts the LR schedule you tuned for.

### 11.5 `src/telemetry.py`

**Prompt:**

> Write `src/telemetry.py` exposing a `Telemetry` class that a training loop calls once per step, writing a row to a CSV and printing a compact summary every N steps. Track:
>
> **Throughput:** tokens/sec, and Model FLOPs Utilization computed as `flops_per_token = 6*n_params + 12*n_layer*n_head*head_dim*block_size`, divided by a peak-dense-FLOPS lookup table keyed on `torch.cuda.get_device_name()` (T4 65e12 fp16, P100 19e12 fp16, L4 121e12, A100 312e12, H100 990e12) with a conservative fallback for unknown devices.
>
> Use the **dense** figures above. Marketing numbers roughly double them (H100 is quoted at 1979 TFLOPS) by assuming 2:4 structured sparsity, which dense pretraining does not use — benchmark against the sparse number and your MFU will look half as good as it is.
>
> **Step-time distribution:** rolling p50, p90, and p99 over the last 200 steps, plus separate timers for data fetch, forward+backward, and optimizer step. Report data-fetch time as a percentage of total.
>
> **Memory and fragmentation:** `torch.cuda.memory_allocated`, `max_memory_allocated`, `memory_reserved`, the reserved-minus-allocated gap, and critically `torch.cuda.memory_stats()["num_alloc_retries"]` and `["num_ooms"]`. Warn whenever `num_alloc_retries` increases, since that is the direct fragmentation signal.
>
> **Hardware:** a daemon thread sampling pynvml every 10 seconds for GPU utilization percent, memory used, temperature, power draw, and `nvmlDeviceGetCurrentClocksThrottleReasons` decoded into readable flags. Log a warning when thermal or power throttling is active.
>
> **Host:** process RSS and available system memory, to catch dataloader-side growth.
>
> All GPU-side reads must be cheap — do not call `torch.cuda.synchronize()` per step purely for timing; use CUDA events instead.

Set this in the environment before training starts:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It largely eliminates fragmentation for fixed-shape training loops and is the first remedy if `num_alloc_retries` climbs.

### 11.6 What to watch on a Tier C run

| Signal | Healthy | Action if not |
|---|---|---|
| MFU | 35–45% (A100), 15–20% (T4) | Stable is what matters; a mid-run drop means throttling or a GPU downgrade |
| `num_alloc_retries` | 0, flat | Climbing → enable `expandable_segments`, then lower micro-batch |
| p99 / p50 step time | < 1.5× | Larger gap → Drive sync blocking, or page faults from memmap |
| data fetch % of step | < 2% | Higher → move to in-RAM array |
| throttle reasons | none / `GpuIdle` | `SwPowerCap` or `HwThermalSlowdown` → expect reduced throughput, not a bug |
| heartbeat mtime in Drive | advancing | Stalled → session died; restart and resume |

### 11.7 Watching a run from outside the session

**Principle: training never blocks on visualization.** Same rule as the Drive sync. The dashboard is a reader of artifacts, never a participant in the loop.

**Preferred mechanism — CLI poll + terminal dashboard.** The metrics CSV is kilobytes, so pull it on a timer and render locally:

```bash
# scripts/watch.sh — run in a tmux pane on your machine
while true; do
  colab download -s boltu logs/metrics.csv /tmp/metrics.csv 2>/dev/null
  colab download -s boltu logs/heartbeat.json /tmp/heartbeat.json 2>/dev/null
  python scripts/dashboard.py /tmp/metrics.csv /tmp/heartbeat.json
  sleep 30
done
```

**Prompt:**

> Write `scripts/dashboard.py`, a read-only terminal dashboard using plotext and rich. It takes a metrics CSV and a heartbeat JSON path. **Drop the final CSV row if its field count doesn't match the header** — downloads taken mid-append produce a torn last line, and this is the most likely corruption you'll hit. Render: train and validation loss against step on a log-x axis; MFU on the same time axis as loss so throughput events can be visually attributed; step-time p50 and p99; and a status line with current step, tokens seen versus budget, ETA at current throughput, `num_alloc_retries`, active throttle reasons, and seconds since the heartbeat mtime advanced. Colour the staleness figure red past 3× the eval interval. The script must never write anything and must exit cleanly on a missing or malformed file.

Plotting MFU against the same axis as loss is the point of the whole exercise — it's what separates "the model stopped learning" from "the GPU got throttled at hour nine."

**Alternative — Gradio with `share=True`.** Launch a second Gradio app in a background thread inside `train.py` reading the same CSV. Gradio is already a dependency from Phase 8, and the public URL means you can check a multi-day run from your phone. Costs a thread and a tunnel that occasionally drops.

**Skip TensorBoard and W&B here.** TensorBoard means dual-logging to event files when CSV is already the source of truth, and it's awkward outside a notebook. W&B is the lowest-effort option if an external service is acceptable, but it puts a network call inside your training process — precisely what §11.3 exists to keep out.

## 12. Kaggle track (alternative / complementary to Colab)

### 12.1 Which platform for which job

| | Colab free | Kaggle free |
|---|---|---|
| GPU | T4 | **T4 ×2** or P100 |
| Weekly quota | opaque, varies | **~30 h, stated and floating**; resets Sat 00:00 UTC |
| Session cap | ~12 h, idle disconnects | 12 h hard (9 h TPU); interactive idles out at ~90 min |
| Concurrency | 1 | **1 interactive + 2 commit jobs** |
| Persistence | Drive, 15 GB | **Kaggle Datasets, 200 GB private** |
| Execution model | interactive (`colab exec`) | **batch** (`kernels push`) |
| Internet | on | **off by default — must be enabled per notebook** |

**Recommended split: develop on Colab, train on Kaggle.** Interactive execution makes Colab far better for Phases 1–5 where you're iterating in minutes. Kaggle's storage quota, stated hours, and concurrent commit jobs make it better for the Phase 6 marathon. The code is identical; only the launcher and the persistence layer change.

Prefer **T4 ×2 over P100**. P100 has more memory bandwidth but no tensor cores, so its fp16 peak (~19 TFLOPS) is far below T4's (~65 TFLOPS) for AMP training. Neither has bf16 — the fp16 + `GradScaler` path from §5 applies on both.

### 12.2 Setup

```bash
pip install kaggle
# Account → Settings → API → Create New Token, then:
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

The token file is the path that reliably works everywhere, including from *inside* a kernel. Newer CLI versions may also offer a browser login — check `kaggle --help`. Inside a notebook, never paste the token: use **Kaggle Secrets** (Add-ons → Secrets) and reconstruct `~/.kaggle/kaggle.json` at runtime.

### 12.3 The batch execution model

There is no `exec` equivalent. You push a folder containing a script/notebook plus a `kernel-metadata.json`:

```json
{
  "id": "<username>/boltu-train",
  "title": "boltu-train",
  "code_file": "run.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<username>/boltu-tokens", "<username>/boltu-checkpoints"]
}
```

```bash
kaggle kernels push -p ./kernel                    # submit
kaggle kernels status <username>/boltu-train         # poll
kaggle kernels output <username>/boltu-train -p ./out  # retrieve after completion
```

`enable_internet: true` is required if the kernel downloads from Hugging Face or pushes checkpoints back via the API. It's off by default and the failure is a confusing DNS error, not a clear message.

### 12.4 Persistence via Datasets, not Drive

Attached datasets mount read-only at `/kaggle/input/<name>` and read fast — much better than Drive. Work at `/kaggle/working` (this becomes the kernel output).

- **Token shards:** upload once as a private dataset, attach to every run. At 200 GB you are not constrained even at Tier C.
- **Checkpoints:** version a private dataset at the end of each session.

```bash
kaggle datasets version -p /kaggle/working/ckpt -m "step 24000" --dir-mode zip
```

Next run attaches it and resumes from `/kaggle/input/boltu-checkpoints/ckpt_latest.pt`. The §7 Drive-pruning script is unnecessary here — keep the version history.

### 12.5 Self-terminate before the platform does (critical)

A batch job killed at the 12 h cap **never runs your save path**. Colab's SIGTERM handler does not save you here. Budget wall-clock time explicitly:

**Prompt:**

> Add a `--max-hours` argument to `src/train.py` (default 11.0 for Kaggle's 12 h cap). Record start time at launch. After each optimizer step, check elapsed time; once it exceeds the budget, break the loop, save a final checkpoint, push it as a new Kaggle dataset version if running on Kaggle, and exit 0 cleanly. Estimate remaining time per step and break *early* if the next checkpoint interval would overrun the budget — never start work you cannot persist. Log the exit reason explicitly as one of: max_steps reached, time budget reached, or error.

Leave a full hour of margin. Dataset versioning of a multi-GB checkpoint takes real minutes, and it is the last thing you want racing a hard kill.

### 12.6 Optional: 2×T4 with DDP

Two T4s give ~1.7–1.8× throughput, not 2×. Worth it only once single-GPU training is fully working. The one correctness trap: **`grad_accum_steps` must be divided by `world_size`**, or your effective batch silently doubles and the §11.4 assertion is the only thing that will catch it.

### 12.7 The monitoring gap

`kernels output` only returns artifacts after the job completes, so the §11.7 live dashboard doesn't work as written. Options, in order of preference:

1. **Push a small metrics dataset from inside the kernel** every 30–60 minutes (not every eval — versioning is rate-limited). Poll it locally with `kaggle datasets download`. The CSV plus heartbeat is kilobytes.
2. **Accept batch blindness.** With `--max-hours` self-termination and reliable resume, a 12 h run that reports at the end is tolerable. This is the honest default.
3. Run the interactive session concurrently for spot checks — you have the concurrency budget for it.

---

## Master checklist

- [ ] **P0** Tier chosen, token budget calculated, `base.yaml` filled in
- [ ] **P1** Repo scaffolded, config parses, **pushed to GitHub** (the Colab sync channel)
- [ ] **P2** WSL2 up (Windows only); `colab` installed **and authed** (`colab --auth=adc whoami`); GPU verified; Drive mounted; bf16 support known; `console`-not-`exec` understood for long runs
- [ ] **P3** Data tokenized to uint16 shards, val holdout cut on document boundaries, round-trip decode verified, shards backed up to Drive
- [ ] **P4** Model builds, param count matches target, forward pass shape correct
- [ ] **P5** Smoke test passes; initial loss ≈ 10.8; **kill-and-resume verified**
- [ ] **P5b** Data in RAM (or memmap justified); sync failures halt training; grad_accum asserted; `telemetry.py` logging MFU and `num_alloc_retries`; `expandable_segments` set
- [ ] **K** *(Kaggle track, §12 — replaces P2/P6 platform steps)* `--max-hours` self-termination verified; token shards uploaded as a private dataset; checkpoint dataset versioning tested end-to-end; `enable_internet: true` confirmed
- [ ] **P6** `session_start.sh` written; first long session completed
- [ ] **P7** Loss curves plotted; samples read as coherent English
- [ ] **P8** Gradio UI streaming from the trained checkpoint
- [ ] **P9** *(optional)* instruction fine-tune

---

## Fast reference: failure modes

| Symptom | Likely cause |
|---|---|
| Loss stuck near 10.8 | LR is zero (warmup misconfigured) or labels aren't shifted by one |
| Loss → NaN | fp16 overflow on T4 (missing/misused GradScaler), or LR too high |
| CUDA OOM | reduce micro-batch and raise grad_accum to compensate; enable gradient checkpointing only as a last resort |
| Throughput drops between sessions | you got a different GPU — check `get_device_name()` |
| Training pauses periodically | you're checkpointing to Drive synchronously; move it to a background thread |
| Generation repeats one phrase | undertrained, or temperature too low / top-k too small |
| Resumed run's loss spikes | optimizer or scheduler state not restored — only model weights were loaded |
| Script ignores your `--config` / runs the wrong tier | you launched via `colab exec -f`, which forwards no argv — use `colab run` or `console` (§3, Option B.1) |
| Compute units draining with nothing training | keep-alive daemon holding an idle VM — `colab stop -s boltu` |
| **Loss looks fine (~3.2) but generations are mush** | see below |

**Diagnosing "good loss, bad output."** This is the most common late-stage confusion, and it's usually not the model. Work through it in this order: (1) does the same prompt with the same seed produce *identical* output? If not, you have a sampler bug. (2) Is temperature ≤0.5 or top-k ≤10? That produces repetitive loops regardless of model quality — try 0.9 / top-k 50. (3) Does the prompt end mid-word or with trailing whitespace? Tokenization boundary artifacts derail generation badly. (4) Is your inference-time tokenizer the *same* encoding used in `data_prep.py`? A mismatch gives fluent-looking nonsense. (5) Only after all four: check tokens seen against the 20:1 budget — at 3.2 loss on a 10× under-trained model, mush is simply the correct output.
