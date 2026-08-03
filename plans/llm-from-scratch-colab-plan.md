# Building an LLM From Scratch on Google Colab — Full Execution Plan

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

**Recommendation:** do Tier A end-to-end first. Get a working pipeline that trains, checkpoints, resumes, and generates text. *Then* scale the config to Tier C and re-run. Debugging a 3-day run is miserable; debugging a 3-hour run is fine. The code is identical — only a config file changes.

**Deliverable:** a filled-in `configs/base.yaml` with your chosen numbers.

---

## 2. Phase 1 — Repo scaffold (local machine)

The guideline builds a local project folder; that still applies, but the repo lives locally/on GitHub and **executes** on Colab. This is what makes the Colab CLI worth using — you keep a real repo with real files instead of copy-pasting into cells.

```
llm-from-scratch/
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

**Done when:** `tree` shows the structure and `configs/base.yaml` parses with `yaml.safe_load`.

---

## 3. Phase 2 — Colab environment (replaces guideline Step 1)

The guideline's Step 1 installs CUDA PyTorch locally. On Colab, PyTorch + CUDA are **pre-installed** — don't reinstall torch, you'll break the CUDA build and waste 10 minutes. Only add what's missing.

### Option A — Notebook

`Runtime → Change runtime type → GPU`, then in the first cell:

```python
!pip install -q tiktoken datasets gradio pyyaml
import torch, subprocess
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

### Option B — Colab CLI (recommended; keeps your repo as the source of truth)

Google released the official Colab CLI in June 2026 — it connects a local terminal to a remote Colab runtime. Linux/macOS only.

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli

colab new -s llm --gpu L4            # or T4 / A100 / H100
colab install -s llm -r requirements.txt
colab drivemount -s llm              # mounts Drive at /content/drive
echo "import torch; print(torch.cuda.get_device_name(0))" | colab exec -s llm
```

Core commands you'll use throughout:

| Command | Use |
|---|---|
| `colab new -s NAME --gpu L4` | provision a session |
| `colab exec -s NAME -f script.py` | run a **local** file remotely (contents are shipped; no upload step) |
| `colab upload/download -s NAME SRC DST` | move data and checkpoints |
| `colab drivemount -s NAME` | mount Drive for persistence |
| `colab log -s NAME -o run.ipynb` | export the session history as a notebook |
| `colab status` / `colab stop -s NAME` | monitor / release the VM |

The CLI has a built-in keep-alive daemon, which is exactly what you want for long training runs — no browser tab babysitting.

**Checklist before moving on:**
- [ ] `torch.cuda.is_available()` is `True`
- [ ] You know your GPU name and VRAM (`nvidia-smi`)
- [ ] You know whether you have bf16 (Ampere+: A100/L4/H100 yes; **T4 no** — T4 is fp16-only, which needs a `GradScaler`)
- [ ] Drive is mounted and `/content/drive/MyDrive/llm-project/` exists

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

### Persist to Drive

Tokenizing 2.5B tokens takes real time. Do it once, save the `.bin` files to Drive, and at the start of every future session copy them back to local disk (fast) rather than reading from Drive (slow):

```bash
# once
cp data/*.bin /content/drive/MyDrive/llm-project/data/
# every session thereafter
cp /content/drive/MyDrive/llm-project/data/*.bin /content/data/
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
> **Monitoring:** estimate validation loss over ~50 batches every eval_interval steps; log step, train loss, val loss, learning rate, tokens/sec, and elapsed time to a CSV; tqdm progress bar; append-mode logging so resumed runs extend the same CSV.
>
> **Safety:** wrap the loop so SIGTERM/KeyboardInterrupt triggers a final checkpoint save before exit.

Starting hyperparameters (124M; scale LR up for smaller models):

```yaml
learning_rate: 6.0e-4      # peak; try 1e-3 for <50M params
min_lr: 6.0e-5
warmup_steps: 700
weight_decay: 0.1
grad_clip: 1.0
betas: [0.9, 0.95]
tokens_per_step: 524288    # 0.5M — set micro_batch × grad_accum to hit this
```

### Run it

```bash
# Smoke test FIRST — 50 steps on the tiny config
colab exec -s llm -f src/train.py --config configs/tiny.yaml --max_steps 50

# Real run
colab exec -s llm -f src/train.py --config configs/base.yaml
```

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

1. `colab new -s llm --gpu L4` and `colab drivemount -s llm`
2. Copy data shards Drive → `/content/data/`
3. Copy `ckpt_latest.pt` Drive → `/content/checkpoints/`
4. `colab exec -s llm -f src/train.py --resume`
5. Let it run; the CLI keep-alive daemon handles idle timeouts
6. Confirm checkpoints are landing in Drive, then `colab stop -s llm`

Wrap steps 1–4 in a `scripts/session_start.sh` so restarting is one command — you'll do it a dozen times.

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

**Optional local desktop GUI:** if you want the guideline's original Tkinter app, download the checkpoint (`colab download -s llm checkpoints/ckpt_best.pt ./`) and run it locally. A 124M model runs fine on CPU for inference — a few tokens/sec, adequate for testing.

---

## 10. Phase 9 — Optional: make it follow instructions

A base model completes text; it doesn't answer questions. If you want chat-like behavior, add a fine-tune stage: format a small instruction dataset (Alpaca, Dolly, or OASST) into a consistent prompt template, mask the loss so it's computed only on response tokens, and train 2–3 epochs at 10–20% of your pretraining LR. This is a separate script (`src/finetune.py`) that loads the pretrained checkpoint — do not mix it into `train.py`.

---

## Master checklist

- [ ] **P0** Tier chosen, token budget calculated, `base.yaml` filled in
- [ ] **P1** Repo scaffolded, config parses
- [ ] **P2** Colab CLI installed, GPU verified, Drive mounted, bf16 support known
- [ ] **P3** Data tokenized to uint16 shards, round-trip decode verified, shards backed up to Drive
- [ ] **P4** Model builds, param count matches target, forward pass shape correct
- [ ] **P5** Smoke test passes; initial loss ≈ 10.8; **kill-and-resume verified**
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
