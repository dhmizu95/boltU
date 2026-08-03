# Building a GPT-Style LLM From Scratch — Colab Execution Plan

Derived from [guideline.md](guideline.md), re-targeted at **Google Colab** as the training
environment (browser notebooks primary, `colab` CLI optional).

---

## 0. Ground Truth & Key Decisions

Facts established before planning (do not re-derive):

| Item | Value |
|---|---|
| Local machine | Windows 11 Pro, Python 3.13.12, **no NVIDIA GPU** (`nvidia-smi` absent) |
| Project dir | `d:\Workspace\Projects\LLM` — currently only `guideline.md`, not a git repo |
| Training compute | Google Colab (T4 free / L4 / A100 paid) |
| `google-colab-cli` | binary `colab`, Python ≥3.12, **Linux + macOS only — needs WSL2 on Windows** |

### Decisions (locked in, with rationale)

1. **Local machine is for authoring only.** No CUDA install locally — it would be dead weight.
   Step 1 of the guideline collapses into "scaffold a repo + a CPU-only smoke-test venv".
2. **Code lives in a git repo, not in notebook cells.** The guideline's Colab section suggests
   pasting code into cells. Don't. Notebook cells are unversioned, undiffable, and you *will*
   lose a session. Instead: real `.py` modules in git, and thin notebooks that
   `git pull && python -m llmscratch.train`. This is the single most important deviation.
3. **Model: GPT-2 small clone** — 12 layers, 768 dim, 12 heads, ctx 1024, vocab 50257
   → **~124M params**. Matches the guideline's "100–200M".
4. **Tokenizer: GPT-2 BPE via `tiktoken`**, pre-tokenized once into flat `uint16` memmap `.bin`
   shards. Never tokenize inside the training loop — Colab sessions are too short to waste
   minutes on re-tokenizing.
5. **Checkpoint/resume is a first-class feature, not an afterthought.** Free Colab: ~12h max
   session, ~90min idle kill, GPU not guaranteed. The training script must be able to die at any
   step and resume bit-for-bit from Drive.
6. **UI: Gradio, not Tkinter.** Guideline Step 5 asks for a desktop GUI; Colab can't launch one.
   Gradio runs inline in the notebook *and* runs locally on Windows later. One codebase, both.

### Reality check on scale (state this up front, then proceed)

A 124M-param model is Chinchilla-optimal at ~2.5B tokens. The guideline's "500MB–1GB of text"
is ~125–250M tokens — **10–20× undertrained**. Rough T4 throughput for this model is
~1B tokens per **~12–15 hours**. So:

| Budget | Tokens | Result |
|---|---|---|
| 1 session (~10h T4) | ~0.7B | Coherent English grammar, no facts. Good demo. |
| ~3 sessions (~30h) | ~2B | Chinchilla-optimal. Noticeably better. Recommended target. |
| A100 (~$10 of units) | ~2B in ~4h | Same result, far less babysitting. |

Plan the data pipeline for **2B tokens** even if you stop early — undersized data is the one
mistake you can't fix without redoing everything downstream.

---

## Phase 1 — Local Scaffold (Windows, ~30 min)

**Goal:** a git repo Colab can clone, plus a CPU venv that can run tests and the Gradio UI.

### 1.1 Repo + venv

```powershell
cd d:\Workspace\Projects\LLM
git init -b main
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tiktoken datasets numpy matplotlib tqdm pyyaml gradio pytest
pip freeze > requirements-dev.txt
```

`requirements.txt` (the *Colab* one — no torch, Colab ships it with CUDA):

```
tiktoken
datasets
numpy
matplotlib
tqdm
pyyaml
gradio
huggingface_hub
```

### 1.2 Target structure

```
LLM/
├── configs/
│   ├── gpt124m_t4.yaml          # fp16, grad-accum tuned for 15GB
│   └── gpt124m_a100.yaml        # bf16, larger micro-batch
├── src/llmscratch/
│   ├── __init__.py
│   ├── config.py                # dataclass <- yaml, single source of truth
│   ├── prepare_data.py          # HF dataset -> uint16 .bin shards
│   ├── data.py                  # memmap batch sampler
│   ├── model.py                 # GPT: embeddings, MHA, MLP, LN, weight tying
│   ├── train.py                 # loop, AMP, ckpt/resume, logging  <- the CLI entrypoint
│   ├── sample.py                # temperature / top-k / top-p generation
│   └── ui.py                    # Gradio app, streams tokens
├── colab/
│   ├── 01_prepare_data.ipynb
│   ├── 02_train.ipynb
│   └── 03_inference_ui.ipynb
├── tests/test_smoke.py          # runs on CPU, tiny config
├── scripts/colab_bootstrap.sh
├── requirements.txt
└── .gitignore                   # .venv/ data/ checkpoints/ *.bin *.pt
```

### 1.3 Push to GitHub (private)

```powershell
gh repo create llm-from-scratch --private --source=. --remote=origin --push
```

This is the sync channel. Colab pulls; you never copy-paste code.

### 1.4 Claude Code prompt for this phase

> Scaffold a Python project for training a GPT-style LLM from scratch, targeting Google Colab
> for training and this Windows machine for authoring only — do NOT install CUDA locally.
> Create the directory layout in PLAN.md §1.2, a `.gitignore` excluding venv/data/checkpoints,
> a `requirements.txt` for Colab (no torch — Colab provides it), and `src/llmscratch/config.py`
> with a `@dataclass TrainConfig` loaded from YAML covering: model dims, context length,
> batch/grad-accum, LR schedule, AMP dtype, checkpoint interval, and data/checkpoint paths.
> Add `tests/test_smoke.py` placeholders. Everything must import cleanly on CPU-only torch.

**Exit criteria:** `pytest -q` passes (trivially), repo pushed, `python -c "import llmscratch"` works.

---

## Phase 2 — Colab Access Path (~20 min, one-time)

Two tracks. **Do Track A first**; add Track B only if you want terminal-driven runs.

### Track A — Browser Colab (primary, works today)

1. New notebook → **Runtime ▸ Change runtime type ▸ T4 GPU** (or L4/A100 if subscribed).
2. Standard bootstrap cell, identical in all three notebooks:

```python
!nvidia-smi
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/llm-scratch/{data,checkpoints,logs}

%cd /content
![ -d repo ] || git clone https://<TOKEN>@github.com/<you>/llm-from-scratch.git repo
%cd /content/repo
!git pull --ff-only
!pip install -q -r requirements.txt
```

Put the clone URL + token in a Colab **Secret** (`🔑` panel) rather than inline.

3. **Anti-idle:** free tier kills after ~90 min idle. Don't rely on JS console hacks — instead
   make training print to stdout every ~30s (the loop does this via tqdm), which counts as
   output activity, and accept that a hard 12h cap will still hit. Resume handles it.

### Track B — `colab` CLI via WSL2 (optional)

The CLI is Linux/macOS only, so:

```powershell
wsl --install -d Ubuntu-24.04       # reboot if first time
```

Inside WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install google-colab-cli

# auth via application-default credentials
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab --auth=adc whoami
```

Command surface you'll actually use:

| Command | Use |
|---|---|
| `colab new -s trainer --gpu T4` | provision a VM (T4/L4/G4/A100/H100) |
| `colab install -r requirements.txt` | deps via `uv` |
| `colab drivemount` | mount Drive on the VM |
| `colab upload LOCAL REMOTE` / `colab download REMOTE LOCAL` | file transfer |
| `colab exec -f script.py` | run a local file remotely |
| `colab console -s trainer` | raw tmux TTY — **this is how you run long training** |
| `colab log -s trainer -o run.ipynb` | export session history as an artifact |
| `colab sessions` / `colab status` / `colab stop -s trainer` | lifecycle |

⚠️ **`exec`/`run` default to a 30-second timeout** — useless for a multi-hour training job.
For training use `colab console` + `tmux` + `nohup python -m llmscratch.train ... &`, then
detach. Poll with `colab exec -s trainer -f scripts/tail_log.py`.

⚠️ VM keep-alive caps at 24h; compute units bill while the VM lives. `colab stop` when done.

### 2.1 Claude Code prompt for this phase

> Write `scripts/colab_bootstrap.sh` (idempotent: mount Drive, clone-or-pull the repo, pip
> install requirements, create Drive dirs for data/checkpoints/logs, print GPU name + VRAM and
> assert CUDA is available). Also write `colab/01_prepare_data.ipynb`, `02_train.ipynb`, and
> `03_inference_ui.ipynb`, each starting with a cell that runs that bootstrap script. Keep the
> notebooks thin — they orchestrate `python -m llmscratch.*`, they don't contain logic.

**Exit criteria:** a Colab cell prints `Tesla T4, 15360 MiB` and the repo is cloned at `/content/repo`.

---

## Phase 3 — Data Pipeline (Guideline Step 2) (~1–2 h, mostly waiting)

**Goal:** `train.bin` + `val.bin` on Drive, ~2B tokens, `uint16`, never regenerated.

### 3.1 Dataset choice

Prefer **`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed** — cleaner than raw
Wikipedia and far better per-token quality for a small model. Alternatives if you want to stay
literal to the guideline: `wikimedia/wikipedia` (`20231101.en`) or Project Gutenberg dumps.

Stream, don't download — 10BT as files is far larger than Colab's disk.

### 3.2 Format

nanoGPT-style flat binary:

- Tokenize with `tiktoken.get_encoding("gpt2")`, append `<|endoftext|>` (50256) between docs.
- Write `uint16` (vocab 50257 fits). 2B tokens = **4 GB** — check your Drive quota now.
- Shard at ~200M tokens/file (`train_0000.bin` …) so a session crash costs one shard, not all.
- Emit `meta.json`: token counts per shard, tokenizer name, dataset revision.
- **Val split: hold out whole documents, not a random 20% of tokens.** The guideline's "80/20"
  applied token-wise leaks context across the boundary. Use ~10M held-out tokens; a 20% val
  split of 2B tokens is 400M tokens of pure waste.

### 3.3 Loader

`data.py`: `np.memmap` per shard, sample random offsets, return `(x, y)` where `y` is `x` shifted
by one. Pin memory, `non_blocking=True` transfer. No `DataLoader`/workers needed — memmap random
reads are fast and worker processes are a common Colab OOM source.

### 3.4 Claude Code prompt

> Write `src/llmscratch/prepare_data.py`: stream `HuggingFaceFW/fineweb-edu` (`sample-10BT`) via
> `datasets`, tokenize with `tiktoken` GPT-2 BPE using a multiprocessing pool, and write flat
> `uint16` `.bin` shards of ~200M tokens each to a target dir, plus `meta.json`. Hold out whole
> documents for validation until ~10M val tokens. Make it **resumable**: on restart, skip shards
> that already exist and are complete. Show a tqdm bar with tokens/sec and an ETA.
> Then write `src/llmscratch/data.py` with `get_batch(split, batch_size, block_size, device)`
> reading via `np.memmap`, and a `--verify` CLI that decodes and prints 3 random samples plus
> total token counts so I can eyeball that tokenization is sane.

**Exit criteria:** `python -m llmscratch.data --verify` prints readable English and
`train ≈ 2.0B / val ≈ 10M` tokens. `.bin` files on Drive.

⚠️ Copy shards from Drive to `/content/data/` at the start of each training session —
Drive FUSE random-read is slow enough to bottleneck the GPU. Local disk is ~100GB, fine for 4GB.

---

## Phase 4 — Model (Guideline Step 3) (~1 h)

### 4.1 Architecture

Standard GPT-2 decoder, pre-LN:

- Token embedding (50257×768) + learned positional embedding (1024×768)
- 12 × block: `LN → causal MHA(12 heads) → residual → LN → MLP(4×, GELU) → residual`
- Final LN → LM head, **weights tied** to token embedding (saves 38M params)
- Init: normal(0, 0.02); residual projections scaled by `1/√(2·n_layer)`

Param count: ~124M. Print a per-module breakdown so the number is verifiable, not asserted.

### 4.2 Attention implementation

Use `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`. On T4 (sm_75) this
picks the memory-efficient kernel — FlashAttention-2 is **not** available on T4. Keep a manual
attention fallback path behind a flag for debugging.

### 4.3 Precision & memory

- **T4: `float16` + `torch.cuda.amp.GradScaler`.** T4 has no bf16 support — this is the single
  most common Colab crash for people copying A100 code.
- **A100/L4: `bfloat16`, no scaler.** Make dtype a config field, not a hardcode.
- `torch.compile`: try it, but keep `compile: false` default on T4 — compile times and
  Triton/sm_75 issues frequently cost more than they save.
- **Gradient checkpointing: implement it, default it OFF.** The guideline asks for it, but 124M
  at ctx 1024 fits T4 comfortably; checkpointing would cost ~30% throughput for nothing. Turn it
  on only if you later scale to 350M+.

### 4.4 Optimizer

AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.1` on 2D params only (no decay on biases/LayerNorm),
`grad_clip=1.0`, `fused=True` when CUDA. Cosine LR schedule, warmup ~2000 steps,
`lr=6e-4 → 6e-5`.

### 4.5 Claude Code prompt

> Write `src/llmscratch/model.py`: a GPT-2-style decoder-only transformer from scratch in
> PyTorch — `CausalSelfAttention` using `F.scaled_dot_product_attention(is_causal=True)`, `MLP`
> with 4× GELU, pre-LN `Block`, and a `GPT` module with tied embed/LM-head weights, GPT-2 init
> with residual scaling, and optional gradient checkpointing behind a config flag (default off).
> Add `GPT.configure_optimizers()` returning AdamW with weight decay applied only to 2D
> parameters, `fused=True` on CUDA. Add `GPT.num_params()` and a `summary()` that prints a
> per-module parameter table and the total. Include a `__main__` that builds the 12L/768d/12H
> config and runs one forward+backward on random data to confirm shapes and report peak VRAM.
> It must run on CPU with a tiny config for tests.

**Exit criteria:** `python -m llmscratch.model` prints ~124M params and completes a
forward/backward; CPU smoke test in `tests/` passes on Windows.

---

## Phase 5 — Training Loop (Guideline Step 4) — the core phase

### 5.1 Requirements the loop must satisfy

1. **Resume is exact.** Checkpoint saves: model state, optimizer state, GradScaler state,
   `step`, `best_val_loss`, config, torch/numpy RNG states, and elapsed-token count.
   On start, `--resume auto` finds the newest checkpoint and continues; the loss curve after a
   resume must be visually continuous. Test this deliberately: kill at step 300, resume, confirm.
2. **Checkpoint policy.** Write to `/content/checkpoints` (fast local), then copy to Drive.
   Every 500 steps → `ckpt_last.pt` (atomic: write `.tmp`, then rename). Every 2000 steps →
   keep a numbered snapshot, retain last 3. Always keep `ckpt_best.pt` by val loss.
   A 124M checkpoint with optimizer state is ~1.5 GB — do the arithmetic against Drive quota
   before enabling numbered snapshots.
3. **Gradient accumulation** to a fixed *token* batch (~250k–500k tokens/step) so the T4 and A100
   configs train the same trajectory at different wall-clock speeds.
4. **Eval** every 250 steps on a fixed set of val batches, with `torch.no_grad()` + `model.eval()`.
   Log train loss, val loss, LR, tokens/sec, MFU estimate, VRAM.
5. **Logging** to a JSONL file on Drive (append-only, survives crashes) + TensorBoard events +
   a `--plot` mode that renders loss curves with matplotlib from the JSONL. Prefer JSONL as the
   source of truth — TensorBoard event files handle interrupted sessions badly.
6. **Sample generation** every 1000 steps: generate 200 tokens from a fixed prompt and log it.
   This is the fastest qualitative signal that training is working.

### 5.2 Config sketch (`configs/gpt124m_t4.yaml`)

```yaml
n_layer: 12
n_head: 12
n_embd: 768
block_size: 1024
dropout: 0.0            # single-epoch-ish on 2B tokens; no overfit risk
micro_batch_size: 8     # T4 15GB, fp16, ctx 1024 -> tune down to 4 if OOM
grad_accum_steps: 32    # 8*32*1024 = 262,144 tokens/step
max_steps: 8000         # ~2.1B tokens
learning_rate: 6.0e-4
min_lr: 6.0e-5
warmup_steps: 500
weight_decay: 0.1
grad_clip: 1.0
amp_dtype: float16      # T4: no bf16
compile: false
grad_checkpointing: false
eval_interval: 250
ckpt_interval: 500
```

`gpt124m_a100.yaml` differs only in `micro_batch_size: 32`, `grad_accum_steps: 8`,
`amp_dtype: bfloat16`, `compile: true`.

### 5.3 Session workflow (repeat until `max_steps`)

```
open notebook -> bootstrap cell -> copy .bin from Drive to /content/data
-> python -m llmscratch.train --config configs/gpt124m_t4.yaml --resume auto
-> train until session dies -> next session repeats, resumes automatically
```

Session 1 should be a **dry run**: `--max_steps 50 --ckpt_interval 10`, verify checkpoints land
on Drive, kill it, resume, confirm continuity. Only then start the real run.

### 5.4 Claude Code prompt

> Write `src/llmscratch/train.py`: a training loop for the GPT in `model.py` reading a YAML
> config. Requirements: gradient accumulation to a fixed tokens-per-step; AMP with `float16`+
> `GradScaler` or `bfloat16` selected by config; cosine LR with linear warmup; grad clipping;
> tqdm showing loss/lr/tokens-per-sec/VRAM. Periodic eval on fixed val batches. Append every
> metric row to a JSONL on Drive and mirror to TensorBoard. Every 1000 steps, generate a sample
> from a fixed prompt and log it. **Checkpointing must be exact-resume**: save model, optimizer,
> GradScaler, step, best_val_loss, config, and torch/numpy RNG states; write atomically to local
> disk then copy to Drive; keep `ckpt_last`, `ckpt_best`, and the last 3 numbered snapshots.
> Support `--resume auto|<path>|none`. Handle SIGTERM/KeyboardInterrupt by saving a checkpoint
> before exiting. Add `--plot` to render train/val loss curves from the JSONL with matplotlib.
> Verify by running 50 steps, killing it, and resuming — the loss curve must be continuous.

**Exit criteria:** resume test passes; val loss trending down; a loss-curve PNG on Drive.

**Expected loss trajectory** (GPT-2 BPE, so these are comparable to nanoGPT):
start ~10.9 → ~6.0 by step 200 → ~4.5 by step 1000 → ~3.4 by step 4000 → ~3.0–3.2 at 8000.
If you're above 5.0 after 2000 steps, something is wrong — most likely LR, or `y` not shifted.

---

## Phase 6 — Inference & UI (Guideline Step 5) (~1 h)

### 6.1 Sampling

`sample.py`: `@torch.no_grad()` autoregressive generation with temperature, top-k, top-p
(nucleus), repetition penalty, and a **streaming generator** (`yield` per token) so both the CLI
and the UI can display token-by-token. Crop context to the last `block_size` tokens.

### 6.2 Gradio UI (replaces the guideline's desktop GUI)

`ui.py`: prompt textbox; sliders for temperature (0.1–2.0), top-k (0–200), top-p (0.1–1.0),
max new tokens; checkpoint picker; streaming output; tokens/sec readout.
`demo.launch(share=True)` inside Colab; plain `demo.launch()` locally on Windows.

Note: the local Windows machine is CPU-only, so a 124M model generates at roughly 5–20 tok/s
there. Usable for demos, not fast. That's expected, not a bug.

### 6.3 Claude Code prompt

> Write `src/llmscratch/sample.py` — load a checkpoint, rebuild the GPT from the embedded
> config, and stream-generate with temperature / top-k / top-p / repetition-penalty, cropping
> context to `block_size`. Expose both a CLI and a `stream_generate()` generator yielding decoded
> token strings. Then write `src/llmscratch/ui.py`, a Gradio app using that generator: prompt
> box, sliders for all sampling params, a dropdown listing checkpoints in the checkpoint dir,
> streaming token-by-token output, and a tokens/sec counter. It must work both in Colab
> (`share=True`) and locally on CPU — auto-detect the device.

**Exit criteria:** Gradio UI renders inline in Colab and streams text from `ckpt_best.pt`.

---

## Phase 7 — Wrap-up

1. `README.md`: what was built, actual final val loss, token count, total GPU-hours, sample
   outputs, and how to reproduce.
2. Optional: `model.push_to_hub()` for the weights + a model card.
3. Optional next steps if you want to keep going — SFT on an instruction dataset
   (Alpaca/Dolly-style) to make it respond to prompts rather than just continue text; this is
   where the model stops feeling like a toy.

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Free-tier GPU unavailable at peak | Fall back to CPU-less waiting, or Colab Pro (~$10) |
| 12h cap / 90min idle kill | Exact-resume checkpointing (Phase 5.1) — non-negotiable |
| Drive quota exhausted (4GB data + ~5GB ckpts) | Check quota in Phase 3; cap numbered snapshots at 3 |
| OOM on T4 | Lower `micro_batch_size` to 4 and double `grad_accum_steps` — same trajectory |
| `bf16` code copied from A100 examples crashes on T4 | `amp_dtype` is a config field; T4 config uses fp16 |
| Drive FUSE bottlenecks the dataloader | Copy `.bin` to `/content/data` each session |
| `colab` CLI 30s exec timeout kills training | Use `colab console` + tmux + nohup, not `exec` |
| Compute units burn on an idle VM | `colab stop -s trainer` when done |

---

## Timeline

| Phase | Effort | Wall clock |
|---|---|---|
| 1 Scaffold | 30 min | 30 min |
| 2 Colab access | 20 min (+1h if WSL2) | 20 min |
| 3 Data | 30 min coding | 1–2 h tokenizing |
| 4 Model | 1 h | 1 h |
| 5 Training | 2 h coding | **10–30 h GPU**, spread over 1–3 sessions |
| 6 UI | 1 h | 1 h |
| 7 Wrap-up | 30 min | 30 min |

Active work ≈ 6 h. Elapsed ≈ 2–4 days depending on how much training you buy.

---

## Deviations From guideline.md (and why)

| Guideline says | Plan does | Why |
|---|---|---|
| Install PyTorch+CUDA locally | Skip; CPU-only local venv | No NVIDIA GPU on this machine |
| Paste code into Colab cells | Git repo + thin notebooks | Cells are unversioned and lost on crash |
| 500MB–1GB of text | ~2B tokens (~8GB raw) | 124M params needs ~2.5B tokens; data is the one thing you can't retrofit |
| 80/20 train/val split | ~10M held-out val tokens, doc-level | 400M val tokens is wasted compute; token-level split leaks context |
| Gradient checkpointing on | Implemented, default off | Model fits T4; would cost ~30% speed for no benefit |
| "Atom optimizer" | AdamW, decoupled decay | Presumed transcription of "Adam"; AdamW is correct here |
| Tkinter desktop GUI | Gradio | Colab can't launch desktop apps; Gradio runs in both places |
