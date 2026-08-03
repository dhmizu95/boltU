# boltU — Building an LLM From Scratch on Kaggle

Adapted from `guideline.md`, restructured for **Kaggle** instead of a local NVIDIA GPU. Colab is
kept as an optional fallback in Appendix A.

---

## 0. Read this before you start

The original guideline is a good skeleton, but three of its assumptions will sink you. Fix them up front.

| Guideline says | Problem | What to do instead |
|---|---|---|
| "split into 80/20 training/validation" | Standard for supervised ML, wrong for LM pretraining. You'd burn 20% of your token budget on a validation set 1000× larger than needed. | Hold out a **fixed 5–10M tokens** for validation (~0.5–2%). Everything else trains. |
| "500 MB to 1 GB" of text for a 100–200M param model | 1 GB of text ≈ 250M tokens. A 124M-param model wants ~2.5B tokens (Chinchilla: ~20 tokens/param). You'd be **10× under-trained** — the model will produce fluent-ish gibberish and you'll blame the code. | Either shrink the model to ~30M params, or get 10 GB+ of text. See §1. |
| "use mixed precision **and** gradient checkpointing" | Mixed precision: yes, always. Gradient checkpointing: costs 25–35% throughput. It's an OOM remedy, not a default. | Enable AMP always. Add gradient checkpointing **only** if you OOM, and try reducing micro-batch + raising grad-accum first. |

### Why Kaggle and not Colab

| | Kaggle free | Colab free |
|---|---|---|
| GPU | **T4 ×2** or P100 | T4 |
| Weekly quota | **~30 h, stated**; resets Sat 00:00 UTC | opaque, varies, can cut you off mid-run |
| Session cap | 12 h hard (9 h TPU) | ~12 h, idle disconnects |
| Concurrency | **1 interactive + 2 batch jobs** | 1 |
| Persistence | **Kaggle Datasets, 200 GB private** | Drive, 15 GB |
| Execution model | batch (`kernels push`) + interactive | interactive |
| Internet | **off by default** — enable per notebook | on |
| Tooling maturity | `kaggle` CLI, stable for years | `colab` CLI shipped June 2026, flags still moving |

The deciding factors are the quota you can plan against and the 200 GB of storage. Colab's free
quota is opaque, so you cannot schedule a multi-session run against it; and Drive's 15 GB forces a
checkpoint-pruning script that Kaggle simply doesn't need. Colab's one real advantage is snappier
interactive iteration — Kaggle's interactive sessions cover that adequately, and not well enough to
justify standing up a second auth setup, a second persistence layer, and a second launcher.

### Two Kaggle realities that shape the whole design

- **Sessions die, and a batch job killed at the 12 h cap never runs your save path.** Checkpoint /
  resume isn't a nice-to-have — it's the core architectural requirement — and it must be paired with
  the wall-clock self-termination in §6. Design for "process is killed at a random step" from day one.
- **`/kaggle/working` is capped at ~20 GB** and is the only writable location that becomes kernel
  output. Attached datasets at `/kaggle/input/` are read-only and read fast. Scratch that you don't
  need to keep goes in `/tmp`.

**Prerequisite:** GPU and internet access both require a phone-verified Kaggle account. Do that now,
not at Phase 3.

---

## 1. Phase 0 — Choose your scope (do this first, on paper)

Pick one row. This single decision determines everything downstream.

| Tier | Layers / d_model / heads | Ctx | Params | Token budget (20×) | Text needed | Realistic on free Kaggle |
|---|---|---|---|---|---|---|
| **A — Learning run** | 6 / 384 / 6 | 512 | ~30M | 600M | ~2.5 GB | T4, **one ~4 h session** |
| **B — Middle** | 8 / 512 / 8 | 1024 | ~51M | 1.0B | ~4 GB | T4 ×2, 2–3 sessions |
| **C — GPT-2 small clone** (guideline's target) | 12 / 768 / 12 | 1024 | ~124M | 2.5B | ~10 GB | T4 ×2, **1–2 weeks of quota** |

Rough throughput to sanity-check your timeline (tokens/sec, 124M params, ctx 1024, AMP, no gradient
checkpointing — **measure your own, these vary 2× by dataloader and kernel**):

| GPU | tok/s | Time for 2.5B tokens |
|---|---|---|
| P100 (fp16, no tensor cores) | ~4–8k | 4–7 days of pure compute |
| T4 (fp16, no bf16) | ~8–15k | 2–4 days |
| T4 ×2 (DDP, ~1.75×) | ~14–26k | 27–50 h |
| L4 *(paid Colab)* | ~30–45k | 15–23 h |
| A100 40GB *(paid Colab)* | ~120–180k | 4–6 h |

**Read the Tier C row honestly.** 27–50 h of compute against a ~30 h weekly quota means Tier C is a
one-to-two-week project with 3–5 sessions, not a weekend. Tier A is a single afternoon. That gap is
the strongest argument for the recommendation below.

**Two things the throughput table assumes.** It is measured at 124M params / ctx 1024 — do not apply
it to Tier A directly. Most of Tier A's speedup comes from the model being ~4× smaller (expect ~3–4×
the tok/s, less than 4× because small models are more overhead-bound), *not* from the shorter
context. Dropping ctx 1024→512 is worth only ~7% here, since attention is roughly 13% of FLOPs/token
at this scale.

**Prefer T4 ×2 over P100.** P100 has more memory bandwidth but no tensor cores, so its fp16 peak
(~19 TFLOPS) is far below T4's (~65 TFLOPS) for AMP training. Neither has bf16 — the fp16 +
`GradScaler` path in §6 applies on both. Check what you actually got with
`torch.cuda.get_device_name()` before committing to a tier.

**Recommendation:** do Tier A end-to-end first. Get a working pipeline that trains, checkpoints,
resumes, and generates text. *Then* scale the config to Tier B or C and re-run. Debugging a 3-day run
is miserable; debugging a 4-hour run is fine. The code is identical — only a config file changes. Tier
A also fits in one session, so you skip the entire marathon apparatus in §7 and most of §11 on your
first pass.

**Deliverable:** a filled-in `configs/base.yaml` with your chosen numbers.

---

## 2. Phase 1 — Repo scaffold (local machine)

The guideline builds a local project folder; that still applies, but the repo lives locally/on GitHub
and **executes** on Kaggle.

**Naming convention.** The project is branded **boltU**, but every *identifier* is lowercase `boltu`:
the Python package and all Kaggle slugs. Two reasons this isn't fussiness. Kaggle lowercases dataset
and kernel slugs regardless of what you type, so `boltU-tokens` silently becomes `boltu-tokens` and
any hardcoded `/kaggle/input/boltU-tokens` path fails at runtime. And if you develop on macOS
(case-insensitive filesystem) with a `boltU/` package, `import boltU` works locally and then dies on
Kaggle's Linux box. Keep `boltU` in prose, READMEs, and the UI title; keep `boltu` everywhere a
machine reads it.

```
boltU/
├── configs/
│   ├── base.yaml            # model + training hyperparams
│   └── tiny.yaml            # 2-min smoke-test config
├── data/                    # gitignored; token shards live here
├── kernel/                  # what gets pushed to Kaggle
│   ├── kernel-metadata.json
│   └── run.ipynb            # thin launcher: clone repo, call train.py
├── src/
│   ├── data_prep.py         # download → tokenize → .bin shards
│   ├── dataset.py           # RAM/memmap loader, batch sampling
│   ├── model.py             # GPT implementation
│   ├── train.py             # loop, checkpointing, resume, --max-hours
│   ├── sample.py            # inference / generation
│   └── app.py               # Gradio UI
├── checkpoints/             # gitignored
├── requirements.txt
└── README.md
```

**Prompt to give Claude Code:**

> Scaffold a Python project for training a GPT-style language model from scratch. Create this directory structure: configs/, data/, kernel/, src/, checkpoints/. Add a requirements.txt with torch, tiktoken, datasets, numpy, matplotlib, tqdm, gradio, pyyaml. Add a .gitignore excluding data/, checkpoints/, *.bin, and __pycache__. Create configs/base.yaml with a documented schema covering model dims (n_layer, n_head, n_embd, block_size, vocab_size, dropout) and training params (batch_size, grad_accum_steps, learning_rate, warmup_steps, max_steps, weight_decay, grad_clip, eval_interval, checkpoint_interval, max_hours). Do not write model or training logic yet — structure and config only.

**Push it to GitHub before Phase 2.** This is the sync channel — the Kaggle kernel does a
`git clone` at startup, so you never copy-paste code into cells and every run records which commit it
trained. A private repo is fine; put the token in a **Kaggle Secret**, not inline.

```bash
gh repo create boltu --private --source=. --remote=origin --push
```

**Done when:** `tree` shows the structure, `configs/base.yaml` parses with `yaml.safe_load`, and the
repo is pushed.

---

## 3. Phase 2 — Kaggle environment (replaces guideline Step 1)

The guideline's Step 1 installs CUDA PyTorch locally. On Kaggle, PyTorch + CUDA are **pre-installed**
— don't reinstall torch, you'll break the CUDA build and waste 10 minutes. Only add what's missing.

### 3.1 Local CLI setup

```bash
pip install kaggle
# Account → Settings → API → Create New Token, then:
export KAGGLE_API_TOKEN=<the token value>
kaggle kernels list -m          # verify auth
```

**Verified against the live API (2026-08):** Kaggle CLI 2.x auth is a single bearer token, not the
classic `username`+`key` pair — the old `~/.kaggle/kaggle.json` **Basic Auth** flow now gets a hard
`401 {"code":401,"message":"Unauthenticated"}` from the server, confirmed with `curl` directly against
`api.kaggle.com`, independent of the CLI. The *same* token value works immediately as a Bearer token
(`kaggle kernels list -m` with `KAGGLE_API_TOKEN` set, or `~/.kaggle/access_token`). If you're on an
older pinned `kaggle<2`, the Basic Auth path it uses is dead regardless of how correct your credentials
are — upgrade instead of debugging the credentials. Inside a notebook, never paste the token: use
**Kaggle Secrets** (Add-ons → Secrets) and read it into the environment at runtime.

### 3.2 Interactive session (Phases 1–5 development)

New Notebook → Settings pane → **Accelerator: GPU T4 ×2**, **Internet: On**. Then in the first cell:

```python
import os, subprocess, torch
from kaggle_secrets import UserSecretsClient

tok = UserSecretsClient().get_secret("GH_TOKEN")
if not os.path.isdir("/kaggle/working/boltU"):
    subprocess.run(["git", "clone", f"https://{tok}@github.com/<you>/boltu.git",
                    "/kaggle/working/boltU"], check=True)
os.chdir("/kaggle/working/boltU")
subprocess.run(["git", "pull", "--ff-only"], check=True)

subprocess.run("pip install -q tiktoken pyyaml".split(), check=True)
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(),
      torch.cuda.get_device_name(0))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

Same cell at the top of every notebook. It's idempotent — clone on first run, `git pull` after.
Interactive sessions idle out after tens of minutes of inactivity; they're for iterating, never for
the long run.

### 3.3 Batch execution — the real training path

There is no `exec`-style remote call. You push a folder containing a notebook plus a
`kernel-metadata.json`:

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
kaggle kernels push -p ./kernel                        # submit
kaggle kernels status <username>/boltu-train           # poll
kaggle kernels output <username>/boltu-train -p ./out  # retrieve after completion
```

`enable_internet: true` is required if the kernel clones from GitHub, downloads from Hugging Face, or
pushes checkpoints back via the API. It's off by default and the failure is a confusing DNS error, not
a clear message. (The UI equivalent of `kernels push` is **Save Version → Save & Run All**.)

`run.ipynb` should stay thin — clone the repo, then call `train.py`. All logic lives in the repo, not
in the notebook.

**Batch jobs run with your machine off.** They execute server-side, fully detached — push the job,
close the browser, shut down. Interactive sessions do not: they're tied to your connection and idle out
shortly after you disconnect. This is why the marathon in §7 is batch-only, and it's a real advantage
over Colab, where the CLI's keep-alive daemon runs on *your* machine and dies with it.

### 3.4 Space check at startup

`/kaggle/working` is ~20 GB, and §7 stages a checkpoint zip in it while the checkpoints themselves are
already there. Fail at minute one, not hour eleven:

```python
import shutil
free_gb = shutil.disk_usage("/kaggle/working").free / 1e9
assert free_gb > 12, f"only {free_gb:.1f} GB free in /kaggle/working"
```

Token shards do **not** count against this — they arrive as a read-only attached dataset at
`/kaggle/input/`. What counts is checkpoints (~1.5 GB each at Tier C, weights + optimizer state) plus
whatever the dataset-versioning step stages.

**On precision:** neither T4 nor P100 has bf16 — that starts at Ampere. Both are fp16-only, which
needs a `GradScaler` (§6). bf16 shares fp32's exponent range and needs no loss scaling, but you only
get it on paid Colab's L4/A100/H100 (Appendix A).

**Checklist before moving on:**
- [ ] Phone-verified account; GPU and Internet both enabled in notebook settings
- [ ] `kaggle kernels list -m` authenticates from your local machine
- [ ] `torch.cuda.is_available()` is `True` and you know your GPU name and count
- [ ] `GH_TOKEN` stored as a Kaggle Secret and the clone cell works
- [ ] Disk assert passes

---

## 4. Phase 3 — Data pipeline (guideline Step 2, corrected)

Tokenize **once**, save as flat `uint16` binary shards, then load them. GPT-2's vocab is 50257 <
65536, so `uint16` halves your disk and I/O versus `int32`. Never re-tokenize at training time.

### Dataset options

| Dataset | Size | Notes |
|---|---|---|
| `HuggingFaceFW/fineweb-edu`, `sample-10BT` | slice to taste | Best quality/effort ratio. Stream it; take as many tokens as your tier needs. |
| `wikitext-103-raw-v1` | ~500 MB | Clean, small, fine for Tier A. |
| `openwebtext` | ~38 GB | The GPT-2 reproduction target; overkill unless Tier C. |
| Project Gutenberg | varies | Public domain, but archaic English — model will sound like 1890. |

**Prompt:**

> Write `src/data_prep.py`. It should stream a Hugging Face dataset (default: HuggingFaceFW/fineweb-edu, sample-10BT) up to a target token count from config. Tokenize with tiktoken's GPT-2 BPE encoding, appending the `<|endoftext|>` token after each document. Write tokens to flat uint16 `.bin` shards of 100M tokens each in `data/`, using multiprocessing over documents with a tqdm progress bar. Hold out the first shard as validation and cap it at 10M tokens — do NOT use an 80/20 split. Save a `data/meta.json` with total token count, vocab size, and shard filenames. At the end, assert that `enc.decode(enc.encode(s)) == s` for a random 500-character sample of the raw text, print the total token count, and decode a random 200-token window back to text.

The `decode(encode(s)) == s` assert is the point — "the sample looks coherent" is a heuristic you can
misread at 2am, and a silent tokenizer mismatch is the top cause of "good loss, bad output" at the end
of this project (§13).

Then `src/dataset.py`:

> Write `src/dataset.py` with a `get_batch(split, batch_size, block_size, device)` function that loads the uint16 shards into RAM with `np.fromfile` at startup (falling back to `np.memmap` above a configurable size threshold), samples random offsets, builds x and y where y is x shifted by one position, and transfers to GPU with `pin_memory()` and `non_blocking=True`. If using memmap, re-open it each call to avoid a known memory-leak pattern. Include a `__main__` block that prints the shapes and dtypes of one batch.

**Cut the val holdout on document boundaries.** "First shard, capped at 10M tokens" is right in
spirit, but if you truncate mid-document the val set's final window shares a document with
`train_0001.bin` — a small, real leak that makes val loss read slightly optimistic. Stop the val shard
at the last complete document before 10M tokens and start training data at the next one.

### Persist as a Kaggle Dataset

Tokenizing 2.5B tokens takes real time. Do it once, upload as a private dataset, and attach it to every
subsequent run — it mounts read-only at `/kaggle/input/boltu-tokens` and reads fast:

```bash
kaggle datasets create -p ./data --dir-mode zip    # once; needs a dataset-metadata.json
kaggle datasets version -p ./data -m "v2: 2.5B tokens" --dir-mode zip
```

At 200 GB of private dataset quota you are not constrained even at Tier C. This is the single biggest
practical advantage over Colab's 15 GB Drive.

**Done when:** `meta.json` reports your target token count, the round-trip assert passes, the decoded
sample window reads as natural language, and the dataset is attached to a test kernel and readable at
`/kaggle/input/`.

---

## 5. Phase 4 — Model architecture (guideline Step 3)

**Prompt:**

> Write `src/model.py`: a decoder-only GPT in pure PyTorch, no Hugging Face modeling classes. Include token + learned positional embeddings, N transformer blocks each with pre-LayerNorm, causal multi-head self-attention using `F.scaled_dot_product_attention(..., is_causal=True)` so it uses Flash Attention kernels, an MLP with 4× expansion and GELU, and residual connections. Tie the token embedding and output head weights. Initialize weights with normal std 0.02, but scale residual-projection weights by `0.02/sqrt(2*n_layer)`. Read all dims from the YAML config. Add a `configure_optimizers` method returning AdamW with betas (0.9, 0.95), applying weight decay 0.1 only to 2D parameters (not biases or LayerNorms), and using the fused implementation when CUDA is available. Add a `generate` method with temperature, top-k, and top-p sampling. Add a `__main__` block that instantiates the model, prints a layer summary and total parameter count, and runs one forward pass on random input to verify output shape is (B, T, vocab_size).

**Done when:** parameter count is within ~5% of your tier's target and the forward pass returns the
right shape. If you're targeting 124M and see 160M, you probably forgot weight tying.

---

## 6. Phase 5 — Training loop (guideline Step 4, hardened for Kaggle)

This is where the plan diverges most from the guideline. The loop must be **interruption-proof** and
must **stop before Kaggle stops it**.

**Prompt:**

> Write `src/train.py` implementing a training loop with these requirements:
>
> **Core:** gradient accumulation to reach an effective batch of ~0.5M tokens per optimizer step; AMP autocast using float16 with a GradScaler on T4/P100 and bfloat16 on Ampere+ if detected; gradient clipping at 1.0; cosine learning-rate decay to 10% of peak with linear warmup; `torch.compile` on the model with a flag to disable it.
>
> **Checkpointing (critical):** every N steps, save model state, optimizer state, scaler state, step number, RNG states, and config to `checkpoints/ckpt_latest.pt` under `/kaggle/working`. Write to a temp file and atomically rename so a mid-write crash can't corrupt it. Keep only the last 2 checkpoints plus the best-validation checkpoint.
>
> **Resume:** a `--resume` flag that loads the latest checkpoint from `/kaggle/input/boltu-checkpoints/` if present, else from local `checkpoints/`, and continues from the exact step with optimizer and scheduler state intact. This must be exact, not approximate.
>
> **Wall-clock budget:** a `--max-hours` argument, default 11.0. Record start time at launch. After each optimizer step, check elapsed time; once it exceeds the budget, break the loop, save a final checkpoint, push it as a new Kaggle dataset version, and exit 0 cleanly. Estimate remaining time per step and break *early* if the next checkpoint interval would overrun the budget — never start work you cannot persist. Log the exit reason explicitly as one of: max_steps reached, time budget reached, or error.
>
> **Monitoring:** estimate validation loss over ~50 batches every eval_interval steps, using a **fixed RNG seed so the same 50 windows are scored at every eval** — otherwise val loss wiggles for reasons unrelated to training and the curve is unreadable; log step, train loss, val loss, learning rate, tokens/sec, and elapsed time to a CSV; tqdm progress bar; append-mode logging so resumed runs extend the same CSV.
>
> **Safety:** wrap the loop so SIGTERM/KeyboardInterrupt triggers a final checkpoint save before exit.

**`--max-hours` is not optional and not a §12 afterthought.** A batch job killed at the 12 h cap never
runs your save path and `/kaggle/working` output is not retained — the whole session is lost. Leave a
full hour of margin: dataset versioning of a multi-GB checkpoint takes real minutes, and it is the last
thing you want racing a hard kill. For runs longer than ~6 h, also push one mid-run dataset version at
the halfway point so a crash at hour seven doesn't cost you everything.

Starting hyperparameters (124M; scale LR up for smaller models):

```yaml
learning_rate: 6.0e-4      # peak; try 1e-3 for <50M params
min_lr_frac: 0.1           # cosine floor = 10% of peak
weight_decay: 0.1
grad_clip: 1.0
betas: [0.9, 0.95]
warmup_frac: 0.02          # 2% of max_steps — DERIVED, not hardcoded
max_hours: 11.0
```

**`tokens_per_step` is per-tier, not a constant.** nanoGPT's 0.5M is calibrated for 124M params.
Applying it to Tier A gives 600M / 0.5M = **1,200 total steps**, which is far too few for a cosine
schedule to do anything sensible — a fixed 700-step warmup would consume more than half the run.

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
# Smoke test FIRST — 50 steps on the tiny config, in the interactive session.
python src/train.py --config configs/tiny.yaml --max_steps 50

# Real run — batch job, detached from your browser entirely.
kaggle kernels push -p ./kernel
kaggle kernels status <username>/boltu-train
```

Log `grad_norm` (pre-clip) at step 1 and step 50 of the smoke test. If it has exploded past ~5 by step
50, your LR or warmup is wrong — find that out in two minutes, not two days.

**Loss sanity checks.** With GPT-2's 50257-token vocab, initial loss should be ≈ `ln(50257)` ≈
**10.8**. If it isn't, your init or your targets are wrong. Then:

| Loss | Meaning |
|---|---|
| ~10.8 | step 0, correct |
| ~6.0 | learned unigram frequencies (fast, first few hundred steps) |
| ~4.5 | basic syntax, plausible word order |
| ~3.5 | coherent short passages |
| ~3.0 | ballpark of GPT-2 small on OpenWebText |
| flat or NaN | LR too high, or fp16 overflow — lower LR, verify GradScaler |

**Done when:** you can kill the process mid-run, relaunch with `--resume`, and see loss continue from
where it stopped with no discontinuity in the curve. **Test this deliberately before the long run.** It
is the single most important test in this project. Then verify `--max-hours` the same way: set it to
0.05, confirm the run exits 0 with a saved checkpoint and a logged exit reason.

---

## 7. Phase 6 — The multi-session marathon

**Tier A skips this section** — 600M tokens fits in one ~4 h session. Come back when you scale up.

For Tier B/C, per-session routine:

1. `kaggle datasets version` the checkpoint from the previous session (if the run didn't already)
2. Confirm `dataset_sources` in `kernel-metadata.json` points at the current checkpoint dataset
3. `kaggle kernels push -p ./kernel`
4. `kaggle kernels status` until it reports running; walk away
5. On completion, `kaggle kernels output` for logs, and confirm the new checkpoint dataset version exists

Wrap 1–3 in `scripts/session_start.sh` — you'll do it a dozen times.

The kernel itself resumes from `/kaggle/input/boltu-checkpoints/ckpt_latest.pt` and pushes a new
version before it exits. **The Drive-pruning script from the Colab version is unnecessary here** —
keep the dataset version history, it's free at 200 GB and it's your only rollback if a session
corrupts a checkpoint.

Watch out for: the ~30 h weekly GPU quota (a Tier C run will consume most of it — check your usage
before pushing a 12 h job on a Friday, since the reset is Saturday 00:00 UTC); silent allocation of
P100 instead of T4 ×2 (log `torch.cuda.get_device_name()` every session so you can explain throughput
changes); and `enable_internet` silently reverting if you edit metadata by hand.

### The monitoring gap

`kernels output` only returns artifacts after the job completes, so live plotting doesn't work as
written. Options, in order of preference:

1. **Accept batch blindness.** With `--max-hours` self-termination and verified resume, a 12 h run
   that reports at the end is tolerable. This is the honest default, and it's what you should do first.
2. **Push a small metrics dataset from inside the kernel** every 30–60 minutes (not every eval —
   versioning is rate-limited). Poll it locally with `kaggle datasets download`. CSV plus heartbeat is
   kilobytes.
3. **Run the interactive session concurrently** for spot checks — you have the concurrency budget for
   it (1 interactive + 2 batch).

---

## 8. Phase 7 — Evaluation and sampling

Loss alone won't tell you if the thing is usable.

**Prompt:**

> Write `src/sample.py`: load a checkpoint, accept a prompt string, and generate text with configurable temperature, top-k, top-p, and max tokens. Support streaming token-by-token to stdout. Add a `--benchmark` mode that computes validation perplexity over the full held-out set. Add a `--compare` mode that generates from the same prompt at temperatures 0.7, 0.9, and 1.2 side by side. Also write `src/plot_curves.py` that reads the training CSV and plots train and validation loss on a log-x axis, saving to `plots/loss.png`.

Judge output on: does it stay on topic for 50+ tokens, is the grammar consistent, does it loop or
repeat, and does validation loss track train loss (divergence = overfitting; on a 20:1 token/param
budget you should be nowhere near it).

---

## 9. Phase 8 — The interface (guideline Step 5, adapted)

The guideline's Tkinter/PyQt desktop GUI **cannot run in a Kaggle kernel** — no display server. Two
options:

**Local, recommended.** `kaggle kernels output` or `kaggle datasets download` the checkpoint and run
the UI on your own machine. A 124M model runs fine on CPU for inference — a few tokens/sec, adequate
for testing — and this is where the guideline's original Tkinter app works unchanged if you want it.

**Gradio in the notebook**, if you want to demo without downloading:

> Write `src/app.py`: a Gradio interface for the trained model. Include a multiline prompt textbox, sliders for temperature (0.1–2.0), top-k (0–200), top-p (0.1–1.0), and max new tokens, plus a model-checkpoint dropdown that scans the checkpoints directory. Stream generated tokens to the output box as they're produced using a generator function. Show tokens/sec and total time under the output. Load the model once at startup, not per request. Launch with `share=True` and `server_name="0.0.0.0"`.

`share=True` needs `enable_internet: true`, and the tunnel dies with the session — this is a demo path,
not a deployment.

---

## 10. Phase 9 — Optional: make it follow instructions

A base model completes text; it doesn't answer questions. If you want chat-like behavior, add a
fine-tune stage: format a small instruction dataset into a consistent prompt template, mask the loss so
it's computed only on response tokens, and train 2–3 epochs. This is a separate script
(`src/finetune.py`) that loads the pretrained checkpoint — do not mix it into `train.py`.

Scale reference: Alpaca is ~52k examples (~10M tokens), Dolly-15k ~3M tokens, OASST1 ~20M. Against
Tier C's 2.5B-token pretraining budget this is under 1% — SFT reshapes output *format*, it does not add
knowledge. At 10–20% of a 6e-4 peak LR you're at roughly **6e-5 to 1.2e-4**, with a short warmup and no
restart of the cosine schedule.

Two things to expect. Loss numbers are not comparable to pretraining (you're scoring only response
tokens over a much narrower distribution, so it will look lower and means nothing across stages). And
SFT alone, without preference data, commonly costs you some base capability — the alignment tax. Keep
the pretrained checkpoint and compare generations side by side; if the tuned model is more polite but
noticeably dumber, that's the tax, and the fix is fewer epochs or a lower LR, not more data.

---

## 11. Phase 5b — Robustness and telemetry

> **Ordering note.** §11 is documented after the phase list but **executed between Phase 5 and Phase
> 6**. The master checklist is in execution order; follow that, not the section numbers.
>
> **Scope this by tier.** §11.1–11.4 are cheap and apply to every run. **§11.5 (telemetry) is Tier B/C
> only** — on a 4-hour Tier A run, a CSV of loss and tok/s tells you everything you need, and building
> MFU tracking and a throttle-reason thread first is how you spend your afternoon on infrastructure
> instead of on a language model. Build it when a run is long enough that you can't just watch it.

### 11.1 Data loading: prefer RAM over memmap

At 2 bytes/token, token budgets are smaller than they look:

| Tier | Tokens | On-disk | Fits in Kaggle RAM (~13 GB)? |
|---|---|---|---|
| A | 600M | 1.2 GB | Easily |
| B | 1.0B | 2.0 GB | Easily |
| C | 2.5B | 5.0 GB | Yes, but the val array, CUDA context, and page cache share the box — watch it |
| beyond | 5B+ | 10 GB+ | No — memmap required |

**Default path:** `np.fromfile(shard, dtype=np.uint16)` once at startup, hold in RAM, slice from it. No
page-cache thrashing, no read amplification, no IOPS pressure. Memmap is the fallback for corpora above
~4B tokens, not the default. Reading directly from `/kaggle/input` via memmap is fine too — it's real
local disk, unlike Drive.

If you do need memmap: re-open it inside `get_batch()` each call (holding a long-lived memmap object
leaks), and log `psutil.Process().memory_info().rss` alongside `/proc/meminfo` `Cached` so you can
distinguish real RSS growth from reclaimable page cache. Only the former is dangerous.

### 11.2 No DataLoader, no workers

**Batch construction detail.** Slicing `arr[i:i+block_size]` from a 1-D contiguous buffer already
returns a contiguous view — no `.contiguous()` needed. But token IDs are `uint16` on disk and
`nn.Embedding` requires int64 indices, so a cast is unavoidable. Do it **once on the stacked batch**,
not per window:

```python
ix = np.random.randint(0, len(arr) - block_size, size=micro_batch)
x = np.stack([arr[i   : i+block_size]   for i in ix]).astype(np.int64)
y = np.stack([arr[i+1 : i+1+block_size] for i in ix]).astype(np.int64)
```

If profiling later shows batch prep is non-trivial, the win is preallocating pinned buffers and copying
into them — not adding `.contiguous()` calls.

`get_batch()` samples random offsets from an in-RAM array and calls `.pin_memory()` on the resulting
tensor — that is the *tensor method*, unrelated to `DataLoader(pin_memory=True)`. There is no
`num_workers` in this design and adding one is not possible without restructuring.

Verify this is fine rather than assuming: log `t_data / t_step`. If data prep exceeds ~2% of step time,
revisit. On Kaggle's low-core instances, worker processes typically cost more in context switching than
they recover.

### 11.3 Checkpoint persistence must fail loudly

Kaggle removes most of the risk here — `/kaggle/working` is local disk, not a FUSE mount, so
`os.replace` is genuinely atomic and there's no background sync thread to die silently. What remains is
the **dataset version push**, which is a network call that can fail at the worst possible moment.

**Prompt:**

> In `src/train.py`, wrap the Kaggle dataset-versioning call in explicit error handling: on failure, retry twice with backoff, and if all attempts fail, log the error at ERROR level and leave the checkpoint on local disk rather than exiting silently. Verify each successful push by calling `kaggle datasets status` (or the API equivalent) and confirming the version count incremented. Also write a `heartbeat.json` next to the metrics CSV every eval interval containing step number, wall-clock timestamp, and latest val loss.

**File mtime is the liveness signal**, not the heartbeat's contents — it's readable without parsing
anything. Wrap any body read in try/except and fall back to the last successful parse. Append-only JSONL
is a fine alternative, since a torn final line is simply discarded.

### 11.4 Derive grad accumulation, never hardcode it

```python
tokens_per_micro = micro_batch * block_size
assert target_tokens_per_step % tokens_per_micro == 0, (
    f"{target_tokens_per_step} not divisible by {tokens_per_micro}; "
    f"adjust micro_batch or target"
)
grad_accum_steps = target_tokens_per_step // tokens_per_micro
```

Recompute on every launch, including after an OOM forces `micro_batch` down. Log the resulting
effective batch size at startup and assert it matches the config target — a silent drift from 0.5M to
0.43M tokens/step distorts the LR schedule you tuned for.

### 11.5 `src/telemetry.py` — Tier B/C only

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

It largely eliminates fragmentation for fixed-shape training loops and is the first remedy if
`num_alloc_retries` climbs.

### 11.6 What to watch on a Tier C run

| Signal | Healthy | Action if not |
|---|---|---|
| MFU | 15–20% (T4), ~10% (P100) | Stable is what matters; a mid-run drop means throttling or a GPU downgrade |
| `num_alloc_retries` | 0, flat | Climbing → enable `expandable_segments`, then lower micro-batch |
| p99 / p50 step time | < 1.5× | Larger gap → page faults from memmap, or a blocking dataset push |
| data fetch % of step | < 2% | Higher → move to in-RAM array |
| throttle reasons | none / `GpuIdle` | `SwPowerCap` or `HwThermalSlowdown` → expect reduced throughput, not a bug |
| heartbeat mtime | advancing | Stalled → session died; check `kernels status`, then resume |

Plotting MFU against the same axis as loss is what separates "the model stopped learning" from "the GPU
got throttled at hour nine." Worth doing offline from the CSV after a Tier C session, not live.

### 11.7 2×T4 with DDP

On free Kaggle this is the **only** way Tier B or C reaches its token budget inside the weekly quota —
treat it as required for those tiers, not an optimization. Two T4s give ~1.7–1.8× throughput, not 2×.
Get single-GPU training fully working first.

The one correctness trap: **`grad_accum_steps` must be divided by `world_size`**, or your effective
batch silently doubles and the §11.4 assertion is the only thing that will catch it.

---

## 12. Appendix A — Colab (optional)

Kaggle is the primary platform. Reach for Colab in exactly two situations.

**You have Colab Pro.** L4/A100 access changes the arithmetic completely: Tier C drops from a 1–2 week
Kaggle project to 15–23 h on an L4 or 4–6 h on an A100, and those GPUs have **bf16**, so the
`GradScaler` path disappears. At that point Colab is the better training platform and this appendix
becomes your §3 and §7. Free-tier Colab is T4-only and A100/H100 require a paid plan — allocation is
never guaranteed on any tier.

**You want faster interactive iteration on Phases 1–5.** Colab's notebook loop is snappier than
Kaggle's. The code is identical; only the launcher and persistence layer change.

### Notebook setup

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

PyTorch + CUDA are pre-installed — don't reinstall torch, you'll break the CUDA build.

### Colab CLI

Google released an official CLI in June 2026 (`google-colab-cli`, binary `colab`, Python ≥3.12,
Linux/macOS only — Windows needs WSL2). It connects a local terminal to a remote runtime, with a
built-in keep-alive daemon that removes the ~90 min idle disconnect but *not* the ~12 h session
ceiling.

```bash
uv tool install google-colab-cli
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab --auth=adc whoami

colab new -s boltu --gpu L4
colab install -s boltu -r requirements.txt
colab drivemount -s boltu
```

**Validate this before building anything on it.** The tool is new and flags have moved; `colab <cmd>
--help` is authoritative. Spend ten minutes on `new` → `console` → `stop` before committing a training
run to it. If anything misbehaves, the notebook path above is the fallback and costs you nothing.

Three verbs, not interchangeable: `exec` (short arg-free snippets against a live session — it has **no
`[ARGS...]`** in its signature, so `colab exec -f src/train.py --config x.yaml` silently drops your
flags), `run` (one-shot provision → run → release, forwards args), and `console` (raw tmux TTY — **the
only one suited to long training runs**).

```bash
colab console -s boltu
# cd /content/boltU && git pull --ff-only
# nohup python src/train.py --config configs/base.yaml --resume > logs/train.out 2>&1 &
# Ctrl-b d to detach, then Ctrl-d. Training keeps running.
```

**`colab stop -s boltu` is not optional housekeeping** — compute units bill for as long as the VM is
alive, and the keep-alive daemon is *designed* to keep it alive.

### Colab persistence

Drive's 15 GB free quota is the real constraint, and checkpoints are ~1.5 GB each at Tier C. Never train
reading directly from Drive and never write checkpoints straight to Drive mid-step: stage data to
`/content` at session start, write checkpoints locally, then sync in a background thread. Add
`scripts/prune_drive.sh` that keeps the newest 2 checkpoints plus `ckpt_best.pt` and prints free space,
and run it *before* launching training so you fail on quota at minute one rather than hour six.

The background Drive copy is the highest-risk silent failure on this path: mounts disconnect and API
rate limits kill the thread invisibly. Use a single-worker `ThreadPoolExecutor`, call `.result(timeout=0)`
on the previous future before submitting the next so worker exceptions surface on the main thread, verify
each copy by size and mtime, and halt training on the second consecutive failure rather than producing
checkpoints that never leave the VM. Do not rely on atomic rename — the Drive FUSE mount is not
POSIX-compliant and `os.replace` gives no atomicity guarantee there.

Live monitoring is easier here than on Kaggle, since the metrics CSV is pullable mid-run:

```bash
# scripts/watch.sh
while true; do
  colab download -s boltu logs/metrics.csv /tmp/metrics.csv 2>/dev/null
  python scripts/dashboard.py /tmp/metrics.csv
  sleep 30
done
```

If you build `scripts/dashboard.py`: **drop the final CSV row if its field count doesn't match the
header** — downloads taken mid-append produce a torn last line, and that's the most likely corruption
you'll hit.

Skip TensorBoard and W&B either way. TensorBoard means dual-logging when CSV is already the source of
truth. W&B is low-effort if an external service is acceptable, but it puts a network call inside your
training process.

---

## Master checklist

- [ ] **P0** Tier chosen, token budget calculated, `base.yaml` filled in
- [ ] **P1** Repo scaffolded, config parses, **pushed to GitHub** (the Kaggle sync channel)
- [ ] **P2** Phone-verified Kaggle account; GPU + Internet enabled; `kaggle` CLI authed; `GH_TOKEN` in Kaggle Secrets; clone cell works; `/kaggle/working` space assert passes
- [ ] **P3** Data tokenized to uint16 shards, `decode(encode(s)) == s` asserted, val holdout cut on document boundaries, shards uploaded as a private dataset and readable at `/kaggle/input/`
- [ ] **P4** Model builds, param count matches target, forward pass shape correct
- [ ] **P5** Smoke test passes; initial loss ≈ 10.8; **kill-and-resume verified**; **`--max-hours` verified** (set it to 0.05 and confirm a clean exit 0 with a saved checkpoint)
- [ ] **P5b** Data in RAM (or memmap justified); dataset push failures logged loudly; grad_accum asserted
- [ ] **P5b+** *(Tier B/C only)* `telemetry.py` logging MFU and `num_alloc_retries`; `expandable_segments` set; DDP `grad_accum // world_size` verified
- [ ] **P6** *(Tier B/C only)* `session_start.sh` written; checkpoint dataset versioning tested end-to-end; first long batch job completed
- [ ] **P7** Loss curves plotted; samples read as coherent English
- [ ] **P8** UI streaming from the trained checkpoint
- [ ] **P9** *(optional)* instruction fine-tune

---

## 13. Fast reference: failure modes

| Symptom | Likely cause |
|---|---|
| Loss stuck near 10.8 | LR is zero (warmup misconfigured) or labels aren't shifted by one |
| Loss → NaN | fp16 overflow on T4/P100 (missing/misused GradScaler), or LR too high |
| CUDA OOM | reduce micro-batch and raise grad_accum to compensate; enable gradient checkpointing only as a last resort |
| Confusing DNS error on dataset download or git clone | `enable_internet: false` in `kernel-metadata.json` |
| `FileNotFoundError` on `/kaggle/input/boltU-...` | Kaggle lowercased your slug — it's `boltu-` (§2) |
| Kernel dies at 12 h with nothing saved | `--max-hours` not set or set too high; output is not retained on a hard kill |
| `No space left on device` mid-run | `/kaggle/working` is ~20 GB and you're staging a checkpoint zip in it (§3.4) |
| Throughput drops between sessions | you got P100 instead of T4 ×2 — check `get_device_name()` |
| Resumed run's loss spikes | optimizer or scheduler state not restored — only model weights were loaded |
| Effective batch silently doubled on 2×T4 | `grad_accum_steps` not divided by `world_size` (§11.7) |
| Generation repeats one phrase | undertrained, or temperature too low / top-k too small |
| **Loss looks fine (~3.2) but generations are mush** | see below |

**Diagnosing "good loss, bad output."** This is the most common late-stage confusion, and it's usually
not the model. Work through it in this order: (1) does the same prompt with the same seed produce
*identical* output? If not, you have a sampler bug. (2) Is temperature ≤0.5 or top-k ≤10? That produces
repetitive loops regardless of model quality — try 0.9 / top-k 50. (3) Does the prompt end mid-word or
with trailing whitespace? Tokenization boundary artifacts derail generation badly. (4) Is your
inference-time tokenizer the *same* encoding used in `data_prep.py`? A mismatch gives fluent-looking
nonsense — the §4 round-trip assert is what catches this before you waste a session. (5) Only after all
four: check tokens seen against the 20:1 budget — at 3.2 loss on a 10× under-trained model, mush is
simply the correct output.
