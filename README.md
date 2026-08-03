# boltU

A GPT-style language model trained from scratch on Kaggle's free T4 GPUs. Full design
rationale and phase-by-phase instructions live in [plans/boltu-plan.md](plans/boltu-plan.md).

**Current tier:** A (learning run) — 6 layer / 384 d_model / 6 head, ctx 512, ~30M params,
600M training tokens. Fits one ~4h Kaggle session. Change tiers by editing `configs/base.yaml`;
see the tier table in the plan (§1) for B/C numbers.

## Layout

```
configs/    model + training hyperparams (base.yaml = Tier A, tiny.yaml = smoke test)
data/       tokenized .bin shards (gitignored, built by src/data_prep.py)
kernel/     what gets pushed to Kaggle (kernel-metadata.json + run.ipynb launcher)
src/        data_prep, dataset, model, train, sample, plot_curves, app
checkpoints/ (gitignored)
```

## Credentials

```bash
cp .env.example .env   # fill in KAGGLE_USERNAME, KAGGLE_API_TOKEN, GH_TOKEN
set -a; source .env; set +a   # export into the shell before running kaggle CLI commands
```

Also add `GH_TOKEN` as a **Kaggle Secret** (notebook Add-ons → Secrets) — that's what
`kernel/run.ipynb` reads on Kaggle itself, `.env` is for local use only.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Tokenize data (once; ~2GB text for Tier A)
python src/data_prep.py --config configs/base.yaml

# 2. Sanity-check the model
python src/model.py --config configs/base.yaml

# 3. Smoke test the training loop (2 min, tiny config)
python src/train.py --config configs/tiny.yaml --max_steps 50

# 4. Real run (push to Kaggle as a batch job — see plans/boltu-plan.md §3.3)
kaggle kernels push -p ./kernel
```

## Before the real run

Verify kill-and-resume works (`plans/boltu-plan.md` calls this the single most important
test in the project):

```bash
python src/train.py --config configs/tiny.yaml   # start it, then Ctrl-C partway through
python src/train.py --config configs/tiny.yaml --resume   # loss should continue, no discontinuity
```
