"""Read the training metrics CSV and plot train/val loss on a log-x axis."""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="checkpoints/metrics.csv")
    ap.add_argument("--out", default="plots/loss.png")
    args = ap.parse_args()

    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            steps.append(step + 1)  # +1: log-x axis can't plot step 0
            train_loss.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(step + 1)
                val_loss.append(float(row["val_loss"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="train", alpha=0.6, linewidth=1)
    plt.plot(val_steps, val_loss, label="val", marker="o", markersize=3)
    plt.xscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("boltU training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
