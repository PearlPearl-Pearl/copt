"""
Plot train and validation loss curves from two Lightning metrics.csv files
on a single axes so architectures can be directly compared.

Usage:
    python plot_losses.py \
        --csv1 results/<run_gcon>/lightning_logs/version_X/metrics.csv \
        --csv2 results/<run_hybridconv>/lightning_logs/version_X/metrics.csv \
        --label1 gcon \
        --label2 hybridconv \
        --out loss_curves.png
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_curves(csv_path):
    df = pd.read_csv(csv_path)

    train = (
        df[["epoch", "loss/train"]]
        .dropna(subset=["loss/train"])
        .groupby("epoch")["loss/train"]
        .mean()
        .reset_index()
        .rename(columns={"loss/train": "train_loss"})
    )
    val = (
        df[["epoch", "loss/valid"]]
        .dropna(subset=["loss/valid"])
        .groupby("epoch")["loss/valid"]
        .mean()
        .reset_index()
        .rename(columns={"loss/valid": "val_loss"})
    )
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv1",   required=True)
    parser.add_argument("--csv2",   required=True)
    parser.add_argument("--label1", default="run1")
    parser.add_argument("--label2", default="run2")
    parser.add_argument("--out",    default="loss_curves.png")
    args = parser.parse_args()

    train1, val1 = load_curves(args.csv1)
    train2, val2 = load_curves(args.csv2)

    colors = {args.label1: "#1f77b4", args.label2: "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for train, val, label in [
        (train1, val1, args.label1),
        (train2, val2, args.label2),
    ]:
        c = colors[label]
        ax.plot(train["epoch"], train["train_loss"],
                color=c, linestyle="--", linewidth=1.5,
                label=f"{label} — train")
        ax.plot(val["epoch"], val["val_loss"],
                color=c, linestyle="-",  linewidth=2.0,
                marker="o", markersize=4,
                label=f"{label} — val")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.set_title(f"{args.label1}  vs  {args.label2} — training curves",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
