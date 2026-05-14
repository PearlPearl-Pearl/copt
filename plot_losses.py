"""
Plot train and validation loss curves from two Lightning metrics.csv files.

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

    # --- train loss: logged per step; average over all steps in each epoch ---
    train = (
        df[["epoch", "loss/train"]]
        .dropna(subset=["loss/train"])
        .groupby("epoch")["loss/train"]
        .mean()
        .reset_index()
        .rename(columns={"loss/train": "train_loss"})
    )

    # --- val loss: logged once per epoch ---
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
    parser.add_argument("--csv1", required=True, help="metrics.csv for run 1")
    parser.add_argument("--csv2", required=True, help="metrics.csv for run 2")
    parser.add_argument("--label1", default="run1", help="Legend label for run 1")
    parser.add_argument("--label2", default="run2", help="Legend label for run 2")
    parser.add_argument("--out", default="loss_curves.png", help="Output file path")
    args = parser.parse_args()

    train1, val1 = load_curves(args.csv1)
    train2, val2 = load_curves(args.csv2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    colors = {"1": "#1f77b4", "2": "#d62728"}

    for ax, (train, val, label, color) in zip(
        axes,
        [
            (train1, val1, args.label1, colors["1"]),
            (train2, val2, args.label2, colors["2"]),
        ],
    ):
        ax.plot(train["epoch"], train["train_loss"],
                color=color, linestyle="--", linewidth=1.5, label="train loss")
        ax.plot(val["epoch"], val["val_loss"],
                color=color, linestyle="-", linewidth=2.0, label="val loss")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{args.label1}  vs  {args.label2} — training curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
