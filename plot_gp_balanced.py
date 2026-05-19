"""
plot_gp_balanced.py

Plots training curves from a gp_loss_balanced run.

Usage:
    python plot_gp_balanced.py                          # auto-finds latest run
    python plot_gp_balanced.py --csv path/to/metrics.csv
    python plot_gp_balanced.py --out my_plot.png
    python plot_gp_balanced.py --compare path/to/other/metrics.csv --label2 gp_loss
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def latest_csv(root="results"):
    hits = glob.glob(os.path.join(root, "*balanced*", "**", "metrics.csv"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(root, "**", "metrics.csv"), recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def epoch_mean(df, col):
    sub = df[["epoch", col]].dropna(subset=[col])
    return sub.groupby("epoch")[col].mean()


def plot_csv(ax_loss, ax_cut, df, label, linestyle="-"):
    train_loss = epoch_mean(df, "loss/train")
    val_loss   = epoch_mean(df, "loss/valid")
    gnn_cut    = epoch_mean(df, "gnn_cut/valid")
    spectral   = epoch_mean(df, "spectral_cut/valid")
    greedy     = epoch_mean(df, "greedy_cut/valid")

    # --- loss panel ---
    if not train_loss.empty:
        ax_loss.plot(train_loss.index, train_loss.values,
                     linestyle=linestyle, label=f"{label} train loss")
    if not val_loss.empty:
        ax_loss.plot(val_loss.index, val_loss.values,
                     linestyle="--", label=f"{label} val loss")

    # --- cut fraction panel ---
    if not gnn_cut.empty:
        ax_cut.plot(gnn_cut.index, gnn_cut.values,
                    linestyle=linestyle, label=f"{label} GNN cut")
    if not spectral.empty:
        ax_cut.plot(spectral.index, spectral.values,
                    linestyle=":", label=f"{label} spectral")
    if not greedy.empty:
        ax_cut.plot(greedy.index, greedy.values,
                    linestyle="-.", label=f"{label} greedy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      default=None,
                        help="Path to metrics.csv (auto-detected if omitted)")
    parser.add_argument("--label1",   default="gp_loss_balanced")
    parser.add_argument("--compare",  default=None,
                        help="Optional second metrics.csv for side-by-side comparison")
    parser.add_argument("--label2",   default="gp_loss")
    parser.add_argument("--out",      default="gp_balanced_curves.png")
    args = parser.parse_args()

    csv1 = args.csv or latest_csv()
    if csv1 is None:
        raise FileNotFoundError("No metrics.csv found. Pass --csv explicitly.")
    print(f"Primary CSV  : {csv1}")

    df1 = pd.read_csv(csv1)

    fig, (ax_loss, ax_cut) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GP Loss Balanced — training curves", fontsize=13)

    plot_csv(ax_loss, ax_cut, df1, args.label1, linestyle="-")

    if args.compare:
        print(f"Compare CSV  : {args.compare}")
        df2 = pd.read_csv(args.compare)
        plot_csv(ax_loss, ax_cut, df2, args.label2, linestyle="--")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_cut.set_xlabel("Epoch")
    ax_cut.set_ylabel("Cut fraction (lower = better)")
    ax_cut.set_title("Cut Fraction — GNN vs Baselines")
    ax_cut.legend(fontsize=8)
    ax_cut.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved → {args.out}")

    # --- text summary ---
    val_loss = epoch_mean(df1, "loss/valid")
    gnn_cut  = epoch_mean(df1, "gnn_cut/valid")
    spec_cut = epoch_mean(df1, "spectral_cut/valid")

    if not val_loss.empty:
        best_ep  = int(val_loss.idxmin())
        print(f"\n{'─'*44}")
        print(f"  Best val loss  : {val_loss.min():.4f}  (epoch {best_ep})")
    if not gnn_cut.empty:
        print(f"  Best GNN cut   : {gnn_cut.min():.4f}  (epoch {int(gnn_cut.idxmin())})")
    if not spec_cut.empty:
        print(f"  Spectral cut   : {spec_cut.mean():.4f}  (mean across epochs)")
    print(f"{'─'*44}\n")


if __name__ == "__main__":
    main()
