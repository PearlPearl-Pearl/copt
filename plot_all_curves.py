"""
plot_all_curves.py

Produces 4 separate plots from GP and MIS comparison runs:
    gp_train_loss.png   — GP   training loss:    gcon vs hybridconv
    gp_val_loss.png     — GP   validation loss:  gcon vs hybridconv
    mis_train_loss.png  — MIS  training loss:    gcon vs hybridconv
    mis_val_loss.png    — MIS  validation loss:  gcon vs hybridconv

CSV paths are auto-detected from the most recent run under results/.
Override any of them with the flags below.

Usage:
    python plot_all_curves.py
    python plot_all_curves.py --gp_gcon   path/to/metrics.csv
    python plot_all_curves.py --out_dir   my_plots/
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── colours ──────────────────────────────────────────────────────────────────
C_GCON   = "#1f77b4"   # blue
C_HYBRID = "#d62728"   # red


# ── helpers ───────────────────────────────────────────────────────────────────

def latest_csv(pattern):
    """Return the most recently modified metrics.csv matching a glob pattern."""
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return os.path.abspath(max(hits, key=os.path.getmtime))


def epoch_mean(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return (
        df[["epoch", col]]
        .dropna(subset=[col])
        .groupby("epoch")[col]
        .mean()
    )


def make_plot(csv_gcon, csv_hybrid, col, ylabel, title, out_path):
    """Plot one column (train or val loss) for gcon vs hybridconv."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for csv_path, label, color in [
        (csv_gcon,   "gcon",       C_GCON),
        (csv_hybrid, "hybridconv", C_HYBRID),
    ]:
        if csv_path is None or not os.path.exists(csv_path):
            print(f"  [warn] CSV not found for {label}, skipping.")
            continue
        df     = pd.read_csv(csv_path)
        series = epoch_mean(df, col)
        if series.empty:
            print(f"  [warn] column '{col}' not found in {csv_path}, skipping.")
            continue
        ax.plot(series.index, series.values,
                color=color, linewidth=2.0,
                marker="o", markersize=3,
                label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel,  fontsize=12)
    ax.set_title(title,    fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp_gcon",
        default=None,
        help="metrics.csv for GP gcon run (auto-detected if omitted)")
    parser.add_argument("--gp_hybrid",
        default=None,
        help="metrics.csv for GP hybridconv run (auto-detected if omitted)")
    parser.add_argument("--mis_gcon",
        default=None,
        help="metrics.csv for MIS gcon run (auto-detected if omitted)")
    parser.add_argument("--mis_hybrid",
        default=None,
        help="metrics.csv for MIS hybridconv run (auto-detected if omitted)")
    parser.add_argument("--out_dir",
        default=".",
        help="Directory to write the 4 PNGs (default: current dir)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── auto-detect CSVs ──────────────────────────────────────────────────────
    gp_gcon   = args.gp_gcon   or latest_csv("results/gp_sbm_small-gp_sbm_small/**/metrics.csv")
    gp_hybrid = args.gp_hybrid or latest_csv("results/gp_sbm_small_hybridconv-*/**/metrics.csv")
    mis_gcon  = args.mis_gcon  or latest_csv("results/mis_rb_small_gcon-*/**/metrics.csv")
    mis_hybrid= args.mis_hybrid or latest_csv("results/mis_rb_small_hybridconv-*/**/metrics.csv")

    print("CSVs detected:")
    print(f"  GP   gcon      : {gp_gcon}")
    print(f"  GP   hybridconv: {gp_hybrid}")
    print(f"  MIS  gcon      : {mis_gcon}")
    print(f"  MIS  hybridconv: {mis_hybrid}")
    print()

    # ── 4 plots ───────────────────────────────────────────────────────────────
    print("Generating plots…")

    make_plot(
        gp_gcon, gp_hybrid,
        col="loss/train",
        ylabel="Training loss",
        title="GP — Training Loss: gcon vs hybridconv",
        out_path=os.path.join(args.out_dir, "gp_train_loss.png"),
    )

    make_plot(
        gp_gcon, gp_hybrid,
        col="loss/valid",
        ylabel="Validation loss",
        title="GP — Validation Loss: gcon vs hybridconv",
        out_path=os.path.join(args.out_dir, "gp_val_loss.png"),
    )

    make_plot(
        mis_gcon, mis_hybrid,
        col="loss/train",
        ylabel="Training loss",
        title="MIS — Training Loss: gcon vs hybridconv",
        out_path=os.path.join(args.out_dir, "mis_train_loss.png"),
    )

    make_plot(
        mis_gcon, mis_hybrid,
        col="loss/valid",
        ylabel="Validation loss",
        title="MIS — Validation Loss: gcon vs hybridconv",
        out_path=os.path.join(args.out_dir, "mis_val_loss.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
