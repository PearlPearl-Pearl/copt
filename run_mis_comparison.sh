#!/usr/bin/env bash
# Full MIS comparison pipeline: trains gcon and hybridconv, then produces
# a single loss-curve plot and a final-metrics summary you can write about.
#
# Usage (local):  ./run_mis_comparison.sh
# Usage (Colab):  bash run_mis_comparison.sh

set -e

CONDA_ENV="main_paper_env"
CFG_GCON="configs/benchmarks/mis/mis_rb_small_gcon.yaml"
CFG_HYBRID="configs/benchmarks/mis/mis_rb_small_hybridconv.yaml"
EPOCHS=100
PLOT_OUT="mis_loss_curves.png"
LOG_GCON="run_mis_gcon.log"
LOG_HYBRID="run_mis_hybridconv.log"

# ── activate conda env (skipped automatically on Colab / bare Python envs) ───
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# ── helper: latest metrics.csv under a lightning_logs dir ────────────────────
latest_csv() {
    local logs_dir="$1"
    local latest_ver
    latest_ver=$(ls "$logs_dir" | grep "^version_" | sort -V | tail -1)
    echo "$logs_dir/$latest_ver/metrics.csv"
}

# ── run gcon ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo " [1/4]  Training gcon   ($EPOCHS epochs)  →  $LOG_GCON"
echo "================================================================"
python main.py --cfg "$CFG_GCON" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG_GCON"
GCON_CSV=$(latest_csv "results/mis_rb_small_gcon-mis_rb_small_gcon/lightning_logs")

# ── run hybridconv ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [2/4]  Training hybridconv   ($EPOCHS epochs)  →  $LOG_HYBRID"
echo "================================================================"
python main.py --cfg "$CFG_HYBRID" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG_HYBRID"
HYBRID_CSV=$(latest_csv "results/mis_rb_small_hybridconv-mis_rb_small_hybridconv/lightning_logs")

# ── loss plot ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [3/4]  Plotting loss curves  →  $PLOT_OUT"
echo "================================================================"
python plot_losses.py \
    --csv1   "$GCON_CSV"   --label1 gcon \
    --csv2   "$HYBRID_CSV" --label2 hybridconv \
    --out    "$PLOT_OUT"

# ── final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [4/4]  Final results summary"
echo "================================================================"
python3 - "$GCON_CSV" "$HYBRID_CSV" <<'EOF'
import sys, pandas as pd

labels = ["gcon", "hybridconv"]

for label, path in zip(labels, sys.argv[1:]):
    df = pd.read_csv(path)

    best_epoch_val = (
        df[["epoch", "loss/valid"]].dropna(subset=["loss/valid"])
        .groupby("epoch")["loss/valid"].mean()
    )
    best_loss_epoch = int(best_epoch_val.idxmin())
    best_loss_val   = best_epoch_val.min()

    mis_val = (
        df[["epoch", "size/valid"]].dropna(subset=["size/valid"])
        .groupby("epoch")["size/valid"].mean()
    )
    greedy_val = (
        df[["epoch", "greedy_size/valid"]].dropna(subset=["greedy_size/valid"])
        .groupby("epoch")["greedy_size/valid"].mean()
    )

    best_mis    = mis_val.max()
    best_greedy = greedy_val.mean()  # greedy is deterministic; mean = constant

    print(f"\n{'─'*48}")
    print(f"  Architecture : {label}")
    print(f"{'─'*48}")
    print(f"  Best val loss      : {best_loss_val:.4f}  (epoch {best_loss_epoch})")
    print(f"  Best GNN IS size   : {best_mis:.2f}")
    print(f"  Greedy IS size     : {best_greedy:.2f}")
    ratio = best_mis / best_greedy if best_greedy > 0 else float('nan')
    print(f"  GNN / Greedy ratio : {ratio:.3f}")

    print(f"\n  Per-epoch loss (train | val):")
    train = (
        df[["epoch", "loss/train"]].dropna(subset=["loss/train"])
        .groupby("epoch")["loss/train"].mean().rename("train")
    )
    val = (
        df[["epoch", "loss/valid"]].dropna(subset=["loss/valid"])
        .groupby("epoch")["loss/valid"].mean().rename("val")
    )
    print(pd.concat([train, val], axis=1).to_string(
        float_format=lambda x: f"{x:.4f}"))

print(f"\n{'═'*48}")
print("  Done. Check mis_loss_curves.png for the plot.")
print(f"{'═'*48}\n")
EOF
