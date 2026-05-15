#!/usr/bin/env bash
# GP comparison: gcon vs hybridconv on SBM small, 100 epochs.
# Produces two log files + a combined loss-curve plot + a final summary.
#
# Usage (local):  ./run_gp_comparison.sh
# Usage (Colab):  bash run_gp_comparison.sh

set -e

CONDA_ENV="main_paper_env"
CFG_GCON="configs/benchmarks/gp/gp_sbm_small.yaml"
CFG_HYBRID="configs/benchmarks/gp/gp_sbm_small_hybridconv.yaml"
EPOCHS=100
LOG_GCON="gp_gcon.log"
LOG_HYBRID="gp_hybridconv.log"
PLOT_OUT="gp_loss_curves.png"

# ── activate conda env (no-op on Colab / bare envs) ──────────────────────────
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# ── helper: most recently modified metrics.csv under a results dir ────────────
latest_csv() {
    find "$1" -name "metrics.csv" -printf "%T@ %p\n" 2>/dev/null \
        | sort -n | tail -1 | awk '{print $2}'
}

# ── run gcon ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo " [1/4]  Training gcon   ($EPOCHS epochs)  →  $LOG_GCON"
echo "================================================================"
python main.py --cfg "$CFG_GCON" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG_GCON"
GCON_CSV=$(latest_csv "results/gp_sbm_small-gp_sbm_small")

# ── run hybridconv ────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [2/4]  Training hybridconv   ($EPOCHS epochs)  →  $LOG_HYBRID"
echo "================================================================"
python main.py --cfg "$CFG_HYBRID" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG_HYBRID"
HYBRID_CSV=$(latest_csv "results/gp_sbm_small_hybridconv-gp_sbm_small_hybridconv")

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
import sys, math
import pandas as pd

labels = ["gcon", "hybridconv"]

for label, path in zip(labels, sys.argv[1:]):
    df = pd.read_csv(path)

    val = (
        df[["epoch", "loss/valid", "gnn_cut/valid", "spectral_cut/valid", "greedy_cut/valid"]]
        .dropna(subset=["loss/valid"])
        .groupby("epoch")
        .mean()
    )
    test = (
        df[["epoch", "gnn_cut/test", "spectral_cut/test", "greedy_cut/test"]]
        .dropna(subset=["gnn_cut/test"])
        .groupby("epoch")
        .mean()
    )

    best_epoch = int(val["gnn_cut/valid"].idxmin())
    best_val   = val["gnn_cut/valid"].min()
    spec_val   = val["spectral_cut/valid"].iloc[0]
    greedy_val = val["greedy_cut/valid"].iloc[0]

    print(f"\n{'─'*52}")
    print(f"  Architecture   : {label}")
    print(f"{'─'*52}")
    print(f"  Best val GNN cut   : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Spectral cut       : {spec_val:.4f}  (constant baseline)")
    print(f"  Greedy cut         : {greedy_val:.4f}  (constant baseline)")
    if not test.empty:
        last = test.iloc[-1]
        print(f"  Test GNN cut       : {last['gnn_cut/test']:.4f}")
        if not math.isnan(last.get("spectral_cut/test", float("nan"))):
            print(f"  Test Spectral cut  : {last['spectral_cut/test']:.4f}")
        if not math.isnan(last.get("greedy_cut/test", float("nan"))):
            print(f"  Test Greedy cut    : {last['greedy_cut/test']:.4f}")

    print(f"\n  Per-epoch val metrics (gnn_cut | spectral_cut | greedy_cut):")
    print(val[["gnn_cut/valid", "spectral_cut/valid", "greedy_cut/valid"]].to_string(
        float_format=lambda x: f"{x:.4f}"))

print(f"\n{'═'*52}")
print("  Done. Check gp_loss_curves.png for the plot.")
print(f"{'═'*52}\n")
EOF
