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
import sys
import pandas as pd

labels = ["gcon", "hybridconv"]

def col_mean(df, col):
    """Per-epoch mean of a column; returns Series or None if column absent."""
    if col not in df.columns:
        return None
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()

for label, path in zip(labels, sys.argv[1:]):
    df = pd.read_csv(path)

    # ── best validation loss ──────────────────────────────────────────────────
    val_loss = col_mean(df, "loss/valid")
    best_epoch = int(val_loss.idxmin()) if val_loss is not None else -1
    best_val   = val_loss.min() if val_loss is not None else float("nan")

    # ── cut fractions ─────────────────────────────────────────────────────────
    gnn_cut  = col_mean(df, "gnn_cut/valid")
    spec_cut = col_mean(df, "spectral_cut/valid")
    grd_cut  = col_mean(df, "greedy_cut/valid")

    best_gnn  = gnn_cut.min()  if gnn_cut  is not None else float("nan")
    spec_val  = spec_cut.mean() if spec_cut is not None else float("nan")
    grd_val   = grd_cut.mean()  if grd_cut  is not None else float("nan")

    print(f"\n{'─'*52}")
    print(f"  Architecture      : {label}")
    print(f"{'─'*52}")
    print(f"  Best val loss     : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Best GNN cut frac : {best_gnn:.4f}   (lower = better)")
    print(f"  Spectral cut frac : {spec_val:.4f}   (constant baseline)")
    print(f"  Greedy cut frac   : {grd_val:.4f}   (constant baseline)")
    if best_gnn == best_gnn and spec_val == spec_val and spec_val > 0:
        print(f"  GNN / Spectral    : {best_gnn/spec_val:.3f}   (<1 = GNN beats spectral)")

    print(f"\n  Per-epoch (train loss | val loss | gnn_cut):")
    train_loss = col_mean(df, "loss/train")
    frames = {}
    if train_loss is not None: frames["train_loss"] = train_loss
    if val_loss   is not None: frames["val_loss"]   = val_loss
    if gnn_cut    is not None: frames["gnn_cut"]    = gnn_cut
    if frames:
        print(pd.DataFrame(frames).to_string(float_format=lambda x: f"{x:.4f}"))

print(f"\n{'═'*52}")
print("  Done. Check gp_loss_curves.png for the plot.")
print(f"{'═'*52}\n")
EOF
