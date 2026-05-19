#!/usr/bin/env bash
# GP balanced-loss run: trains gcon on SBM small for 100 epochs using
# gp_loss_balanced, then plots loss curves, prints a summary, and
# visualises the partition on a sample test graph.
#
# Usage (local):  ./run_gp_balanced.sh
# Usage (Colab):  bash run_gp_balanced.sh

set -e

CONDA_ENV="main_paper_env"
CFG="configs/benchmarks/gp/gp_sbm_small_balanced.yaml"
EPOCHS=100
LOG="gp_balanced.log"
PLOT_LOSS="gp_balanced_curves.png"
PLOT_PART="gp_balanced_partition.png"
GRAPH_IDX=0        # which test graph to visualise

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

# ── [1/4] train ───────────────────────────────────────────────────────────────
echo "================================================================"
echo " [1/4]  Training gp_loss_balanced  ($EPOCHS epochs)  →  $LOG"
echo "================================================================"
python main.py --cfg "$CFG" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG"
BALANCED_CSV=$(latest_csv "results/gp_sbm_small_balanced-gp_sbm_small_balanced")

# ── [2/4] loss curves ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [2/4]  Plotting loss curves  →  $PLOT_LOSS"
echo "================================================================"
python plot_gp_balanced.py --csv "$BALANCED_CSV" --out "$PLOT_LOSS"

# ── [3/4] partition visualisation ────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [3/4]  Partition visualisation  →  $PLOT_PART"
echo "================================================================"
python test_gp_partition.py \
    --cfg      "$CFG" \
    --graph_idx "$GRAPH_IDX" \
    --out      "$PLOT_PART"

# ── [4/4] final summary ───────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " [4/4]  Final results summary"
echo "================================================================"
python3 - "$BALANCED_CSV" <<'EOF'
import sys
import pandas as pd

path = sys.argv[1]
df   = pd.read_csv(path)

def col_mean(col):
    if col not in df.columns:
        return None
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()

val_loss = col_mean("loss/valid")
gnn_cut  = col_mean("gnn_cut/valid")
spec_cut = col_mean("spectral_cut/valid")
grd_cut  = col_mean("greedy_cut/valid")

best_epoch = int(val_loss.idxmin())   if val_loss is not None else -1
best_val   = val_loss.min()           if val_loss is not None else float("nan")
best_gnn   = gnn_cut.min()            if gnn_cut  is not None else float("nan")
spec_val   = spec_cut.mean()          if spec_cut is not None else float("nan")
grd_val    = grd_cut.mean()           if grd_cut  is not None else float("nan")

print(f"\n{'─'*52}")
print(f"  Loss function     : gp_loss_balanced (Laplacian + balance)")
print(f"{'─'*52}")
print(f"  Best val loss     : {best_val:.4f}  (epoch {best_epoch})")
print(f"  Best GNN cut frac : {best_gnn:.4f}   (lower = better)")
print(f"  Spectral cut frac : {spec_val:.4f}   (constant baseline)")
print(f"  Greedy cut frac   : {grd_val:.4f}   (constant baseline)")
if spec_val > 0:
    print(f"  GNN / Spectral    : {best_gnn/spec_val:.3f}   (<1 = GNN beats spectral)")

print(f"\n  Per-epoch (train loss | val loss | gnn_cut):")
frames = {}
tl = col_mean("loss/train")
if tl       is not None: frames["train_loss"] = tl
if val_loss is not None: frames["val_loss"]   = val_loss
if gnn_cut  is not None: frames["gnn_cut"]    = gnn_cut
if frames:
    print(pd.DataFrame(frames).to_string(float_format=lambda x: f"{x:.4f}"))

print(f"\n{'═'*52}")
print(f"  Plots saved:")
print(f"    loss curves  →  gp_balanced_curves.png")
print(f"    partition    →  gp_balanced_partition.png")
print(f"{'═'*52}\n")
EOF
