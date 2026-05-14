#!/usr/bin/env bash
# Runs MIS pipeline for gcon and hybridconv (10 epochs each),
# prints per-epoch losses, then saves a combined loss plot.

set -e

CONDA_ENV="main_paper_env"
CFG_GCON="configs/benchmarks/mis/mis_rb_small_gcon.yaml"
CFG_HYBRID="configs/benchmarks/mis/mis_rb_small_hybridconv.yaml"
EPOCHS=10
PLOT_OUT="mis_loss_curves.png"

# ── activate conda env ────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── helper: find latest metrics.csv under a lightning_logs dir ───────────────
latest_csv() {
    local logs_dir="$1"
    local latest_ver
    latest_ver=$(ls "$logs_dir" | grep "^version_" | sort -V | tail -1)
    echo "$logs_dir/$latest_ver/metrics.csv"
}

# ── helper: print epoch-level losses from a metrics.csv ──────────────────────
print_losses() {
    local label="$1"
    local csv="$2"
    echo ""
    echo "=== $label ==="
    python3 - "$csv" <<'EOF'
import sys, pandas as pd

df = pd.read_csv(sys.argv[1])

train = (
    df[["epoch", "loss/train"]].dropna(subset=["loss/train"])
    .groupby("epoch")["loss/train"].mean()
)
val = (
    df[["epoch", "loss/valid"]].dropna(subset=["loss/valid"])
    .groupby("epoch")["loss/valid"].mean()
)
combined = pd.concat([train.rename("train"), val.rename("val")], axis=1)
print(combined.to_string(float_format=lambda x: f"{x:.6f}"))
EOF
}

# ── run gcon ──────────────────────────────────────────────────────────────────
LOG_GCON="run_mis_gcon.log"
echo ">>> Running gcon for $EPOCHS epochs (logging to $LOG_GCON)..."
python main.py --cfg "$CFG_GCON" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG_GCON"

GCON_LOGS="results/mis_rb_small_gcon-mis_rb_small_gcon/lightning_logs"
GCON_CSV=$(latest_csv "$GCON_LOGS")
echo "gcon results: $GCON_CSV"

# ── run hybridconv ────────────────────────────────────────────────────────────
LOG_HYBRID="run_mis_hybridconv.log"
echo ""
echo ">>> Running hybridconv for $EPOCHS epochs (logging to $LOG_HYBRID)..."
python main.py --cfg "$CFG_HYBRID" optim.max_epoch "$EPOCHS" 2>&1 | tee "$LOG_HYBRID"

HYBRID_LOGS="results/mis_rb_small_hybridconv-mis_rb_small_hybridconv/lightning_logs"
HYBRID_CSV=$(latest_csv "$HYBRID_LOGS")
echo "hybridconv results: $HYBRID_CSV"

# ── print losses ──────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────"
echo "  Per-epoch losses (train | val)"
echo "────────────────────────────────────────────"

print_losses "gcon" "$GCON_CSV"
print_losses "hybridconv" "$HYBRID_CSV"

# ── plot ──────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Plotting loss curves..."
python plot_losses.py \
    --csv1 "$GCON_CSV" \
    --csv2 "$HYBRID_CSV" \
    --label1 gcon \
    --label2 hybridconv \
    --out "$PLOT_OUT"

echo ""
echo "Done. Plot saved to $PLOT_OUT"
