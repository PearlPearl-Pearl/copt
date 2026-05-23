#!/bin/bash
set -e

echo "=========================================="
echo " GP Balanced k=3 HARD  —  GCON"
echo " p_in=0.3  p_out=0.2  (ratio 1.5)"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_sbm_small_balanced_k3_hard_gcon.yaml \
    2>&1 | tee run_gp_balanced_k3_hard_gcon.log

echo ""
echo "=========================================="
echo " GP Balanced k=3 HARD  —  ScatteringClique"
echo " p_in=0.3  p_out=0.2  (ratio 1.5)"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_sbm_small_balanced_k3_hard_hybridconv.yaml \
    2>&1 | tee run_gp_balanced_k3_hard_hybridconv.log

echo ""
echo "=========================================="
echo " Plotting results"
echo "=========================================="
python plot_gp_k3_hard_results.py

echo ""
echo "=========================================="
echo " Downloading results"
echo "=========================================="
python - <<'EOF'
import glob, os, zipfile
from google.colab import files

with zipfile.ZipFile("gp_k3_hard_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["run_gp_balanced_k3_hard_gcon.log", "run_gp_balanced_k3_hard_hybridconv.log"]:
        if os.path.exists(f): zf.write(f)
    for f in ["gp_k3_hard_train_loss.png", "gp_k3_hard_val_loss.png", "gp_k3_hard_solutions.png"]:
        if os.path.exists(f): zf.write(f)
    for f in glob.glob("results/gp_sbm_small_balanced_k3_hard*/**/*.ckpt", recursive=True):
        zf.write(f)
    for f in glob.glob("results/gp_sbm_small_balanced_k3_hard*/**/metrics.csv", recursive=True):
        zf.write(f)

print("Contents:")
for name in zf.namelist():
    print(f"  {name}")

files.download("gp_k3_hard_results.zip")
EOF

echo ""
echo "Done."
