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
echo "Done."
