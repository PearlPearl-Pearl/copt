#!/bin/bash
set -e

echo "=========================================="
echo " GP Balanced k=3  —  GCON"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_sbm_small_balanced_k3.yaml \
    optim.max_epoch 100 \
    2>&1 | tee run_gp_balanced_k3_gcon.log

echo ""
echo "=========================================="
echo " GP Balanced k=3  —  ScatteringClique"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_sbm_small_balanced_k3_hybridconv.yaml \
    2>&1 | tee run_gp_balanced_k3_hybridconv.log

echo ""
echo "=========================================="
echo " Plotting results"
echo "=========================================="
python plot_gp_k3_results.py

echo ""
echo "Done."
