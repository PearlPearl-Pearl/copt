#!/bin/bash
set -e

cd /content/copt

echo "=========================================="
echo " Weighted GP — GatedGCN (10 epochs sanity)"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_weighted_gatedgcn.yaml \
    optim.max_epoch 10 \
    2>&1 | tee run_gp_weighted_sanity.log

echo ""
echo "=========================================="
echo " Plotting solutions"
echo "=========================================="
python plot_gp_weighted_sanity.py

echo ""
echo "Done."
