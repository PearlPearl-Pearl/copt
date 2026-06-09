#!/bin/bash
set -e

# cd /content/copt

echo "=========================================="
echo " Weighted GP — GCON (100 epochs)"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_ising_weighted_gcon.yaml \
    2>&1 | tee run_gp_ising_weighted_gcon.log

echo ""
echo "=========================================="
echo " Weighted GP — ScatteringClique (100 epochs)"
echo "=========================================="
python main.py --cfg configs/benchmarks/gp/gp_ising_weighted_hybridconv.yaml \
    2>&1 | tee run_gp_ising_weighted_hybridconv.log

echo ""
echo "=========================================="
echo " Plotting solutions and loss curves"
echo "=========================================="
python plot_gp_ising_weighted_sanity.py

echo ""
echo "Done."
