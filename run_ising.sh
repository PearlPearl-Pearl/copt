#!/bin/bash
set -e

cd /content/copt

echo "=========================================="
echo " Ising — Sanity Check (GCON, 5 epochs)"
echo "=========================================="
python main.py --cfg configs/benchmarks/ising/ising_mixed_sanity.yaml \
    2>&1 | tee run_ising_sanity.log

echo ""
echo "=========================================="
echo " Ising — GatedGCN + Ising Loss (100 epochs)"
echo "=========================================="
python main.py --cfg configs/benchmarks/ising/ising_mixed_gatedgcn.yaml \
    2>&1 | tee run_ising_gatedgcn.log

echo ""
echo "Done."
