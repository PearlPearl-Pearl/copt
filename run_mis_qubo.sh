#!/bin/bash
set -e

echo "=========================================="
echo " MIS QUBO  —  GCON"
echo "=========================================="
python main.py --cfg configs/benchmarks/mis/mis_rb_small_qubo_gcon.yaml \
    optim.max_epoch 10 \
    train.ckpt_period 1 \
    2>&1 | tee run_mis_qubo_gcon.log

echo ""
echo "=========================================="
echo " MIS QUBO  —  ScatteringClique"
echo "=========================================="
python main.py --cfg configs/benchmarks/mis/mis_rb_small_qubo_hybridconv.yaml \
    optim.max_epoch 10 \
    train.ckpt_period 1 \
    2>&1 | tee run_mis_qubo_hybridconv.log

echo ""
echo "=========================================="
echo " Evaluating against greedy heuristic"
echo "=========================================="
python plot_mis_qubo_results.py

echo ""
echo "Done."
