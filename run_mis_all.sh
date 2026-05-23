#!/bin/bash
set -e

echo "=========================================="
echo " MIS Linear Loss  —  GCON"
echo "=========================================="
python main.py --cfg configs/benchmarks/mis/mis_rb_small_gcon.yaml \
    optim.max_epoch 100 \
    train.ckpt_period 1 \
    2>&1 | tee run_mis_linear_gcon.log

echo ""
echo "=========================================="
echo " MIS Linear Loss  —  ScatteringClique"
echo "=========================================="
python main.py --cfg configs/benchmarks/mis/mis_rb_small_hybridconv.yaml \
    optim.max_epoch 100 \
    train.ckpt_period 1 \
    2>&1 | tee run_mis_linear_hybridconv.log

echo ""
echo "=========================================="
echo " MIS QUBO Loss  —  GCON"
echo "=========================================="
python main.py --cfg configs/benchmarks/mis/mis_rb_small_qubo_gcon.yaml \
    optim.max_epoch 100 \
    train.ckpt_period 1 \
    2>&1 | tee run_mis_qubo_gcon.log

echo ""
echo "=========================================="
echo " MIS QUBO Loss  —  ScatteringClique"
echo "=========================================="
python main.py --cfg configs/benchmarks/mis/mis_rb_small_qubo_hybridconv.yaml \
    optim.max_epoch 100 \
    train.ckpt_period 1 \
    2>&1 | tee run_mis_qubo_hybridconv.log

echo ""
echo "=========================================="
echo " Plotting all results"
echo "=========================================="
python plot_mis_all_results.py

echo ""
echo "Done."
