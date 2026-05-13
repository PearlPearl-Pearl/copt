"""
Quick test: generate a 6-node SBM graph, run spectral partitioning, visualise.
Run from the copt/ root directory:
    python test_spectral_partition.py
"""

import matplotlib
matplotlib.use("Agg")   # works without a display server
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from utils.sbm import stochastic_block_model
from utils.spectral import spectral_partition, normalised_cut


# ── 1. Generate a 6-node SBM (2 communities of 3) ──────────────────────────

N, K = 10, 3
P_IN, P_OUT = 0.9, 0.1
SEED = 7          # change this to try different realisations

A, true_labels = stochastic_block_model(N, K, P_IN, P_OUT, seed=SEED)
g = nx.from_numpy_array(A)

print("Adjacency matrix:")
print(A.astype(int))
print(f"\nGround-truth labels: {true_labels}")
print(f"Edges: {list(g.edges())}")


# ── 2. Run spectral partitioning heuristic ─────────────────────────────────

pred_labels = spectral_partition(A, k=K)
ncut = normalised_cut(A, pred_labels, K)

print(f"\nPredicted labels:    {pred_labels}")
print(f"Normalised cut:      {ncut:.4f}")

from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(true_labels, pred_labels)
print(f"ARI: {ari:.4f}")
# acc = partition_accuracy(true_labels, pred_labels, K)
# print(f"Partition accuracy:  {acc:.2%}")


# ── 3. Visualise ────────────────────────────────────────────────────────────

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
pos = nx.spring_layout(g, seed=SEED)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(
    f"6-node SBM  (p_in={P_IN}, p_out={P_OUT})  —  Spectral Partitioning",
    fontsize=13, fontweight="bold"
)

for ax, labels, title in zip(
    axes,
    [true_labels, pred_labels],
    ["Ground-truth communities", f"Spectral partition  (NCut={ncut:.3f})"]
):
    node_colours = [PALETTE[l % len(PALETTE)] for l in labels]
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, width=1.5, edge_color="grey")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colours,
                           node_size=600, linewidths=1.5,
                           edgecolors="white")
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white",
                            font_size=11, font_weight="bold")
    legend = [
        mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=f"Block {i}") for i in range(K)
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.tight_layout()
out_path = "spectral_partition_test.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()
