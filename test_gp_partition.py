"""
test_gp_partition.py

Builds a simple 5-node graph, applies:
  - spectral partitioning  (heuristic baseline)
  - GNN decoder            (probabilities loaded from probs_test.pt)

Visualises both side-by-side with cut edges in red.

Usage:
    python test_gp_partition.py                  # default 5-node graph
    python test_gp_partition.py --n 10 --seed 7  # different size/seed
    python test_gp_partition.py --out my_fig.png
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from utils.sbm import stochastic_block_model
from utils.spectral import spectral_partition

COLORS = ["#4878CF", "#D65F5F"]   # blue = partition 0,  red = partition 1


# ── helpers ───────────────────────────────────────────────────────────────────

def find_probs_file():
    """Return the most recently modified probs_test.pt under results/."""
    pattern = os.path.join("results", "**", "probs_test.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def decode_gnn(probs: np.ndarray) -> np.ndarray:
    """y = 2p − 1;  y < 0 → partition 0,  y ≥ 0 → partition 1."""
    return ((2 * probs - 1) >= 0).astype(int)


def cut_size(edges, labels) -> int:
    return sum(1 for u, v in edges if labels[u] != labels[v])


# ── visualisation ─────────────────────────────────────────────────────────────

def draw_partition(ax, G, pos, probs, labels, title):
    cut  = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    same = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]
    n_cut = len(cut)

    nx.draw_networkx_nodes(G, pos,
                           node_color=[COLORS[l] for l in labels],
                           node_size=700, ax=ax)
    # label each node with its index and probability
    node_labels = {i: f"{i}\np={probs[i]:.2f}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels,
                            font_size=7, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same,
                           edge_color="#aaaaaa", alpha=0.6, width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut,
                           edge_color="crimson", width=2.5,
                           style="dashed", ax=ax)

    ax.set_title(f"{title}\ncut size = {n_cut}", fontsize=11, pad=10)
    ax.axis("off")

    legend = [
        mpatches.Patch(facecolor=COLORS[0], label="Partition 0"),
        mpatches.Patch(facecolor=COLORS[1], label="Partition 1"),
        mlines.Line2D([], [], color="crimson", linewidth=2,
                      linestyle="dashed", label=f"Cut edges ({n_cut})"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.85)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,   default=5,   help="Number of nodes")
    parser.add_argument("--k",    type=int,   default=2,   help="Number of partitions")
    parser.add_argument("--p_in", type=float, default=0.8, help="Intra-community edge prob")
    parser.add_argument("--p_out",type=float, default=0.1, help="Inter-community edge prob")
    parser.add_argument("--seed", type=int,   default=42,  help="Random seed")
    parser.add_argument("--out",  type=str,   default="gp_partition_test.png")
    args = parser.parse_args()

    # ── generate graph ────────────────────────────────────────────────────────
    A, true_labels = stochastic_block_model(
        args.n, args.k, args.p_in, args.p_out, seed=args.seed)

    rows, cols = np.where(np.triu(A, k=1) > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))

    G = nx.Graph()
    G.add_nodes_from(range(args.n))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=args.seed)

    print(f"Graph: N={args.n}, K={args.k}, "
          f"p_in={args.p_in}, p_out={args.p_out}, seed={args.seed}")
    print(f"Edges : {edges}")
    print(f"Ground-truth partitions: "
          + "  ".join(f"P{i}={np.where(true_labels==i)[0].tolist()}"
                      for i in range(args.k)))

    # ── spectral baseline ─────────────────────────────────────────────────────
    spectral_labels = spectral_partition(A, args.k)
    spectral_probs  = np.where(spectral_labels == 0, 0.1, 0.9).astype(float)
    s_cut = cut_size(edges, spectral_labels)

    print(f"\nSpectral  → partitions: "
          + "  ".join(f"P{i}={np.where(spectral_labels==i)[0].tolist()}"
                      for i in range(args.k)))
    print(f"Spectral cut size : {s_cut}")

    # ── GNN probabilities — load from probs_test.pt ───────────────────────────
    probs_path = find_probs_file()
    if probs_path is None:
        print("\n[warn] No probs_test.pt found under results/ — using random probs.")
        rng = np.random.default_rng(args.seed + 99)
        gnn_probs = rng.uniform(0, 1, size=args.n)
        probs_note = "random placeholder"
    else:
        all_probs = torch.load(probs_path, map_location="cpu").float().numpy()
        gnn_probs = all_probs[:args.n]
        probs_note = os.path.relpath(probs_path)
        print(f"\nLoaded GNN probs from : {probs_note}")
        print(f"  (taking first {args.n} of {len(all_probs)} values)")

    gnn_labels = decode_gnn(gnn_probs)
    g_cut = cut_size(edges, gnn_labels)

    print(f"\nGNN probs : {np.round(gnn_probs, 4).tolist()}")
    print(f"GNN y=2p-1: {np.round(2*gnn_probs-1, 4).tolist()}")
    print(f"GNN labels: {gnn_labels.tolist()}  "
          + "  ".join(f"P{i}={np.where(gnn_labels==i)[0].tolist()}"
                      for i in range(args.k)))
    print(f"GNN cut size       : {g_cut}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"Graph Partitioning — N={args.n} SBM  "
        f"(p_in={args.p_in}, p_out={args.p_out})",
        fontsize=13,
    )

    draw_partition(axes[0], G, pos, spectral_probs, spectral_labels,
                   "Spectral baseline")
    draw_partition(axes[1], G, pos, gnn_probs,      gnn_labels,
                   f"GNN  [{probs_note}]")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
