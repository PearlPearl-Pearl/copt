"""
plot_gp_k3_hard_greedy.py

Loads the hard SBM k=3 test set, runs the greedy balanced k-partition
heuristic on graph_idx, and plots Input Graph + Greedy solution side by side.

Usage:
    python plot_gp_k3_hard_greedy.py
    python plot_gp_k3_hard_greedy.py --graph_idx 3
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops

COLORS = ["#4878CF", "#D65F5F", "#6ACC65"]


def load_test_graph(cfg_path, graph_idx):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from graphgym.patches import create_loader

    set_cfg(cfg)
    cfg.train.mode = None

    class _Args:
        cfg_file = cfg_path
        opts = []

    load_cfg(cfg, _Args())
    cfg.dataset.split_index = 0

    loaders = create_loader()
    test_graphs = []
    for batch in loaders[-1]:
        test_graphs.extend(batch.to_data_list())

    return test_graphs[graph_idx]


def greedy_partition(data, k=3):
    """
    Greedy balanced k-partition from utils/metrics.py:
    process nodes in descending degree order, assign each to the
    non-full partition with the most already-assigned neighbours.
    """
    n = data.num_nodes
    ei = remove_self_loops(data.edge_index)[0].cpu().numpy()

    neighbors = [[] for _ in range(n)]
    for src, dst in zip(ei[0], ei[1]):
        neighbors[src].append(dst)

    degrees = np.array([len(neighbors[v]) for v in range(n)])
    order = np.argsort(-degrees)

    labels = -np.ones(n, dtype=int)
    count = [0] * k
    cap = (n + k - 1) // k

    for v in order:
        edges_to = [0] * k
        for u in neighbors[v]:
            if labels[u] >= 0:
                edges_to[labels[u]] += 1

        available = [p for p in range(k) if count[p] < cap]
        p = max(available, key=lambda p: edges_to[p])
        labels[v] = p
        count[p] += 1

    return labels


def cut_fraction(data, labels):
    ei = remove_self_loops(data.edge_index)[0].numpy()
    total = ei.shape[1] // 2
    cut   = int((labels[ei[0]] != labels[ei[1]]).sum()) // 2
    return cut / total if total > 0 else 0.0


def draw_input(ax, G, pos, title):
    nx.draw_networkx_nodes(G, pos, node_color="#95a5a6", node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")


def draw_partition(ax, G, pos, labels, title, cf, k):
    node_colors = [COLORS[l] for l in labels]
    cut_edges  = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    same_edges = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same_edges, alpha=0.25, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                           edge_color="crimson", width=1.5, style="dashed", alpha=0.7, ax=ax)

    sizes = "  ".join(f"P{p}={(labels==p).sum()}" for p in range(k))
    ax.set_title(f"{title}\ncut fraction = {cf:.4f}  |  {sizes}", fontsize=11, pad=8)
    handles = [mpatches.Patch(facecolor=COLORS[p], label=f"Partition {p}") for p in range(k)]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    cfg_path = "configs/benchmarks/gp/gp_sbm_small_balanced_k3_hard_gcon.yaml"
    k = 3

    print(f"Loading test graph {args.graph_idx} from hard SBM dataset...")
    data = load_test_graph(cfg_path, args.graph_idx)
    n = data.num_nodes

    ei = remove_self_loops(data.edge_index)[0].numpy()
    mask = ei[0] < ei[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0][mask], ei[1][mask]))
    pos = nx.spring_layout(G, seed=42)

    print("Running greedy partition...")
    greedy_labels = greedy_partition(data, k=k)
    cf = cut_fraction(data, greedy_labels)
    print(f"  cut fraction = {cf:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"GP k=3 Hard (p_in=0.107, p_out=0.0467, n=75) — test graph {args.graph_idx}  "
        f"({n} nodes,  {G.number_of_edges()} edges)",
        fontsize=13,
    )

    draw_input(axes[0], G, pos, f"Input Graph\n{n} nodes,  {G.number_of_edges()} edges")
    draw_partition(axes[1], G, pos, greedy_labels, "Greedy Balanced Partition", cf, k)

    plt.tight_layout()
    out = f"gp_k3_hard_greedy_idx{args.graph_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
