"""
plot_gp_weighted_sanity.py

Loads the weighted GP test set and plots:
  - Input graph (coloured by edge weight: blue=+1, red=-1)
  - GatedGCN solution (argmax of 2-dim softmax output)

Usage:
    python plot_gp_weighted_sanity.py
    python plot_gp_weighted_sanity.py --graph_idx 3
"""

import argparse
import glob
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

PART_COLORS = ["#4878CF", "#D65F5F"]   # partition colours
NODE_DEFAULT = "#95a5a6"


def latest_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return os.path.abspath(max(hits, key=os.path.getmtime))


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


def run_gnn(ckpt_path, cfg_path, data):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.data import Batch
    from modules.architecture.copt_module import COPTModule

    set_cfg(cfg)
    cfg.train.mode = None

    class _Args:
        cfg_file = cfg_path
        opts = []

    load_cfg(cfg, _Args())
    cfg.dataset.split_index = 0

    n = data.num_nodes
    cfg.share.dim_in  = data.x.shape[1]
    cfg.share.dim_out = cfg.dim_out
    model = COPTModule.load_from_checkpoint(
        ckpt_path, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=cfg
    )
    model.eval()
    device = next(model.parameters()).device

    data.batch = torch.zeros(n, dtype=torch.long)
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        model(batch)

    logits = getattr(batch, "logits", batch.x)
    logits = logits.squeeze().cpu().numpy()
    # scalar sigmoid output: threshold at 0.5
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return (logits.squeeze() >= 0.5).astype(int)
    return np.argmax(logits, axis=-1).astype(int)


def cut_fraction(data, labels):
    ei = remove_self_loops(data.edge_index)[0].numpy()
    total = ei.shape[1] // 2
    cut   = int((labels[ei[0]] != labels[ei[1]]).sum()) // 2
    return cut / total if total > 0 else 0.0


def build_graph(data):
    n = data.num_nodes
    ei = remove_self_loops(data.edge_index)[0].numpy()
    mask = ei[0] < ei[1]
    edges = list(zip(ei[0][mask], ei[1][mask]))

    # recover per-edge weights (directed, take first direction)
    weights = {}
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        w = data.edge_weight.squeeze().numpy()
    elif data.edge_attr is not None:
        w = data.edge_attr.squeeze().numpy()
    else:
        w = np.ones(ei.shape[1])

    for idx, (u, v) in enumerate(zip(ei[0][mask], ei[1][mask])):
        # find the index in the directed edge list
        orig_idx = np.where((ei[0] == u) & (ei[1] == v))[0]
        weights[(u, v)] = float(w[orig_idx[0]]) if len(orig_idx) > 0 else 1.0

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G, weights


def draw_input(ax, G, pos, weights, title):
    pos_edges = [(u, v) for (u, v) in G.edges() if weights.get((u, v), 1.0) >= 0]
    neg_edges = [(u, v) for (u, v) in G.edges() if weights.get((u, v), 1.0) <  0]

    nx.draw_networkx_nodes(G, pos, node_color=NODE_DEFAULT, node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges,
                           edge_color="#2196F3", alpha=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges,
                           edge_color="#E53935", alpha=0.5, style="dashed", ax=ax)

    handles = [
        mpatches.Patch(color="#2196F3", label="+1 (ferromagnetic)"),
        mpatches.Patch(color="#E53935", label="−1 (anti-ferromagnetic)"),
    ]
    if neg_edges:
        ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.8)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")


def draw_partition(ax, G, pos, labels, title, cf, weights):
    node_colors = [PART_COLORS[l] for l in labels]
    cut_edges  = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    same_edges = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same_edges, alpha=0.25, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                           edge_color="crimson", width=1.5, style="dashed", alpha=0.7, ax=ax)

    sizes = "  ".join(f"P{p}={(labels==p).sum()}" for p in range(2))
    ax.set_title(f"{title}\ncut fraction = {cf:.4f}  |  {sizes}", fontsize=11, pad=8)
    handles = [mpatches.Patch(facecolor=PART_COLORS[p], label=f"Partition {p}") for p in range(2)]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    cfg_path  = "configs/benchmarks/gp/gp_weighted_gatedgcn.yaml"
    ckpt_path = latest_file("results/gp_weighted_gatedgcn-*/**/*.ckpt")

    print(f"Loading test graph {args.graph_idx}...")
    data = load_test_graph(cfg_path, args.graph_idx)
    n = data.num_nodes

    G, weights = build_graph(data)
    pos = nx.spring_layout(G, seed=42)

    has_neg = any(v < 0 for v in weights.values())
    graph_type_label = "spin-glass" if has_neg else "unweighted"

    if ckpt_path:
        print(f"Running GatedGCN inference  ({ckpt_path})...")
        gnn_labels = run_gnn(ckpt_path, cfg_path, data)
        gnn_cf = cut_fraction(data, gnn_labels)
        print(f"  GNN cut fraction = {gnn_cf:.4f}")
        ncols = 2
    else:
        print("  [warn] no checkpoint found — plotting input graph only")
        ncols = 1

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        f"Weighted GP k=2 — test graph {args.graph_idx}  "
        f"({n} nodes, {G.number_of_edges()} edges, {graph_type_label})",
        fontsize=13,
    )

    draw_input(axes[0], G, pos, weights,
               f"Input Graph\n{n} nodes,  {G.number_of_edges()} edges")

    if ncols == 2:
        draw_partition(axes[1], G, pos, gnn_labels,
                       "GatedGCN (weighted GP loss)", gnn_cf, weights)

    plt.tight_layout()
    out = f"gp_weighted_sanity_idx{args.graph_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
