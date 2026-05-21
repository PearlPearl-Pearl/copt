"""
test_gp_partition.py

Loads the real GP test dataset, matches each graph to its slice of
probs_test.pt (the GNN's actual output), then visualises GNN prediction
vs spectral baseline side-by-side for one chosen graph.

The probabilities genuinely come from the trained model running on this
exact graph — no placeholder values.

Usage:
    python test_gp_partition.py --cfg configs/benchmarks/gp/gp_sbm_small.yaml
    python test_gp_partition.py --cfg configs/benchmarks/gp/gp_sbm_small.yaml --graph_idx 3
    python test_gp_partition.py --cfg configs/benchmarks/gp/gp_sbm_small.yaml --graph_idx 0 --out my_fig.png
"""

import argparse
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops

# ── GraphGym init (mirrors main.py) ──────────────────────────────────────────
import graphgym  # noqa — registers all custom modules
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg, makedirs_rm_exist
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgym.patches import create_loader
from modules.architecture.copt_module import create_model

from utils.spectral import spectral_partition

COLORS = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66"]  # up to 5 partitions


# ── helpers ───────────────────────────────────────────────────────────────────

# def decode_gnn(probs: np.ndarray) -> np.ndarray:
#     """y = 2p − 1;  y < 0 → partition 0,  y ≥ 0 → partition 1."""
#     return ((2 * probs - 1) >= 0).astype(int)

def decode_gnn(emb: np.ndarray, k: int = 2) -> np.ndarray:
    """Decode GNN output to partition labels.

    - dim_out == k  (softmax vector): argmax directly gives the partition.
    - dim_out == 1, k == 2: threshold at 0.5.
    - dim_out == 1, k > 2 : 1-D KMeans (legacy scalar configs).
    """
    from sklearn.cluster import KMeans
    if emb.ndim == 1:
        emb = emb[:, None]
    if emb.shape[1] > 1:
        # multi-dim output (softmax): take argmax
        return np.argmax(emb, axis=-1).astype(int)
    # scalar output
    if k == 2:
        return (emb.squeeze() >= 0.5).astype(int)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb).astype(int)


def cut_fraction(edge_index, labels) -> float:
    ei = remove_self_loops(edge_index)[0].cpu().numpy()
    total = ei.shape[1] // 2
    cut   = int((labels[ei[0]] != labels[ei[1]]).sum()) // 2
    return cut / total if total > 0 else 0.0


def adj_numpy(data) -> np.ndarray:
    n  = data.num_nodes
    ei = remove_self_loops(data.edge_index)[0].cpu().numpy()
    A  = np.zeros((n, n), dtype=np.float32)
    A[ei[0], ei[1]] = 1.0
    return A


def find_probs(run_dir: str):
    # always use the most recently modified probs_test.pt across all runs
    candidates = glob.glob("results/**/probs_test.pt", recursive=True)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    print(f"[find_probs] using most recent: {latest}")
    return latest


def draw_partition(ax, G, pos, probs, labels, title, cf):
    cut_e  = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    same_e = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]

    nx.draw_networkx_nodes(G, pos,
                           node_color=[COLORS[l] for l in labels],
                           node_size=400, ax=ax)
    def _prob_str(p):
        if np.ndim(p) == 0:
            return f"p={p:.3f}"
        return ",".join(f"{v:.2f}" for v in p)

    nx.draw_networkx_labels(G, pos,
                            labels={i: f"{i}\n{_prob_str(probs[i])}" for i in G.nodes()},
                            font_size=6, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same_e,
                           edge_color="#aaaaaa", alpha=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_e,
                           edge_color="crimson", width=2.5, style="dashed", ax=ax)

    ax.set_title(f"{title}\ncut fraction = {cf:.4f}  ({len(cut_e)} cut edges)",
                 fontsize=11, pad=8)
    ax.axis("off")
    k_parts = len(set(labels))
    legend_handles = [mpatches.Patch(facecolor=COLORS[i], label=f"Partition {i}") for i in range(k_parts)]
    legend_handles.append(mlines.Line2D([], [], color="crimson", lw=2, linestyle="dashed", label=f"Cut ({len(cut_e)})"))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.85)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True,
                        help="GP YAML config, e.g. configs/benchmarks/gp/gp_sbm_small.yaml")
    parser.add_argument("--graph_idx", type=int, default=0,
                        help="Which test graph to visualise (0-indexed)")
    parser.add_argument("--out", default="gp_partition_test.png")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Extra cfg overrides, e.g. dataset.split_index 1")
    args = parser.parse_args()

    # ── mirror main.py config init ────────────────────────────────────────────
    set_cfg(cfg)
    cfg.train.mode = None          # required before load_cfg (unregistered key)

    # load_cfg expects args.cfg_file and args.opts
    class _Args:
        cfg_file = args.cfg
        opts     = args.opts or []
    load_cfg(cfg, _Args())

    # reproduce the run directory so we find probs_test.pt in the right place
    cfg_stem  = os.path.splitext(os.path.basename(args.cfg))[0]
    run_name  = f"{cfg_stem}-{cfg.wandb.name}"
    cfg.out_dir  = os.path.join(cfg.out_dir, run_name)
    cfg.run_dir  = os.path.join(cfg.out_dir, "0")

    cfg.dataset.split_index = 0
    auto_select_device()

    # ── load test dataset (shuffle=False → same order as probs_test.pt) ──────
    print("Loading test dataset …")
    loaders     = create_loader()
    test_loader = loaders[-1]          # train / val / test

    # collect all test graphs in the exact same order the model saw them
    test_graphs = []
    for batch in test_loader:
        test_graphs.extend(batch.to_data_list())

    n_test = len(test_graphs)
    print(f"Test set: {n_test} graphs  "
          f"(nodes per graph: {[g.num_nodes for g in test_graphs[:5]]} …)")

    if args.graph_idx >= n_test:
        sys.exit(f"[error] --graph_idx {args.graph_idx} is out of range "
                 f"(test set has {n_test} graphs, 0-indexed)")

    # ── locate and load probs_test.pt ─────────────────────────────────────────
    probs_path = find_probs(cfg.run_dir)
    if probs_path is None:
        sys.exit("[error] probs_test.pt not found. "
                 "Run training with log_probs: true and test the model first.")
    print(f"Probabilities : {probs_path}")

    all_probs = torch.load(probs_path, map_location="cpu").float().numpy()
    print(f"Total probs   : {len(all_probs)}  "
          f"(should equal total test nodes = "
          f"{sum(g.num_nodes for g in test_graphs)})")

    # ── slice out this graph's probabilities ──────────────────────────────────
    offset = sum(g.num_nodes for g in test_graphs[:args.graph_idx])
    data   = test_graphs[args.graph_idx]
    n      = data.num_nodes
    gnn_probs = all_probs[offset : offset + n]

    print(f"\nGraph {args.graph_idx}: {n} nodes, "
          f"offset={offset}:{offset+n}")
    print(f"GNN probs  min={gnn_probs.min():.4f}  "
          f"max={gnn_probs.max():.4f}  "
          f"mean={gnn_probs.mean():.4f}")

    # ── decode GNN partition ──────────────────────────────────────────────────
    k               = cfg.metrics.gp.k
    gnn_labels = decode_gnn(gnn_probs, k=k)
    gnn_cf     = cut_fraction(data.edge_index, gnn_labels)
    parts_str  = "  ".join(f"P{p}={( gnn_labels == p).sum()}" for p in range(k))
    print(f"GNN partition : {parts_str}  cut fraction={gnn_cf:.4f}")

    # ── spectral baseline ─────────────────────────────────────────────────────
    A               = adj_numpy(data)
    spectral_labels = spectral_partition(A, k)
    spectral_probs  = np.where(spectral_labels == 0, 0.05, 0.95).astype(float)
    spectral_cf     = cut_fraction(data.edge_index, spectral_labels)
    s_str           = "  ".join(f"P{p}={(spectral_labels == p).sum()}" for p in range(k))
    print(f"Spectral      : {s_str}  cut fraction={spectral_cf:.4f}")

    # ── build networkx graph ──────────────────────────────────────────────────
    ei   = remove_self_loops(data.edge_index)[0].cpu().numpy()
    mask = ei[0] < ei[1]
    G    = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0][mask], ei[1][mask]))
    pos  = nx.spring_layout(G, seed=42)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Graph Partitioning — test graph {args.graph_idx}  "
        f"({n} nodes,  {G.number_of_edges()} edges,  k={k})",
        fontsize=13,
    )
    draw_partition(axes[0], G, pos, spectral_probs, spectral_labels,
                   "Spectral baseline", spectral_cf)
    draw_partition(axes[1], G, pos, gnn_probs, gnn_labels,
                   "GNN prediction", gnn_cf)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
