"""
plot_gp_k3_hard_results.py

Same as plot_gp_k3_results.py but targets the hard-instance result directories.

Usage:
    python plot_gp_k3_hard_results.py
    python plot_gp_k3_hard_results.py --graph_idx 3
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
import pandas as pd
import torch
from torch_geometric.utils import remove_self_loops

COLORS = ["#4878CF", "#D65F5F", "#6ACC65"]


def latest_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return os.path.abspath(max(hits, key=os.path.getmtime))


def epoch_mean(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()


def plot_loss_curves(gcon_csv, hybrid_csv):
    dfs = {}
    if gcon_csv:   dfs["gcon"]   = pd.read_csv(gcon_csv)
    if hybrid_csv: dfs["hybrid"] = pd.read_csv(hybrid_csv)

    colors = {"gcon": "#1f77b4", "hybrid": "#d62728"}
    labels = {"gcon": "GCON", "hybrid": "ScatteringClique"}

    for col, ylabel, title, fname in [
        ("loss/train", "Training loss",   "GP k=3 Hard — Training Loss",   "gp_k3_hard_train_loss.png"),
        ("loss/valid", "Validation loss", "GP k=3 Hard — Validation Loss", "gp_k3_hard_val_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in dfs.items():
            series = epoch_mean(df, col)
            if series.empty:
                print(f"  [warn] '{col}' not found for {name}")
                continue
            ax.plot(series.index, series.values,
                    label=labels[name], color=colors[name],
                    linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved → {fname}")


def get_gp_solution(ckpt_path, cfg_path, graph_idx):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from graphgym.patches import create_loader
    from torch_geometric.data import Batch
    from modules.architecture.copt_module import COPTModule

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

    data = test_graphs[graph_idx]
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

    logits = getattr(batch, "logits", None)
    if logits is None:
        logits = batch.x
    logits = logits.squeeze().cpu().numpy()
    labels = np.argmax(logits, axis=-1).astype(int)

    ei = remove_self_loops(data.edge_index)[0].numpy()
    mask = ei[0] < ei[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0][mask], ei[1][mask]))

    return data, G, labels


def spectral_partition(data, k=3):
    from utils.spectral import spectral_partition as _sp
    n  = data.num_nodes
    ei = remove_self_loops(data.edge_index)[0].numpy()
    A  = np.zeros((n, n), dtype=np.float32)
    A[ei[0], ei[1]] = 1.0
    return _sp(A, k)


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


def visualise_combined(gcon_ckpt, gcon_cfg, hybrid_ckpt, hybrid_cfg, graph_idx, k=3):
    print("  Running GCON inference...")
    data, G, gcon_labels = get_gp_solution(gcon_ckpt, gcon_cfg, graph_idx)
    n = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=42)

    print("  Running ScatteringClique inference...")
    _, _, hybrid_labels = get_gp_solution(hybrid_ckpt, hybrid_cfg, graph_idx)

    print("  Computing spectral baseline...")
    spectral_labels = spectral_partition(data, k=k)

    gcon_cf     = cut_fraction(data, gcon_labels)
    hybrid_cf   = cut_fraction(data, hybrid_labels)
    spectral_cf = cut_fraction(data, spectral_labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"GP k=3 Hard (p_in=0.107, p_out=0.0467, n=75) — test graph {graph_idx}  "
        f"({n} nodes,  {G.number_of_edges()} edges)",
        fontsize=13, y=1.01,
    )

    draw_input(axes[0][0], G, pos, f"Input Graph\n{n} nodes,  {G.number_of_edges()} edges")
    draw_partition(axes[0][1], G, pos, spectral_labels, "Spectral Baseline", spectral_cf, k)
    draw_partition(axes[1][0], G, pos, gcon_labels,     "GCON",              gcon_cf,     k)
    draw_partition(axes[1][1], G, pos, hybrid_labels,   "ScatteringClique",  hybrid_cf,   k)

    plt.tight_layout()
    out = "gp_k3_hard_solutions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    gcon_csv    = latest_file("results/gp_sbm_small_balanced_k3_hard_gcon-*/**/metrics.csv")
    hybrid_csv  = latest_file("results/gp_sbm_small_balanced_k3_hard_hybridconv-*/**/metrics.csv")
    gcon_ckpt   = latest_file("results/gp_sbm_small_balanced_k3_hard_gcon-*/**/*.ckpt")
    hybrid_ckpt = latest_file("results/gp_sbm_small_balanced_k3_hard_hybridconv-*/**/*.ckpt")

    gcon_cfg   = "configs/benchmarks/gp/gp_sbm_small_balanced_k3_hard_gcon.yaml"
    hybrid_cfg = "configs/benchmarks/gp/gp_sbm_small_balanced_k3_hard_hybridconv.yaml"

    print("── Loss curves ──────────────────────────────────────────────")
    plot_loss_curves(gcon_csv, hybrid_csv)

    print("\n── GP solution visualisation ────────────────────────────────")
    if gcon_ckpt and hybrid_ckpt:
        print(f"  gcon      : {gcon_ckpt}")
        print(f"  hybridconv: {hybrid_ckpt}")
        visualise_combined(gcon_ckpt, gcon_cfg, hybrid_ckpt, hybrid_cfg, args.graph_idx, k=3)
    else:
        missing = []
        if not gcon_ckpt:   missing.append("gcon checkpoint")
        if not hybrid_ckpt: missing.append("hybridconv checkpoint")
        print(f"  [warn] missing: {', '.join(missing)} — skipping solution plot")

    print("\nDone.")


if __name__ == "__main__":
    main()
