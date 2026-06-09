"""
plot_gp_ising_weighted_sanity.py

Loads the spin-glass test set and plots:
  2x2 partition grid: Spectral | Greedy | GCON | ScatteringClique
  Separate files: training loss curve, validation loss curve

Usage:
    python plot_gp_ising_weighted_sanity.py
    python plot_gp_ising_weighted_sanity.py --graph_idx 3
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

PART_COLORS  = ["#4878CF", "#D65F5F"]
NODE_DEFAULT = "#95a5a6"

GCON_CFG    = "configs/benchmarks/gp/gp_ising_weighted_gcon.yaml"
HYBRID_CFG  = "configs/benchmarks/gp/gp_ising_weighted_hybridconv.yaml"
GCON_CKPT   = "results/gp_ising_weighted_gcon-*/**/*.ckpt"
HYBRID_CKPT = "results/gp_ising_weighted_hybridconv-*/**/*.ckpt"
GCON_CSV    = "results/gp_ising_weighted_gcon-*/**/metrics.csv"
HYBRID_CSV  = "results/gp_ising_weighted_hybridconv-*/**/metrics.csv"


def latest_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    return os.path.abspath(max(hits, key=os.path.getmtime)) if hits else None


def load_test_graph(graph_idx):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graphgym.config
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from graphgym.patches import create_loader

    set_cfg(cfg)
    cfg.train.mode = None

    class _A:
        cfg_file = GCON_CFG
        opts = []

    load_cfg(cfg, _A())
    cfg.dataset.split_index = 0

    loaders = create_loader()
    test_graphs = []
    for batch in loaders[-1]:
        test_graphs.extend(batch.to_data_list())
    return test_graphs[graph_idx]


def run_gnn(ckpt_path, cfg_path, data):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import graphgym.config
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.data import Batch
    from modules.architecture.copt_module import COPTModule

    set_cfg(cfg)
    cfg.train.mode = None

    class _A:
        cfg_file = cfg_path
        opts = []

    load_cfg(cfg, _A())
    cfg.dataset.split_index = 0

    n = data.num_nodes
    cfg.share.dim_in  = data.x.shape[1]
    cfg.share.dim_out = cfg.dim_out
    model = COPTModule.load_from_checkpoint(
        ckpt_path, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=cfg
    )
    model.eval()
    device = next(model.parameters()).device

    data2 = data.clone()
    data2.batch = torch.zeros(n, dtype=torch.long)
    batch = Batch.from_data_list([data2]).to(device)
    with torch.no_grad():
        model(batch)

    logits = getattr(batch, "logits", batch.x).squeeze().cpu().numpy()
    if logits.ndim > 1:
        return np.argmax(logits, axis=-1).astype(int)
    return (logits >= 0.5).astype(int)


def cut_fraction(data, labels):
    src = data.edge_index[0]
    dst = data.edge_index[1]
    w   = data.edge_attr.view(-1).float() if data.edge_attr is not None \
          else torch.ones(src.shape[0])
    labels_t = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    cut_mask     = labels_t[src] != labels_t[dst]
    weighted_cut = torch.abs(torch.sum(w[cut_mask]))
    total_weight = torch.sum(torch.abs(w))
    return (weighted_cut / total_weight).item() if total_weight > 0 else 0.0


def build_graph(data):
    n  = data.num_nodes
    ei = remove_self_loops(data.edge_index)[0].numpy()
    w  = data.edge_attr.view(-1).numpy() if data.edge_attr is not None else np.ones(ei.shape[1])
    mask = ei[0] < ei[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    weights = {}
    for idx, (u, v) in enumerate(zip(ei[0][mask], ei[1][mask])):
        orig = np.where((ei[0] == u) & (ei[1] == v))[0]
        weights[(int(u), int(v))] = float(w[orig[0]]) if len(orig) else 1.0
        G.add_edge(int(u), int(v))
    return G, weights


def draw_partition(ax, G, pos, labels, title, cf):
    node_colors = [PART_COLORS[l] for l in labels]
    cut_e  = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    same_e = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same_e, alpha=0.25, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_e,
                           edge_color="crimson", width=1.5, style="dashed", alpha=0.7, ax=ax)
    sizes = "  ".join(f"P{p}={(labels==p).sum()}" for p in range(2))
    ax.set_title(f"{title}\nweighted cut = {cf:.4f}  |  {sizes}", fontsize=11, pad=8)
    handles = [mpatches.Patch(facecolor=PART_COLORS[p], label=f"Partition {p}") for p in range(2)]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.8)
    ax.axis("off")


def spectral_labels(data):
    from utils.spectral import spectral_partition
    n  = data.num_nodes
    ei = remove_self_loops(data.edge_index)[0].numpy()
    A  = np.zeros((n, n), dtype=np.float32)
    A[ei[0], ei[1]] = 1.0
    return spectral_partition(A, k=2)


def greedy_labels(data):
    from utils.metrics import _gp_greedy_labels
    return _gp_greedy_labels(data, k=2)


def epoch_mean(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()


def plot_loss_curves(gcon_csv, hybrid_csv):
    dfs = {}
    if gcon_csv:   dfs["gcon"]   = pd.read_csv(gcon_csv)
    if hybrid_csv: dfs["hybrid"] = pd.read_csv(hybrid_csv)

    if not dfs:
        print("  [warn] no metrics CSVs found — skipping loss curves")
        return

    colors = {"gcon": "#1f77b4", "hybrid": "#d62728"}
    labels = {"gcon": "GCON", "hybrid": "ScatteringClique"}

    for col, ylabel, title, fname in [
        ("loss/train", "Training loss",   "Weighted GP — Training Loss",   "gp_ising_weighted_train_loss.png"),
        ("loss/valid", "Validation loss", "Weighted GP — Validation Loss", "gp_ising_weighted_val_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for name, df in dfs.items():
            series = epoch_mean(df, col)
            if series.empty:
                print(f"  [warn] '{col}' not found for {name}")
                continue
            ax.plot(series.index, series.values,
                    label=labels[name], color=colors[name],
                    linewidth=2, marker="o", markersize=3)
            plotted = True
        if not plotted:
            plt.close()
            continue
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved → {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    gcon_ckpt   = latest_file(GCON_CKPT)
    hybrid_ckpt = latest_file(HYBRID_CKPT)
    gcon_csv    = latest_file(GCON_CSV)
    hybrid_csv  = latest_file(HYBRID_CSV)

    print(f"Loading test graph {args.graph_idx}...")
    data = load_test_graph(args.graph_idx)
    G, weights = build_graph(data)
    pos = nx.spring_layout(G, seed=42)
    n   = data.num_nodes

    print("Computing spectral partition...")
    spec_labels = spectral_labels(data)
    spec_cf     = cut_fraction(data, spec_labels)
    print(f"  spectral  weighted cut = {spec_cf:.4f}")

    print("Computing greedy partition...")
    grdy_labels = greedy_labels(data)
    grdy_cf     = cut_fraction(data, grdy_labels)
    print(f"  greedy    weighted cut = {grdy_cf:.4f}")

    gcon_labs = hybrid_labs = None
    gcon_cf   = hybrid_cf   = None

    if gcon_ckpt:
        print(f"Running GCON inference ({gcon_ckpt})...")
        gcon_labs = run_gnn(gcon_ckpt, GCON_CFG, data)
        gcon_cf   = cut_fraction(data, gcon_labs)
        print(f"  GCON      weighted cut = {gcon_cf:.4f}")
    else:
        print("  [warn] no GCON checkpoint found")

    if hybrid_ckpt:
        print(f"Running ScatteringClique inference ({hybrid_ckpt})...")
        hybrid_labs = run_gnn(hybrid_ckpt, HYBRID_CFG, data)
        hybrid_cf   = cut_fraction(data, hybrid_labs)
        print(f"  ScatteringClique weighted cut = {hybrid_cf:.4f}")
    else:
        print("  [warn] no ScatteringClique checkpoint found")

    # --- 2×2 solutions plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Weighted GP — 5-regular spin-glass (±1) — test graph {args.graph_idx}  "
        f"({n} nodes,  {G.number_of_edges()} edges)",
        fontsize=13, y=1.01,
    )

    draw_partition(axes[0][0], G, pos, spec_labels, "Spectral Bisection (baseline)", spec_cf)
    draw_partition(axes[0][1], G, pos, grdy_labels, "Greedy Balanced (baseline)",    grdy_cf)

    if gcon_labs is not None:
        draw_partition(axes[1][0], G, pos, gcon_labs,
                       "GCON — signed weighted GP loss", gcon_cf)
    else:
        axes[1][0].set_title("GCON\n(no checkpoint)", fontsize=11)
        axes[1][0].axis("off")

    if hybrid_labs is not None:
        draw_partition(axes[1][1], G, pos, hybrid_labs,
                       "ScatteringClique — signed weighted GP loss", hybrid_cf)
    else:
        axes[1][1].set_title("ScatteringClique\n(no checkpoint)", fontsize=11)
        axes[1][1].axis("off")

    plt.tight_layout()
    out = f"gp_ising_weighted_sanity_idx{args.graph_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")

    # --- loss curves ---
    print("\n── Loss curves ─────────────────────────────────────────────")
    plot_loss_curves(gcon_csv, hybrid_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
