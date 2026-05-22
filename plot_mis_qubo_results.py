"""
plot_mis_qubo_results.py

1. Evaluates GCON-QUBO and ScatteringClique-QUBO on the full test set and
   compares IS sizes against the GMIN greedy heuristic.
2. Prints a summary table: mean IS size and approximation ratio per method.
3. Saves a 2x2 visualisation of one test graph:
      top-left    : Input graph
      top-right   : Greedy (GMIN)
      bottom-left : GCON QUBO
      bottom-right: ScatteringClique QUBO

Usage:
    python plot_mis_qubo_results.py
    python plot_mis_qubo_results.py --graph_idx 3
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


# ── auto-detect helpers ───────────────────────────────────────────────────────

def latest_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return os.path.abspath(max(hits, key=os.path.getmtime))


def epoch_mean(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()




# ── loss curve plots ──────────────────────────────────────────────────────────

def plot_loss_curves(gcon_csv, hybrid_csv):
    dfs = {}
    if gcon_csv:   dfs["gcon"]   = pd.read_csv(gcon_csv)
    if hybrid_csv: dfs["hybrid"] = pd.read_csv(hybrid_csv)

    colors = {"gcon": "#1f77b4", "hybrid": "#d62728"}
    labels = {"gcon": "GCON QUBO", "hybrid": "ScatteringClique QUBO"}

    for col, ylabel, title, fname in [
        ("loss/train", "Training loss",   "MIS QUBO — Training Loss",   "mis_qubo_train_loss.png"),
        ("loss/valid", "Validation loss", "MIS QUBO — Validation Loss", "mis_qubo_val_loss.png"),
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


# ── greedy baseline (GMIN) ───────────────────────────────────────────────────

def greedy_mis(G):
    """Repeatedly pick the minimum-degree node, add to IS, remove neighbours."""
    remaining = set(G.nodes())
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    selected = np.zeros(G.number_of_nodes(), dtype=bool)
    while remaining:
        node = min(remaining, key=lambda v: len(adj[v] & remaining))
        selected[node] = True
        remaining -= adj[node] & remaining
        remaining.discard(node)
    return selected


# ── GNN inference ─────────────────────────────────────────────────────────────

def load_model_and_testset(ckpt_path, cfg_path):
    """Returns (model, test_graphs, device)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from graphgym.patches import create_loader
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

    cfg.share.dim_in  = test_graphs[0].x.shape[1]
    cfg.share.dim_out = cfg.dim_out
    model = COPTModule.load_from_checkpoint(
        ckpt_path, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=cfg
    )
    model.eval()
    device = next(model.parameters()).device
    return model, test_graphs, device


def gnn_is(model, data, device):
    """Run inference on one graph, return boolean IS mask via greedy decoder."""
    from torch_geometric.data import Batch

    n = data.num_nodes
    data.batch = torch.zeros(n, dtype=torch.long)
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        model(batch)

    probs = batch.x.squeeze().cpu().numpy()

    ei = remove_self_loops(data.edge_index)[0].numpy()
    adj = {i: set() for i in range(n)}
    for s, d in zip(ei[0], ei[1]):
        adj[s].add(d); adj[d].add(s)

    order = np.argsort(probs)[::-1]
    selected = np.zeros(n, dtype=bool)
    blocked  = np.zeros(n, dtype=bool)
    for idx in order:
        if not blocked[idx]:
            selected[idx] = True
            for nb in adj[idx]:
                blocked[nb] = True

    return selected


def build_graph(data):
    ei = remove_self_loops(data.edge_index)[0].numpy()
    mask = ei[0] < ei[1]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(zip(ei[0][mask], ei[1][mask]))
    return G, ei


# ── evaluation across test set ────────────────────────────────────────────────

def evaluate(model, test_graphs, device, label):
    sizes = []
    for data in test_graphs:
        sel = gnn_is(model, data, device)
        sizes.append(sel.sum())
    arr = np.array(sizes, dtype=float)
    print(f"  {label:30s}  mean IS = {arr.mean():.2f}  ±{arr.std():.2f}  "
          f"min={arr.min():.0f}  max={arr.max():.0f}")
    return arr


# ── draw helpers ──────────────────────────────────────────────────────────────

def draw_input(ax, G, pos):
    nx.draw_networkx_nodes(G, pos, node_color="#95a5a6", node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title(f"Input Graph\n{G.number_of_nodes()} nodes,  "
                 f"{G.number_of_edges()} edges", fontsize=12)
    ax.axis("off")


def draw_is(ax, G, pos, selected, title):
    n = G.number_of_nodes()
    is_size  = selected.sum()
    bad      = [(u, v) for u, v in G.edges() if selected[u] and selected[v]]
    colors   = ["#e74c3c" if selected[i] else "#95a5a6" for i in range(n)]
    sizes    = [100 if selected[i] else 40 for i in range(n)]

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)
    if bad:
        nx.draw_networkx_edges(G, pos, edgelist=bad,
                               edge_color="black", width=2.5, ax=ax)

    handles = [
        mpatches.Patch(color="#e74c3c", label=f"In IS ({is_size})"),
        mpatches.Patch(color="#95a5a6", label=f"Not in IS ({n - is_size})"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower left", framealpha=0.85)
    ax.set_title(
        f"{title}\nIS size = {is_size} ({100*is_size/n:.1f}%)  "
        f"violations = {len(bad)}",
        fontsize=11,
    )
    ax.axis("off")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    gcon_ckpt   = latest_file("results/mis_rb_small_qubo_gcon*/**/*.ckpt")
    hybrid_ckpt = latest_file("results/mis_rb_small_qubo_hybridconv*/**/*.ckpt")
    gcon_csv    = latest_file("results/mis_rb_small_qubo_gcon*/**/metrics.csv")
    hybrid_csv  = latest_file("results/mis_rb_small_qubo_hybridconv*/**/metrics.csv")
    gcon_cfg    = "configs/benchmarks/mis/mis_rb_small_qubo_gcon.yaml"
    hybrid_cfg  = "configs/benchmarks/mis/mis_rb_small_qubo_hybridconv.yaml"

    print("── Loss curves ──────────────────────────────────────────────")
    plot_loss_curves(gcon_csv, hybrid_csv)

    if not gcon_ckpt or not hybrid_ckpt:
        missing = []
        if not gcon_ckpt:   missing.append("gcon checkpoint")
        if not hybrid_ckpt: missing.append("hybridconv checkpoint")
        sys.exit(f"[error] missing: {', '.join(missing)}")

    print(f"  gcon      : {gcon_ckpt}")
    print(f"  hybridconv: {hybrid_ckpt}")

    # ── load models ───────────────────────────────────────────────────────────
    print("\nLoading GCON model...")
    gcon_model, test_graphs, gcon_device = load_model_and_testset(gcon_ckpt, gcon_cfg)

    print("Loading ScatteringClique model...")
    hybrid_model, _, hybrid_device = load_model_and_testset(hybrid_ckpt, hybrid_cfg)

    # ── evaluate on full test set ─────────────────────────────────────────────
    print(f"\n── IS sizes across {len(test_graphs)} test graphs ──────────────────")

    greedy_sizes = np.array([
        greedy_mis(build_graph(d)[0]).sum() for d in test_graphs
    ], dtype=float)
    print(f"  {'Greedy (GMIN)':30s}  mean IS = {greedy_sizes.mean():.2f}  "
          f"±{greedy_sizes.std():.2f}  min={greedy_sizes.min():.0f}  "
          f"max={greedy_sizes.max():.0f}")

    gcon_sizes   = evaluate(gcon_model,   test_graphs, gcon_device,   "GCON QUBO")
    hybrid_sizes = evaluate(hybrid_model, test_graphs, hybrid_device, "ScatteringClique QUBO")

    print(f"\n── Approximation ratio vs GMIN ──────────────────────────────")
    print(f"  {'GCON QUBO':30s}  {(gcon_sizes / greedy_sizes).mean():.4f}")
    print(f"  {'ScatteringClique QUBO':30s}  {(hybrid_sizes / greedy_sizes).mean():.4f}")

    # ── 2x2 visualisation ─────────────────────────────────────────────────────
    print(f"\n── Visualising test graph {args.graph_idx} ───────────────────────")
    data = test_graphs[args.graph_idx]
    G, _ = build_graph(data)
    pos  = nx.spring_layout(G, seed=42)

    selected_greedy = greedy_mis(G)
    selected_gcon   = gnn_is(gcon_model,   data, gcon_device)
    selected_hybrid = gnn_is(hybrid_model, data, hybrid_device)

    n = G.number_of_nodes()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"MIS QUBO — test graph {args.graph_idx}  "
        f"({n} nodes,  {G.number_of_edges()} edges)",
        fontsize=14,
    )

    draw_input(axes[0][0], G, pos)
    draw_is(axes[0][1], G, pos, selected_greedy, "Greedy (GMIN)")
    draw_is(axes[1][0], G, pos, selected_gcon,   "GCON QUBO")
    draw_is(axes[1][1], G, pos, selected_hybrid, "ScatteringClique QUBO")

    plt.tight_layout()
    out = "mis_qubo_solutions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
