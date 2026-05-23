"""
plot_mis_all_results.py

For both the Linear loss and the QUBO loss:
  1. Training loss curve  (GCON vs ScatteringClique)
  2. Validation loss curve (GCON vs ScatteringClique)
  3. 2x2 solution visualisation on one test graph:
        top-left    : Greedy (GMIN)
        top-right   : Input Graph
        bottom-left : GCON
        bottom-right: ScatteringClique

Usage:
    python plot_mis_all_results.py
    python plot_mis_all_results.py --graph_idx 3
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

C_GCON   = "#1f77b4"
C_HYBRID = "#d62728"


# ── helpers ───────────────────────────────────────────────────────────────────

def latest_file(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return os.path.abspath(max(hits, key=os.path.getmtime))


def epoch_mean(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()


# ── loss curves ───────────────────────────────────────────────────────────────

def plot_loss_curves(gcon_csv, hybrid_csv, loss_tag, title_prefix):
    dfs = {}
    if gcon_csv:   dfs["gcon"]   = pd.read_csv(gcon_csv)
    if hybrid_csv: dfs["hybrid"] = pd.read_csv(hybrid_csv)

    for col, ylabel, title, fname in [
        ("loss/train", "Training loss",
         f"{title_prefix} — Training Loss",
         f"mis_{loss_tag}_train_loss.png"),
        ("loss/valid", "Validation loss",
         f"{title_prefix} — Validation Loss",
         f"mis_{loss_tag}_val_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, color, label in [
            ("gcon",   C_GCON,   "GCON"),
            ("hybrid", C_HYBRID, "ScatteringClique"),
        ]:
            if name not in dfs:
                continue
            series = epoch_mean(dfs[name], col)
            if series.empty:
                print(f"  [warn] '{col}' not found for {name}")
                continue
            ax.plot(series.index, series.values,
                    label=label, color=color,
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


# ── GNN inference ─────────────────────────────────────────────────────────────

def load_model_testset(ckpt_path, cfg_path):
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


def greedy_mis(G):
    remaining = set(G.nodes())
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    selected = np.zeros(G.number_of_nodes(), dtype=bool)
    while remaining:
        node = min(remaining, key=lambda v: len(adj[v] & remaining))
        selected[node] = True
        remaining -= adj[node] & remaining
        remaining.discard(node)
    return selected


def build_graph(data):
    ei = remove_self_loops(data.edge_index)[0].numpy()
    mask = ei[0] < ei[1]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(zip(ei[0][mask], ei[1][mask]))
    return G


# ── draw helpers ──────────────────────────────────────────────────────────────

def draw_input(ax, G, pos, title):
    nx.draw_networkx_nodes(G, pos, node_color="#95a5a6", node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")


def draw_is(ax, G, pos, selected, title):
    n = G.number_of_nodes()
    is_size = selected.sum()
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
        f"{title}\nIS size = {is_size} ({100*is_size/n:.1f}%)  violations = {len(bad)}",
        fontsize=11,
    )
    ax.axis("off")


def visualise(gcon_model, hybrid_model, gcon_device, hybrid_device,
              test_graphs, graph_idx, loss_tag, title_prefix):
    data = test_graphs[graph_idx]
    G    = build_graph(data)
    pos  = nx.spring_layout(G, seed=42)
    n    = G.number_of_nodes()

    selected_greedy = greedy_mis(G)
    selected_gcon   = gnn_is(gcon_model,   data, gcon_device)
    selected_hybrid = gnn_is(hybrid_model, data, hybrid_device)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"MIS {title_prefix} — test graph {graph_idx}  "
        f"({n} nodes,  {G.number_of_edges()} edges)",
        fontsize=14,
    )

    draw_is(axes[0][0],   G, pos, selected_greedy, "Greedy (GMIN)")
    draw_input(axes[0][1], G, pos,
               f"Input Graph\n{n} nodes,  {G.number_of_edges()} edges")
    draw_is(axes[1][0],   G, pos, selected_gcon,   "GCON")
    draw_is(axes[1][1],   G, pos, selected_hybrid, "ScatteringClique")

    plt.tight_layout()
    out = f"mis_{loss_tag}_solutions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    runs = {
        "linear": {
            "gcon_csv":    latest_file("results/mis_rb_small_gcon*/**/metrics.csv"),
            "hybrid_csv":  latest_file("results/mis_rb_small_hybridconv*/**/metrics.csv"),
            "gcon_ckpt":   latest_file("results/mis_rb_small_gcon*/**/*.ckpt"),
            "hybrid_ckpt": latest_file("results/mis_rb_small_hybridconv*/**/*.ckpt"),
            "gcon_cfg":    "configs/benchmarks/mis/mis_rb_small_gcon.yaml",
            "hybrid_cfg":  "configs/benchmarks/mis/mis_rb_small_hybridconv.yaml",
            "title":       "Linear Loss (α=1, β=10)",
        },
        "qubo": {
            "gcon_csv":    latest_file("results/mis_rb_small_qubo_gcon*/**/metrics.csv"),
            "hybrid_csv":  latest_file("results/mis_rb_small_qubo_hybridconv*/**/metrics.csv"),
            "gcon_ckpt":   latest_file("results/mis_rb_small_qubo_gcon*/**/*.ckpt"),
            "hybrid_ckpt": latest_file("results/mis_rb_small_qubo_hybridconv*/**/*.ckpt"),
            "gcon_cfg":    "configs/benchmarks/mis/mis_rb_small_qubo_gcon.yaml",
            "hybrid_cfg":  "configs/benchmarks/mis/mis_rb_small_qubo_hybridconv.yaml",
            "title":       "QUBO Loss (penalty=2)",
        },
    }

    for loss_tag, r in runs.items():
        print(f"\n{'='*60}")
        print(f" {r['title']}")
        print(f"{'='*60}")

        print("── Loss curves ──")
        plot_loss_curves(r["gcon_csv"], r["hybrid_csv"], loss_tag, r["title"])

        if r["gcon_ckpt"] and r["hybrid_ckpt"]:
            print("── Loading models ──")
            gcon_model,   test_graphs, gcon_device   = load_model_testset(r["gcon_ckpt"],   r["gcon_cfg"])
            hybrid_model, _,           hybrid_device = load_model_testset(r["hybrid_ckpt"], r["hybrid_cfg"])
            print("── Visualising ──")
            visualise(gcon_model, hybrid_model, gcon_device, hybrid_device,
                      test_graphs, args.graph_idx, loss_tag, r["title"])
        else:
            missing = []
            if not r["gcon_ckpt"]:   missing.append("gcon ckpt")
            if not r["hybrid_ckpt"]: missing.append("hybridconv ckpt")
            print(f"  [warn] missing {', '.join(missing)} — skipping visualisation")

    print("\nDone.")


if __name__ == "__main__":
    main()
