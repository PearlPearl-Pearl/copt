"""
plot_mis_results.py

1. Plots train + val loss curves separately for gcon and hybridconv.
2. Loads the best gcon checkpoint, runs inference on one test graph,
   and draws the IS solution (selected nodes highlighted).

Usage:
    python plot_mis_results.py
    python plot_mis_results.py --graph_idx 3
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import remove_self_loops

# ── auto-detect helpers ───────────────────────────────────────────────────────

def latest_csv(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return os.path.abspath(max(hits, key=os.path.getmtime))


def epoch_mean(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[["epoch", col]].dropna(subset=[col]).groupby("epoch")[col].mean()


# ── loss curve plots ──────────────────────────────────────────────────────────

C_GCON   = "#1f77b4"   # blue
C_HYBRID = "#d62728"   # red

def plot_comparison(gcon_csv, hybrid_csv):
    """Two figures: one comparing train losses, one comparing val losses."""
    dfs = {}
    if gcon_csv:   dfs["gcon"]       = pd.read_csv(gcon_csv)
    if hybrid_csv: dfs["hybridconv"] = pd.read_csv(hybrid_csv)

    colors = {"gcon": C_GCON, "hybridconv": C_HYBRID}

    for col, ylabel, title, fname in [
        ("loss/train", "Training loss",   "MIS — Training Loss: gcon vs hybridconv",   "mis_train_loss.png"),
        ("loss/valid", "Validation loss", "MIS — Validation Loss: gcon vs hybridconv", "mis_val_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in dfs.items():
            series = epoch_mean(df, col)
            if series.empty:
                print(f"  [warn] '{col}' not found for {name}")
                continue
            ax.plot(series.index, series.values, label=name,
                    color=colors[name], linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved → {fname}")


# ── IS graph visualisation ────────────────────────────────────────────────────

def visualise_mis(ckpt_path, cfg_path, graph_idx, out_path, model_name="gcon"):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from graphgym.loader.dataset.rb_dataset import RBDataset

    # ── load config ───────────────────────────────────────────────────────────
    set_cfg(cfg)
    cfg.train.mode = None

    class _Args:
        cfg_file = cfg_path
        opts = []

    load_cfg(cfg, _Args())

    # ── load dataset ──────────────────────────────────────────────────────────
    from graphgym.patches import create_loader
    from torch_geometric.data import Batch
    cfg.dataset.split_index = 0
    loaders = create_loader()
    test_loader = loaders[-1]   # train / val / test

    test_graphs = []
    for batch in test_loader:
        test_graphs.extend(batch.to_data_list())

    print(f"Test set: {len(test_graphs)} graphs")
    data = test_graphs[graph_idx]
    n = data.num_nodes
    print(f"\nGraph {graph_idx}: {n} nodes")

    # ── load model and run inference ──────────────────────────────────────────
    from modules.architecture.copt_module import COPTModule
    cfg.share.dim_in  = data.x.shape[1] if data.x is not None else 1
    cfg.share.dim_out = cfg.dim_out

    model = COPTModule.load_from_checkpoint(
        ckpt_path, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=cfg
    )
    model.eval()
    device = next(model.parameters()).device

    # add batch index so the model can handle a single graph
    data.batch = torch.zeros(n, dtype=torch.long)
    batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        model(batch)

    probs = batch.x.squeeze().cpu().numpy()   # (n,) after sigmoid

    # ── greedy IS decoder ─────────────────────────────────────────────────────
    order = np.argsort(probs)[::-1]           # descending probability
    ei = remove_self_loops(data.edge_index)[0].numpy()
    adj = {i: set() for i in range(n)}
    for s, d in zip(ei[0], ei[1]):
        adj[s].add(d); adj[d].add(s)

    selected = np.zeros(n, dtype=bool)
    blocked  = np.zeros(n, dtype=bool)
    for idx in order:
        if not blocked[idx]:
            selected[idx] = True
            for nb in adj[idx]:
                blocked[nb] = True

    is_size = selected.sum()
    print(f"IS size: {is_size}  ({100*is_size/n:.1f}% of nodes)")

    # ── draw ──────────────────────────────────────────────────────────────────
    mask = ei[0] < ei[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0][mask], ei[1][mask]))
    pos = nx.spring_layout(G, seed=42)

    colors = ["#e74c3c" if selected[i] else "#95a5a6" for i in range(n)]
    sizes  = [120 if selected[i] else 40 for i in range(n)]

    bad_edges = [(u, v) for u, v in G.edges() if selected[u] and selected[v]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"MIS — {model_name} — test graph {graph_idx}  ({n} nodes,  {G.number_of_edges()} edges)",
        fontsize=13,
    )

    # left: raw graph
    ax = axes[0]
    nx.draw_networkx_nodes(G, pos, node_color="#95a5a6", node_size=60, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title("Input graph", fontsize=11)
    ax.axis("off")

    # right: IS solution
    ax = axes[1]
    colors = ["#e74c3c" if selected[i] else "#95a5a6" for i in range(n)]
    sizes  = [120 if selected[i] else 40 for i in range(n)]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    if bad_edges:
        nx.draw_networkx_edges(G, pos, edgelist=bad_edges,
                               edge_color="black", width=3, ax=ax)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label=f"In IS ({is_size})"),
        Patch(color="#95a5a6", label=f"Not in IS ({n - is_size})"),
    ], fontsize=10)
    ax.set_title(
        f"GNN solution — IS size = {is_size} ({100*is_size/n:.1f}%)  violations = {len(bad_edges)}",
        fontsize=11,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx", type=int, default=0)
    args = parser.parse_args()

    gcon_csv    = latest_csv("results/mis_rb_small_gcon*/**/metrics.csv")
    hybrid_csv  = latest_csv("results/mis_rb_small_hybridconv*/**/metrics.csv")
    gcon_ckpt   = latest_csv("results/mis_rb_small_gcon*/**/*.ckpt")
    gcon_cfg    = "configs/benchmarks/mis/mis_rb_small_gcon.yaml"

    print("── Loss curves ──────────────────────────────────────────────")
    plot_comparison(gcon_csv, hybrid_csv)

    hybrid_ckpt = latest_csv("results/mis_rb_small_hybridconv*/**/*.ckpt")
    hybrid_cfg  = "configs/benchmarks/mis/mis_rb_small_hybridconv.yaml"

    print("\n── IS visualisation (gcon) ──────────────────────────────────")
    if gcon_ckpt and os.path.exists(gcon_cfg):
        print(f"  checkpoint : {gcon_ckpt}")
        visualise_mis(gcon_ckpt, gcon_cfg, args.graph_idx, "mis_gcon_is_graph.png", "gcon")
    else:
        print("  [warn] gcon checkpoint or config not found, skipping")

    print("\n── IS visualisation (hybridconv) ────────────────────────────")
    if hybrid_ckpt and os.path.exists(hybrid_cfg):
        print(f"  checkpoint : {hybrid_ckpt}")
        visualise_mis(hybrid_ckpt, hybrid_cfg, args.graph_idx, "mis_hybridconv_is_graph.png", "hybridconv")
    else:
        print("  [warn] hybridconv checkpoint or config not found, skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
