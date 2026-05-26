"""
test_ising_viz.py

Visualise Ising graph-partitioning results from the GatedGCN run.

  Row 0 — Training curves (loss/train, loss/valid vs epoch)

  Row 1 — One real test graph from the Ising dataset:
             col 0: Ground truth partition (stored y labels)
             col 1: GNN prediction         (probs_test.pt slice)

  Edge colours encode satisfaction of the Ising objective:
    Green solid  — edge is SATISFIED  (ferromagnetic + not cut,
                                       or antiferromagnetic + cut)
    Red dashed   — edge is VIOLATED   (ferromagnetic + cut,
                                       or antiferromagnetic + not cut)

Usage:
    python test_ising_viz.py
    python test_ising_viz.py --graph_idx 3 --split_index 0 --out ising_viz.png
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


COLORS = ["#4878CF", "#D65F5F"]   # blue = partition 0, red = partition 1


# ── helpers ───────────────────────────────────────────────────────────────────

def find_latest(pattern):
    candidates = glob.glob(pattern, recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


def decode(probs: np.ndarray) -> np.ndarray:
    return (probs >= 0.5).astype(int)


def ising_energy(labels, ei_src, ei_dst, weights) -> float:
    """Compute Ising energy E = -sum_{(i,j)} w_ij * s_i * s_j (undirected)."""
    s = 2.0 * labels.astype(float) - 1.0
    return -float(np.sum(weights * s[ei_src] * s[ei_dst])) / 2.0


# ── dataset loading ───────────────────────────────────────────────────────────

def load_test_graph(graph_idx: int, split_index: int = 0):
    """Return (A, weights_upper, true_labels, n, ei_src, ei_dst, weights_full)."""
    data_path = os.path.join("datasets", "ising", "mixed", "processed", "data.pt")
    data, slices = torch.load(data_path, map_location="cpu")

    split_file = os.path.join("splits", "ising_mixed_kfold-5.json")
    with open(split_file) as f:
        splits = json.load(f)
    test_idx = splits[str(split_index)]

    g = test_idx[graph_idx]
    n_start = slices["y"][g].item()
    n_end   = slices["y"][g + 1].item()
    e_start = slices["edge_index"][g].item()
    e_end   = slices["edge_index"][g + 1].item()

    n = n_end - n_start
    true_labels = data.y[n_start:n_end].numpy()

    ei = data.edge_index[:, e_start:e_end].numpy()   # local 0-based indices
    ei_src, ei_dst = ei[0], ei[1]

    # Original coupling weights (stored as edge_weight or edge_attr)
    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        raw = data.edge_weight[e_start:e_end].squeeze().numpy()
    else:
        raw = data.edge_attr[e_start:e_end].squeeze().numpy()

    # Build adjacency for upper-triangle only (undirected, no double-counting)
    A = np.zeros((n, n), dtype=np.float32)
    W = np.zeros((n, n), dtype=np.float32)
    for s, d, w in zip(ei_src, ei_dst, raw):
        A[s, d] = 1.0
        W[s, d] = w

    return A, W, true_labels, n, ei_src, ei_dst, raw


def probs_offset_for(graph_idx: int, split_index: int = 0):
    """Node offset into probs_test.pt for test graph `graph_idx`."""
    data, slices = torch.load(
        os.path.join("datasets", "ising", "mixed", "processed", "data.pt"),
        map_location="cpu",
    )
    split_file = os.path.join("splits", "ising_mixed_kfold-5.json")
    with open(split_file) as f:
        splits = json.load(f)
    test_idx = splits[str(split_index)]

    offset = 0
    for i in range(graph_idx):
        g = test_idx[i]
        offset += slices["y"][g + 1].item() - slices["y"][g].item()
    return offset


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_ising(ax, G, pos, labels, probs, W_mat, title):
    """Draw graph with nodes coloured by partition and edges by Ising satisfaction."""
    node_colors = [COLORS[int(l)] for l in labels]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=220, ax=ax)

    if probs is not None:
        node_lbls = {i: f"{probs[i]:.2f}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_lbls,
                                font_size=4.5, font_color="white", ax=ax)

    satisfied, violated = [], []
    for u, v in G.edges():
        w = W_mat[u, v]
        cut = (labels[u] != labels[v])
        # ferromagnetic (w>0): satisfied if NOT cut; antiferromagnetic (w<0): satisfied if cut
        ok = (w > 0 and not cut) or (w < 0 and cut)
        (satisfied if ok else violated).append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=satisfied,
                           edge_color="#2ca02c", alpha=0.6, width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=violated,
                           edge_color="crimson", width=2.0,
                           style="dashed", ax=ax)

    n_sat = len(satisfied)
    n_viol = len(violated)
    ax.set_title(f"{title}\n{n_sat} satisfied / {n_viol} violated", fontsize=9, pad=4)
    ax.axis("off")


# ── training curves ───────────────────────────────────────────────────────────

def plot_curves(axes, csv_path):
    ax_loss, ax_cut = axes
    df = pd.read_csv(csv_path)

    # loss
    for col, label, color, marker in [
        ("loss/train", "train", "#4878CF", "o"),
        ("loss/valid", "val",   "#D65F5F", "s"),
    ]:
        rows = df.dropna(subset=[col])[["epoch", col]]
        if not rows.empty:
            ax_loss.plot(rows["epoch"], rows[col],
                         label=label, color=color, marker=marker, markersize=3)
    ax_loss.set_xlabel("Epoch", fontsize=8)
    ax_loss.set_ylabel("Loss", fontsize=8)
    ax_loss.set_title("Training loss", fontsize=9)
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.tick_params(labelsize=7)

    # cut metrics
    for col, label, color, style in [
        ("gnn_cut/valid",     "GNN",     "#4878CF", "-"),
        ("spectral_cut/valid","Spectral", "gray",    "--"),
        ("greedy_cut/valid",  "Greedy",  "#ff7f0e", ":"),
    ]:
        if col in df.columns:
            rows = df.dropna(subset=[col])[["epoch", col]]
            if not rows.empty:
                ax_cut.plot(rows["epoch"], rows[col], label=label,
                            color=color, linestyle=style, marker="o", markersize=3)

    # test stars
    for col, label, color in [
        ("gnn_cut/test",     "GNN test",     "#4878CF"),
        ("spectral_cut/test","Spectral test", "gray"),
        ("greedy_cut/test",  "Greedy test",  "#ff7f0e"),
    ]:
        if col in df.columns:
            rows = df.dropna(subset=[col])
            if not rows.empty:
                last = rows.iloc[-1]
                ax_cut.scatter(last["epoch"], last[col],
                               marker="*", s=140, color=color, zorder=5,
                               label=f"{label}={last[col]:.3f}")

    ax_cut.set_xlabel("Epoch", fontsize=8)
    ax_cut.set_ylabel("Cut fraction", fontsize=8)
    ax_cut.set_title("GNN vs baselines", fontsize=9)
    ax_cut.legend(fontsize=7)
    ax_cut.grid(True, alpha=0.3)
    ax_cut.tick_params(labelsize=7)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx",   type=int, default=0)
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--out",         type=str, default="ising_viz.png")
    args = parser.parse_args()

    # find results from ising run specifically
    csv_path   = find_latest(os.path.join("results", "*ising*", "**", "metrics.csv"))
    probs_path = find_latest(os.path.join("results", "*ising*", "**", "probs_test.pt"))

    # fallback to any results if ising-specific not found
    if csv_path is None:
        csv_path = find_latest(os.path.join("results", "**", "metrics.csv"))
    if probs_path is None:
        probs_path = find_latest(os.path.join("results", "**", "probs_test.pt"))

    print(f"Metrics  : {csv_path}")
    print(f"Probs    : {probs_path}")

    A, W_mat, true_labels, n_nodes, ei_src, ei_dst, weights = load_test_graph(
        args.graph_idx, args.split_index
    )
    print(f"Test graph {args.graph_idx}: {n_nodes} nodes, {int(A.sum()//2)} edges")

    # GNN predictions
    offset = probs_offset_for(args.graph_idx, args.split_index)
    if probs_path:
        all_probs  = torch.load(probs_path, map_location="cpu").float().numpy()
        gnn_probs  = all_probs[offset: offset + n_nodes]
        gnn_labels = decode(gnn_probs)
        confident  = ((gnn_probs > 0.8) | (gnn_probs < 0.2)).mean()
        print(f"  GNN probs [{offset}:{offset+n_nodes}]  "
              f"mean={gnn_probs.mean():.3f}  confident: {confident:.1%}")
    else:
        gnn_probs = gnn_labels = None

    # Ising energies
    e_true = ising_energy(true_labels, ei_src, ei_dst, weights)
    e_gnn  = ising_energy(gnn_labels,  ei_src, ei_dst, weights) if gnn_labels is not None else float("nan")
    print(f"  Ising energy  gt={e_true:.2f}  gnn={e_gnn:.2f}")

    # Build NetworkX graph (upper triangle)
    rows_e, cols_e = np.where(np.triu(A, k=1) > 0)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(rows_e.tolist(), cols_e.tolist()))
    pos = nx.spring_layout(G, seed=42)

    # ── figure ────────────────────────────────────────────────────────────────
    has_curves = csv_path is not None
    n_rows = (1 if has_curves else 0) + 1
    height_ratios = ([2.5] if has_curves else []) + [3.5]

    fig = plt.figure(figsize=(10, 2.8 * has_curves + 4.5))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.5, wspace=0.08,
                          height_ratios=height_ratios)

    if has_curves:
        axes_curves = [fig.add_subplot(gs[0, c]) for c in range(2)]
        plot_curves(axes_curves, csv_path)

    graph_row = 1 if has_curves else 0
    ax_gt  = fig.add_subplot(gs[graph_row, 0])
    ax_gnn = fig.add_subplot(gs[graph_row, 1])

    draw_ising(ax_gt, G, pos, true_labels, None, W_mat,
               f"Ground truth  E={e_true:.1f}")

    if gnn_labels is not None:
        draw_ising(ax_gnn, G, pos, gnn_labels, gnn_probs, W_mat,
                   f"GNN  E={e_gnn:.1f}")
    else:
        ax_gnn.text(0.5, 0.5, "No probs_test.pt found",
                    ha="center", va="center", transform=ax_gnn.transAxes)
        ax_gnn.axis("off")

    # legend for edge colours
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#2ca02c", lw=2, label="Satisfied edge"),
        Line2D([0], [0], color="crimson", lw=2, linestyle="--", label="Violated edge"),
        Line2D([0], [0], color="#4878CF", marker="o", lw=0, markersize=8, label="Partition 0"),
        Line2D([0], [0], color="#D65F5F", marker="o", lw=0, markersize=8, label="Partition 1"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Ising GP  |  test graph {args.graph_idx}  "
        f"({n_nodes} nodes, {int(A.sum()//2)} edges)",
        fontsize=13, y=1.01,
    )

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
