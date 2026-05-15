"""
test_gp_k2_viz.py

Two-panel visualisation of k=2 Graph Partitioning results:

  Row 0  — Training curves from the most recently modified metrics.csv:
             left : loss/train and loss/valid vs epoch
             mid  : gnn_cut/valid vs spectral_cut/valid vs epoch
             right: probs mean (train + val) — shows whether the model is
                    converging toward confident {0,1} predictions

  Row 1  — One real test graph from the SBM dataset:
             col 0: Ground truth        (stored partition_labels)
             col 1: Spectral baseline   (Fiedler vector + k-means)
             col 2: GNN prediction      (probs_test.pt slice for this graph)

Usage:
    python test_gp_k2_viz.py
    python test_gp_k2_viz.py --graph_idx 5 --out my_viz.png
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

from utils.spectral import spectral_partition, normalised_cut

COLORS = ["#4878CF", "#D65F5F"]   # blue = partition 0,  red = partition 1


# ── helpers ───────────────────────────────────────────────────────────────────

def find_latest(pattern):
    candidates = glob.glob(pattern, recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


def decode(probs: np.ndarray) -> np.ndarray:
    return (probs >= 0.5).astype(int)


def cut_size(edges, labels) -> int:
    return sum(1 for u, v in edges if labels[u] != labels[v])


# ── per-graph drawing ─────────────────────────────────────────────────────────

def draw_partition(ax, G, pos, labels, probs, title, n_nodes):
    cut  = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    same = [(u, v) for u, v in G.edges() if labels[u] == labels[v]]

    nx.draw_networkx_nodes(G, pos,
                           node_color=[COLORS[l] for l in labels],
                           node_size=220, ax=ax)
    if probs is not None:
        node_lbls = {i: f"{probs[i]:.2f}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_lbls,
                                font_size=4.5, font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same,
                           edge_color="#aaaaaa", alpha=0.5, width=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut,
                           edge_color="crimson", width=2.0,
                           style="dashed", ax=ax)
    ax.set_title(f"{title}\ncut={len(cut)}", fontsize=9, pad=4)
    ax.axis("off")


# ── training-curve panel ──────────────────────────────────────────────────────

def plot_curves(axes, csv_path):
    ax_loss, ax_cut, ax_probs = axes
    df = pd.read_csv(csv_path)

    # --- loss ---
    train_loss = df.dropna(subset=["loss/train"])[["epoch", "loss/train"]]
    val_loss   = df.dropna(subset=["loss/valid"])[["epoch", "loss/valid"]]
    if not train_loss.empty:
        ax_loss.plot(train_loss["epoch"], train_loss["loss/train"],
                     label="train", color="#4878CF", marker="o", markersize=3)
    if not val_loss.empty:
        ax_loss.plot(val_loss["epoch"], val_loss["loss/valid"],
                     label="val", color="#D65F5F", marker="s", markersize=3)
    ax_loss.set_xlabel("Epoch", fontsize=8)
    ax_loss.set_ylabel("Loss", fontsize=8)
    ax_loss.set_title("Training loss", fontsize=9)
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.tick_params(labelsize=7)

    # --- cut comparison ---
    cut_rows = df.dropna(subset=["gnn_cut/valid"])[
        ["epoch", "gnn_cut/valid", "spectral_cut/valid"]
    ]
    if not cut_rows.empty:
        ax_cut.plot(cut_rows["epoch"], cut_rows["gnn_cut/valid"],
                    label="GNN", color="#4878CF", marker="o", markersize=3)
        ax_cut.plot(cut_rows["epoch"], cut_rows["spectral_cut/valid"],
                    label="Spectral", color="gray", linestyle="--", linewidth=1.5)
    # add test point if available
    test_rows = df.dropna(subset=["gnn_cut/test"])
    if not test_rows.empty:
        last = test_rows.iloc[-1]
        ax_cut.scatter(last["epoch"], last["gnn_cut/test"],
                       marker="*", s=120, color="#4878CF", zorder=5,
                       label=f"GNN test={last['gnn_cut/test']:.3f}")
        if not np.isnan(last.get("spectral_cut/test", float("nan"))):
            ax_cut.scatter(last["epoch"], last["spectral_cut/test"],
                           marker="*", s=120, color="gray", zorder=5,
                           label=f"Spectral test={last['spectral_cut/test']:.3f}")
    ax_cut.set_xlabel("Epoch", fontsize=8)
    ax_cut.set_ylabel("Cut (normalised)", fontsize=8)
    ax_cut.set_title("GNN vs Spectral cut", fontsize=9)
    ax_cut.legend(fontsize=7)
    ax_cut.grid(True, alpha=0.3)
    ax_cut.tick_params(labelsize=7)

    # --- probs mean ---
    prob_train = df.dropna(subset=["probs/mean_p0_train"])[
        ["epoch", "probs/mean_p0_train"]
    ]
    prob_val = df.dropna(subset=["probs/mean_p0_val"])[
        ["epoch", "probs/mean_p0_val"]
    ]
    if not prob_train.empty:
        ax_probs.plot(prob_train["epoch"], prob_train["probs/mean_p0_train"],
                      label="train", color="#4878CF", marker="o", markersize=3)
    if not prob_val.empty:
        ax_probs.plot(prob_val["epoch"], prob_val["probs/mean_p0_val"],
                      label="val", color="#D65F5F", marker="s", markersize=3)
    ax_probs.axhline(0.5, color="black", linestyle=":", linewidth=1, label="p=0.5")
    ax_probs.set_ylim(0, 1)
    ax_probs.set_xlabel("Epoch", fontsize=8)
    ax_probs.set_ylabel("Mean prob output", fontsize=8)
    ax_probs.set_title("Mean GNN prob (should diverge from 0.5)", fontsize=9)
    ax_probs.legend(fontsize=7)
    ax_probs.grid(True, alpha=0.3)
    ax_probs.tick_params(labelsize=7)


# ── main ──────────────────────────────────────────────────────────────────────

def load_test_graph(graph_idx: int, split_index: int = 0):
    """Return (A, true_labels, n_nodes) for test graph `graph_idx` (0-based)."""
    data_tuple = torch.load(
        os.path.join("datasets", "sbm", "small", "processed", "data.pt"),
        map_location="cpu",
    )
    data, slices = data_tuple

    split_file = os.path.join("splits", "sbm_small_kfold-5.json")
    with open(split_file) as f:
        splits = json.load(f)
    test_idx = splits[str(split_index)]   # list of global graph indices for this test fold

    g = test_idx[graph_idx]               # global index into the full dataset
    n_start = slices["partition_labels"][g].item()
    n_end   = slices["partition_labels"][g + 1].item()
    e_start = slices["edge_index"][g].item()
    e_end   = slices["edge_index"][g + 1].item()

    true_labels = data.partition_labels[n_start:n_end].numpy()
    # edge_index is stored with local (0-based) node indices
    ei = data.edge_index[:, e_start:e_end].numpy()

    n = n_end - n_start
    A = np.zeros((n, n), dtype=np.float32)
    A[ei[0], ei[1]] = 1.0

    return A, true_labels, n


def probs_offset_for(graph_idx: int, split_index: int = 0):
    """Return the byte offset (in nodes) into probs_test.pt for test graph `graph_idx`."""
    data_tuple = torch.load(
        os.path.join("datasets", "sbm", "small", "processed", "data.pt"),
        map_location="cpu",
    )
    _, slices = data_tuple

    split_file = os.path.join("splits", "sbm_small_kfold-5.json")
    with open(split_file) as f:
        splits = json.load(f)
    test_idx = splits[str(split_index)]

    offset = 0
    for i in range(graph_idx):
        g = test_idx[i]
        offset += slices["partition_labels"][g + 1].item() - slices["partition_labels"][g].item()
    return offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_idx",   type=int, default=0,
                        help="Which test graph to show (0-based index into test split)")
    parser.add_argument("--split_index", type=int, default=0,
                        help="Which k-fold split was used for training (default 0)")
    parser.add_argument("--out",         type=str, default="gp_k2_viz.png")
    args = parser.parse_args()

    csv_path   = find_latest(os.path.join("results", "**", "metrics.csv"))
    probs_path = find_latest(os.path.join("results", "**", "probs_test.pt"))

    print(f"Metrics  : {csv_path}")
    print(f"Probs    : {probs_path}")

    # ── load real test graph ──────────────────────────────────────────────────
    A, true_labels, n_nodes = load_test_graph(args.graph_idx, args.split_index)
    print(f"Test graph {args.graph_idx}: {n_nodes} nodes, {int(A.sum()//2)} edges")

    offset = probs_offset_for(args.graph_idx, args.split_index)
    if probs_path:
        all_probs = torch.load(probs_path, map_location="cpu").float().numpy()
        gnn_probs  = all_probs[offset: offset + n_nodes]
        gnn_labels = decode(gnn_probs)
        print(f"  GNN probs [{offset}:{offset+n_nodes}]  "
              f"mean={gnn_probs.mean():.3f}  confident: "
              f"{((gnn_probs > 0.8) | (gnn_probs < 0.2)).mean():.1%}")
    else:
        gnn_probs  = None
        gnn_labels = None

    spec_labels = spectral_partition(A, 2)

    ncut_true = normalised_cut(A, true_labels, 2)
    ncut_spec = normalised_cut(A, spec_labels, 2)
    ncut_gnn  = normalised_cut(A, gnn_labels, 2) if gnn_labels is not None else float("nan")

    rows_e, cols_e = np.where(np.triu(A, k=1) > 0)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(rows_e.tolist(), cols_e.tolist()))
    pos = nx.spring_layout(G, seed=42)

    # ── figure layout ─────────────────────────────────────────────────────────
    has_curves = csv_path is not None
    n_rows = (1 if has_curves else 0) + 1
    height_ratios = ([2.5] if has_curves else []) + [3.5]
    fig = plt.figure(figsize=(13, 2.8 * has_curves + 4.5))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.5, wspace=0.08,
                          height_ratios=height_ratios)

    if has_curves:
        axes_curves = [fig.add_subplot(gs[0, c]) for c in range(3)]
        plot_curves(axes_curves, csv_path)

    # ── graph row ─────────────────────────────────────────────────────────────
    graph_row = 1 if has_curves else 0
    ax_gt   = fig.add_subplot(gs[graph_row, 0])
    ax_spec = fig.add_subplot(gs[graph_row, 1])
    ax_gnn  = fig.add_subplot(gs[graph_row, 2])

    draw_partition(ax_gt,   G, pos, true_labels, None,      f"Ground truth  NCut={ncut_true:.3f}", n_nodes)
    draw_partition(ax_spec, G, pos, spec_labels, None,      f"Spectral       NCut={ncut_spec:.3f}", n_nodes)
    if gnn_labels is not None:
        draw_partition(ax_gnn, G, pos, gnn_labels, gnn_probs, f"GNN            NCut={ncut_gnn:.3f}", n_nodes)
    else:
        ax_gnn.text(0.5, 0.5, "No probs_test.pt found",
                    ha="center", va="center", transform=ax_gnn.transAxes)
        ax_gnn.axis("off")

    fig.suptitle(
        f"GP k=2  |  test graph {args.graph_idx}  ({n_nodes} nodes, "
        f"{int(A.sum()//2)} edges)",
        fontsize=13, y=1.01,
    )

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
