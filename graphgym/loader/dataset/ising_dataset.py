import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from graphgym.loader.dataset.synthetic import SyntheticDataset


# ── graph generators ──────────────────────────────────────────────────────────

def _generate_graph(n, graph_type):
    if graph_type == "er":
        return nx.erdos_renyi_graph(n, p=5 / n)
    elif graph_type in ("regular", "spin_glass"):
        # need even n for random_regular_graph
        if n % 2 != 0:
            n += 1
        return nx.random_regular_graph(5, n)
    else:
        raise ValueError(f"unknown graph_type: {graph_type}")


def _assign_weights(G, graph_type):
    W = {}
    for (i, j) in G.edges():
        if graph_type == "spin_glass":
            w = float(np.random.choice([-1.0, 1.0]))
        else:
            w = 1.0
        W[(i, j)] = w
        W[(j, i)] = w
    return W


def _init_labels(n):
    labels = np.zeros(n, dtype=int)
    perm = np.random.permutation(n)
    labels[perm[:n // 2]] = 0
    labels[perm[n // 2:]] = 1
    return labels


def _refine_balanced(G, labels, W, eps=0.05, steps=50):
    n = len(labels)
    target = n / 2
    min_size = int((1 - eps) * target)
    max_size = int((1 + eps) * target)
    sizes = np.bincount(labels, minlength=2)

    for _ in range(steps):
        improved = False
        for i in np.random.permutation(n):
            current = labels[i]
            other = 1 - current
            if sizes[current] - 1 < min_size:
                continue
            if sizes[other] + 1 > max_size:
                continue
            si = -1 if current == 0 else 1
            delta = sum(2 * W[(i, j)] * si * (-1 if labels[j] == 0 else 1)
                        for j in G.neighbors(i))
            if delta < 0:
                labels[i] = other
                sizes[current] -= 1
                sizes[other] += 1
                improved = True
        if not improved:
            break
    return labels


def _node_features(G):
    deg = np.array([G.degree(i) for i in G.nodes()], dtype=float)
    deg = deg / (deg.max() + 1e-8)
    return torch.tensor(deg, dtype=torch.float).unsqueeze(1)


def _build_data(G, labels, W):
    edge_index, edge_attr = [], []
    for (i, j) in G.edges():
        edge_index += [[i, j], [j, i]]
        edge_attr  += [W[(i, j)], W[(j, i)]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float).unsqueeze(1)
    x = _node_features(G)
    y = torch.tensor(labels, dtype=torch.long)
    # Keep original scalar coupling weights so loss functions can use them
    # even after GatedGCN overwrites edge_attr with learned embeddings.
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                edge_weight=edge_attr.clone(), y=y)


def generate_ising_instance(n=100, graph_type="er", eps=0.05):
    G = _generate_graph(n, graph_type)
    G = nx.convert_node_labels_to_integers(G)
    W = _assign_weights(G, graph_type)
    labels = _init_labels(G.number_of_nodes())
    labels = _refine_balanced(G, labels, W, eps=eps)
    return _build_data(G, labels, W)


# ── PyG InMemoryDataset wrapper ───────────────────────────────────────────────

_GRAPH_TYPE_PROBS = [("er", 0.4), ("regular", 0.4), ("spin_glass", 0.2)]


class IsingDataset(SyntheticDataset):
    """Mixed Ising graph-partitioning dataset.

    Each graph is one of:
      - ER (p=5/n), unweighted           — 40 %
      - 5-regular, unweighted            — 40 %
      - 5-regular, ±1 spin-glass weights — 20 %

    Data fields per graph:
      x          : (n, 1) normalised degree
      edge_index : (2, 2|E|)  directed edge list (both directions)
      edge_attr  : (2|E|, 1)  edge weight  ∈ {-1, +1, 1}
      y          : (n,)       ground-truth balanced binary partition
    """

    def __init__(self, name, root, transform=None, pre_transform=None):
        super().__init__('ising', name, root, transform, pre_transform)

    def create_graph(self, idx):
        r = np.random.rand()
        cumulative = 0.0
        graph_type = _GRAPH_TYPE_PROBS[-1][0]
        for gtype, prob in _GRAPH_TYPE_PROBS:
            cumulative += prob
            if r < cumulative:
                graph_type = gtype
                break

        return generate_ising_instance(
            n=self.params.n,
            graph_type=graph_type,
            eps=self.params.eps,
        )
