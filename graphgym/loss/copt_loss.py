from typing import Union, Tuple, List, Dict, Any

import torch

from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index, remove_self_loops
from torch_geometric.graphgym.register import register_loss

from torch_scatter import scatter


def entropy(output, epsilon=1e-8):
    batch_size = output.batch.unique().size(0)
    p = output.x.squeeze()
    entropy = - (p * torch.log(p + epsilon) + (1 - p) * torch.log(1 - p + epsilon)).sum()
    return entropy / batch_size


### MAXCLIQUE ###

@register_loss("maxclique_loss")
def maxclique_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()

    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]

        loss1 = torch.sum(data.x[src] * data.x[dst])
        loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
        loss += (- loss1 + beta * loss2) * data.num_nodes

    return loss / batch.size(0)


def maxclique_loss(output, data, beta=0.1):
    adj = data.get('adj')

    loss1 = torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output))
    loss2 = output.sum() ** 2 - loss1 - torch.sum(output ** 2)

    return - loss1.sum() + beta * loss2.sum()


### MAXCUT ###

@register_loss("maxcut_loss")
def maxcut_loss_pyg(data):
    x = (data.x - 0.5) * 2
    src, dst = data.edge_index[0], data.edge_index[1]
    return torch.sum(x[src] * x[dst]) / len(data.batch.unique())


def maxcut_loss(data):
    x = (data['x'] - 0.5) * 2
    adj = data['adj_mat']
    return torch.matmul(x.transpose(-1, -2), torch.matmul(adj, x)).mean()


@register_loss("maxcut_mae")
def maxcut_mae_pyg(data):
    x = (data.x > 0.5).float()
    x = (x - 0.5) * 2
    y = data.cut_binary
    y = (y - 0.5) * 2

    x_list = unbatch(x, data.batch)
    y_list = unbatch(y, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    ae_list = []
    for x, y, edge_index in zip(x_list, y_list, edge_index_list):
        ae_list.append(torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) - torch.sum(y[edge_index[0]] * y[edge_index[1]] == -1.0))

    return 0.5 * torch.Tensor(ae_list).abs().mean()


def maxcut_mae(data):
    output = (data['x'] > 0.5).double()
    target = torch.nan_to_num(data['cut_binary'])

    adj = data['adj_mat']
    adj_weight = adj.sum(-1).sum(-1)
    target_size = adj_weight.clone()
    pred_size = adj_weight.clone()

    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target = 1 - target
    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target_size /= 2

    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    output = 1 - output
    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    pred_size /= 2

    return torch.mean(torch.abs(pred_size - target_size))


### COLORING ###

def color_loss(output, adj):
    output = (output - 0.5) * 2

    return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).diagonal(dim1=-1, dim2=-2).sum() - 4 * torch.abs(output).sum()


### PLANTEDCLIQUE ###

from torch.nn import BCEWithLogitsLoss
ce_loss = BCEWithLogitsLoss()

@register_loss("plantedclique_loss")
def plantedclique_loss_pyg(data):
    return ce_loss(data.x, data.y.unsqueeze(-1))


### MDS ###

@register_loss("mds_loss")
def mds_loss_pyg(data, beta=1.0):
    batch_size = data.batch.max() + 1.0
    
    p = data.x.squeeze()
    edge_index = remove_self_loops(data.edge_index)[0]
    row, col = edge_index[0], edge_index[1]

    loss = p.sum() + beta * (
        scatter(
            torch.log1p(-p)[row],
            index=col,
            reduce='sum',
        ).exp() * (1 - p)
    ).sum()

    return loss / batch_size


### MIS ###

# @register_loss("mis_loss")
# def mis_loss_pyg(data, beta=1.0, k=2, eps=1e-1):
#     batch_size = data.batch.max() + 1.0

#     edge_index = remove_self_loops(data.edge_index)[0]
#     row, col = edge_index[0], edge_index[1]
#     degree = torch.exp(data.degree)

#     l1 = - torch.sum(data.x ** 2)
#     l2 = + torch.sum((data.x[row] * data.x[col]) ** 2)

#     # l1 = - torch.sum(torch.log(1 - data.x) * degree)
#     # l2 = + torch.log((data.x[row] * data.x[col]) ** 1).sum()

#     # l1 = - data.x.sum()
#     # l2 = + ((data.x[row] * data.x[col]) ** k).sum()

#     loss = l1 + beta * l2

#     return loss #/ batch_size


# @register_loss("mis_loss")
# def mis_loss_pyg(batch, beta=0.1):
#     data_list = batch.to_data_list()

#     loss = 0.0
#     for data in data_list:
#         src, dst = data.edge_index[0], data.edge_index[1]

#         loss1 = torch.sum(data.x[src] * data.x[dst])
#         loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
#         loss += (- loss2 + beta * loss1) * data.num_nodes

#     return loss / batch.size(0)


@register_loss("mis_loss")
def mis_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()

    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]

        loss1 = torch.sum(data.x)
        loss2 = torch.sum(data.x[src]*data.x[dst])

        loss += (-loss1 + beta*loss2)*data.num_nodes

        return loss/batch.size(0)

### MAXBIPARTITE ###

def maxbipartite_loss(output, adj, beta):
    return maxclique_loss(output, torch.matrix_power(adj, 2), beta)



### GRAPH PARTITIONING ###
# @register_loss("gp_loss")
# def gp_loss_pyg(data, beta=100, gamma=100):
#     batch_size = data.batch.unique().size(0)
#     x = data.x.squeeze()  # probabilities in [0,1]
#     src, dst = data.edge_index[0], data.edge_index[1]

#     # Term 1: minimize edges crossing the cut
#     loss1 = torch.sum((x[src] - x[dst]) ** 2)

#     # Term 2 & 3: soft balance constraint — keep partition sizes reasonable
#     n = x.size(0)
#     partition_sum = x.sum()
#     loss2 = torch.log(1 + torch.exp(1 - partition_sum))
#     loss3 = torch.log(1 + torch.exp(partition_sum - (n - 1)))

#     # Term 4: discreteness penalty — push x_i toward {0, 1}
#     loss4 = torch.sum(x * (1 - x))

#     return (loss1 + beta * (loss2 + loss3) + gamma * loss4) / batch_size


# @register_loss("gp_loss")
# def gp_loss_pyg(data, beta=1.0, gamma=1.0):
#     batch_size = data.batch.unique().size(0)
#     x = data.x  # (total_nodes, k) — softmax output, rows sum to 1
#     src, dst = data.edge_index[0], data.edge_index[1]
#     k = cfg.metrics.gp.k
#     diff = x[src] - x[dst]
#     loss1 = (diff ** 2).sum()
#     partition_sums = x.sum(dim=0)
#     loss2 = torch.log(1 + torch.exp(1 - partition_sums)).sum()
#     loss3 = (x * (1 - x)).sum()
#     return (loss1 + beta * loss2 + gamma * loss3) / batch_size


# @register_loss("gp_loss")
# def gp_loss_pyg(data, lambda_=0.1):
#     """
#     Loss for 2-partition Graph Partitioning (k=2, scalar output).
#     y = 2p - 1 maps p in [0,1] to y in [-1, 1].

#     Term 1 (cut): -sum_{(i,j) in E} y_i * y_j
#         Minimised when adjacent nodes have the same sign (same partition).

#     Term 2 (balance): per-graph penalty on mean(y_g)^2
#         Normalised by graph size so the penalty is scale-independent.
#         Zero when the partition is perfectly balanced within each graph.
#     """
#     from torch_scatter import scatter

#     batch_size = data.batch.unique().size(0)
#     p = data.x.squeeze()                                          # (N,) in [0, 1]
#     y = 2 * p - 1                                                 # (N,) in [-1, 1]
#     src, dst = data.edge_index[0], data.edge_index[1]

#     loss_cut = -torch.sum(y[src] * y[dst])

#     graph_sums = scatter(y, data.batch, reduce='sum')             # (B,)
#     n_per_graph = scatter(torch.ones_like(y), data.batch, reduce='sum')  # (B,)
#     loss_bal = ((graph_sums / n_per_graph) ** 2).sum()            # scalar in [0, B]

#     return (loss_cut + lambda_ * loss_bal) / batch_size


### Literature version of loss:

# @register_loss("gp_loss")
# def gp_loss_pyg(data, eps=1e-9):
#     """
#     DMoN modularity loss (Tsitsulin et al., 2020), adapted for k=2 scalar sigmoid.
    
#     Modularity Q = (1 / 2m) * sum_{ij} (A_ij - d_i d_j / 2m) * delta(c_i, c_j)
#                  = (1/2m) tr(S^T B S)  where B = A - dd^T/2m
    
#     Loss = -modularity + collapse_regulariser
#     """
#     from torch_scatter import scatter

#     x = data.x.squeeze().clamp(eps, 1 - eps)
#     src, dst = data.edge_index[0], data.edge_index[1]
#     batch = data.batch
#     N = x.size(0)
#     B = batch.max().item() + 1

#     S = torch.stack([x, 1 - x], dim=1)                        # (N, 2)

#     # Per-graph: 2m, degree
#     deg = scatter(torch.ones_like(src, dtype=x.dtype), src, dim=0, dim_size=N, reduce='sum')
#     two_m_per_graph = scatter(deg, batch, dim=0, dim_size=B, reduce='sum')   # (B,)

#     # tr(S^T A S) per graph (same as MinCutPool numerator)
#     edge_contrib = (S[src] * S[dst]).sum(dim=1)
#     edge_batch = batch[src]
#     tr_SAS = scatter(edge_contrib, edge_batch, dim=0, dim_size=B, reduce='sum')

#     # tr(S^T d d^T S / 2m) per graph = sum_c (sum_i d_i S_ic)^2 / 2m
#     # First compute d^T S per graph and per cluster
#     dS = deg.unsqueeze(1) * S                                  # (N, 2)
#     dS_per_graph = scatter(dS, batch, dim=0, dim_size=B, reduce='sum')  # (B, 2)
#     tr_SddS = (dS_per_graph ** 2).sum(dim=1) / (two_m_per_graph + eps)  # (B,)

#     modularity = (tr_SAS - tr_SddS) / (two_m_per_graph + eps)  # (B,)
#     L_mod = -modularity.mean()

#     # Collapse regulariser: penalise unbalanced cluster sizes per graph
#     # || sum_i S_i / N_g ||_F * sqrt(k) - 1
#     cluster_sizes = scatter(S, batch, dim=0, dim_size=B, reduce='sum')  # (B, 2)
#     n_per_graph = scatter(torch.ones_like(x), batch, dim=0, dim_size=B, reduce='sum')  # (B,)
#     cluster_frac = cluster_sizes / n_per_graph.unsqueeze(1)
#     L_collapse = (torch.norm(cluster_frac, dim=1) * (2 ** 0.5) - 1).mean()

#     return L_mod + L_collapse

# @register_loss("gp_loss")
# def gp_loss_pyg(batch, beta=1000):
#     data_list = batch.to_data_list()
#     loss = 0.0
#     for data in data_list:
#         src, dst = data.edge_index[0], data.edge_index[1]
#
#         # term 1: push adjacent nodes to same partition
#         loss1 = torch.sum((data.x[src] - data.x[dst]) ** 2)
#
#         # term 2: penalize non-adjacent nodes in same partition
#         # compute all pairwise differences
#         diff = data.x.unsqueeze(0) - data.x.unsqueeze(1)  # (n, n, 1)
#         all_pairs = (diff ** 2).squeeze(-1)  # (n, n)
#
#         # mask out diagonal and adjacent pairs
#         adj_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
#         adj_mask[src, dst] = True
#         adj_mask.fill_diagonal_(True)
#
#         non_adj_pairs = (~adj_mask)
#         loss2 = torch.sum(1 - all_pairs[non_adj_pairs])
#
#         loss += (loss1 + beta * loss2) * data.num_nodes
#     return loss / batch.size(0)


# @register_loss("gp_loss")
# def gp_loss_pyg(batch, beta=1000, **kwargs):
#     data_list = batch.to_data_list()
#     loss = 0.0
#     for data in data_list:
#         src, dst = data.edge_index[0], data.edge_index[1]

#         # term 1: push adjacent nodes to same partition
#         # data.x is (n, k); squared L2 norm of differences, summed over edges
#         loss1 = torch.sum((data.x[src] - data.x[dst]) ** 2)

#         # term 2: penalize non-adjacent nodes in same partition
#         # compute all pairwise squared L2 distances
#         diff = data.x.unsqueeze(0) - data.x.unsqueeze(1)  # (n, n, k)
#         all_pairs = (diff ** 2).sum(dim=-1)  # (n, n) — squared L2 norm per pair

#         # mask out diagonal and adjacent pairs
#         adj_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
#         adj_mask[src, dst] = True
#         adj_mask.fill_diagonal_(True)

#         non_adj_pairs = (~adj_mask)
#         loss2 = torch.sum(1 - 0.5 * all_pairs[non_adj_pairs])

#         loss += (loss1 + beta * loss2) * data.num_nodes
#     return loss / batch.size(0)

# @register_loss("gp_loss")
# def gp_loss_pyg(batch, beta1=10.0, beta2=1.0, **kwargs):
#     data_list = batch.to_data_list()
#     loss = 0.0
#     for data in data_list:
#         src, dst = data.edge_index[0], data.edge_index[1]
        
#         # term 1: 1/2 * sum_{(i,j) in E} ||s_i - s_j||^2
#         loss1 = 0.5 * torch.sum((data.x[src] - data.x[dst]) ** 2)
        
#         # term 2: sum_{i != j, (i,j) not in E} <s_i, s_j>
#         all_inner = data.x @ data.x.T  # (n, n) — <s_i, s_j>
#         adj_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool, device=data.x.device)
#         adj_mask[src, dst] = True
#         adj_mask.fill_diagonal_(True)
#         non_adj_pairs = (~adj_mask)
#         loss2 = torch.sum(all_inner[non_adj_pairs])
        
#         # term 3: discreteness penalty — push ||s_i|| toward 1 (i.e., toward one-hot)
#         norms = torch.linalg.norm(data.x, dim=-1)  # (n,)
#         loss3 = torch.sum((1 - norms) ** 2)
        
#         loss += (loss1 + beta1 * loss2 + beta2 * loss3) * data.num_nodes
#     return loss / batch.size(0)

@register_loss("gp_loss_balanced")
def gp_loss_balanced_pyg(
    batch,
    lam=1.0,
    **kwargs
):
    """
    Laplacian-quadratic graph bisection loss (scalar node assignment).

        L = x^T L x  +  lam * (sum_i x_i - n/2)^2

    where L = D - A is the unnormalized graph Laplacian.

    The first term equals sum_{(i,j) in E} (x_i - x_j)^2, penalising cut
    edges.  The second term is a soft balance constraint pushing the total
    assignment toward n/2 (i.e. equal-sized partitions when x_i in [0,1]).
    """
    data_list = batch.to_data_list()
    total_loss = 0.0

    for data in data_list:
        x = data.x.squeeze()          # (n,) scalar assignment per node
        n = data.num_nodes
        src = data.edge_index[0]
        dst = data.edge_index[1]

        loss_cut     = torch.sum((x[src] - x[dst]) ** 2)
        loss_balance = (x.sum() - n / 2.0) ** 2

        total_loss += loss_cut + lam * loss_balance

    return total_loss / batch.size(0)


@register_loss("gp_loss")
def gp_loss_pyg(
    batch,
    k=2,
    alpha=1.0,
    beta=10.0,
    gamma=1.0,
    delta=1.0,
    eps=1e-8,
    **kwargs
):
    """
    Graph k-partition loss.

    data.x is assumed to contain node embeddings of shape (num_nodes, d).

    Terms:
        1. Edge smoothness:
           connected nodes should be close on the unit sphere.

        2. Quantization:
           each node should move toward one of k prototype directions.

        3. Balance:
           avoid putting all nodes in the same partition.

        4. Norm:
           keep embeddings close to the unit sphere.
    """

    data_list = batch.to_data_list()
    total_loss = 0.0

    for data in data_list:
        x = data.x
        device = x.device
        n, d = x.shape

        src, dst = data.edge_index[0], data.edge_index[1]

        # Normalize embeddings for dot-product geometry
        x_norm = x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)

        # -------------------------------------------------
        # Build k prototype directions
        # -------------------------------------------------
        if d == 2:
            angles = torch.arange(k, device=device, dtype=x.dtype) * (2.0 * torch.pi / k)
            prototypes = torch.stack(
                [torch.cos(angles), torch.sin(angles)],
                dim=-1
            )
        else:
            # For d > 2, use random fixed directions on the unit sphere
            # You may replace this later by simplex prototypes.
            prototypes = torch.randn(k, d, device=device, dtype=x.dtype)
            prototypes = prototypes / (
                torch.linalg.norm(prototypes, dim=-1, keepdim=True) + eps
            )

        # -------------------------------------------------
        # Term 1: edge attraction
        # L_edge = sum_{(i,j) in E} w_ij (1 - <x_i, x_j>)
        # -------------------------------------------------
        edge_dot = torch.sum(x_norm[src] * x_norm[dst], dim=-1)
        loss_edge = torch.mean(1.0 - edge_dot)

        # -------------------------------------------------
        # Term 2: quantization to nearest prototype
        # L_quant = sum_i min_r ||x_i - c_r||^2
        # -------------------------------------------------
        dist_to_proto = torch.cdist(x_norm, prototypes, p=2) ** 2
        min_dist = torch.min(dist_to_proto, dim=1).values
        loss_quant = torch.mean(min_dist)

        # -------------------------------------------------
        # Soft assignment to prototypes
        # q_ir = softmax(<x_i, c_r>)
        # -------------------------------------------------
        logits = x_norm @ prototypes.T
        q = torch.softmax(logits, dim=-1)

        # -------------------------------------------------
        # Term 3: balance penalty
        # each cluster should receive about n/k nodes
        # -------------------------------------------------
        cluster_mass = torch.mean(q, dim=0)
        target_mass = torch.full_like(cluster_mass, 1.0 / k)
        loss_balance = torch.sum((cluster_mass - target_mass) ** 2)

        # -------------------------------------------------
        # Term 4: unit norm penalty
        # -------------------------------------------------
        norms = torch.linalg.norm(x, dim=-1)
        loss_norm = torch.mean((norms - 1.0) ** 2)

        loss = (
            alpha * loss_edge
            + beta * loss_quant
            + gamma * loss_balance
            + delta * loss_norm
        )

        total_loss += loss

    return total_loss / batch.size(0)