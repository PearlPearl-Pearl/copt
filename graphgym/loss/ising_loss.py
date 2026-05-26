import torch
from torch_geometric.graphgym.register import register_loss
from torch_geometric.utils import remove_self_loops


@register_loss("ising_loss")
def ising_loss_pyg(batch, gamma=1000.0, **kwargs):
    """
    Weighted Ising loss for balanced binary graph partitioning.

    For soft assignments x_i in [0, 1], maps to spins s_i = 2*x_i - 1 in [-1, 1].

    L = (1/B) * sum_g  n_g * [
          - sum_{(i,j) in E}  w_ij * s_i * s_j          (Ising energy)
          + gamma * (sum_i x_i  -  n/2)^2               (balance)
        ]

    Minimising the Ising energy encourages:
      - ferromagnetic edges  (w > 0): neighbours in the same partition
      - antiferromagnetic edges (w < 0): neighbours in opposite partitions

    gamma controls how strongly the balanced-partition constraint is enforced.
    """
    data_list = batch.to_data_list()
    loss = 0.0

    for data in data_list:
        x = data.x.squeeze()           # (n,)  probabilities in [0, 1]
        s = 2.0 * x - 1.0              # (n,)  soft spins in [-1, 1]
        n = data.num_nodes

        ei = remove_self_loops(data.edge_index)[0]
        src, dst = ei[0], ei[1]

        # edge weights: use edge_attr if available, else default to 1
        if data.edge_attr is not None:
            w = data.edge_attr.squeeze()   # (|E|,)
        else:
            w = torch.ones(src.shape[0], device=x.device)

        # term 1: Ising energy (divide by 2 — undirected edges appear twice)
        ising_energy = -torch.sum(w * s[src] * s[dst]) / 2.0

        # term 2: balance penalty
        balance = (x.sum() - n / 2.0) ** 2

        loss += (ising_energy + gamma * balance) * n

    return loss / batch.size(0)
