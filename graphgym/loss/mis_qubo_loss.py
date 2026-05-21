import torch
from torch_geometric.graphgym.register import register_loss
from torch_geometric.utils import remove_self_loops


@register_loss("mis_loss_qubo")
def mis_loss_qubo_pyg(batch, penalty=2.0):
    """
    QUBO loss for Maximum Independent Set.

    Minimises  x^T Q x  =  -sum_i x_i^2  +  penalty * sum_{(i,j) in E} x_i * x_j

    The quadratic node term -x_i^2 differs from the linear Hamiltonian -x_i
    for x_i in (0,1): it penalises uncertainty more sharply and pushes
    probabilities toward {0,1} during continuous optimisation.

    Feasibility of the IS constraint requires penalty > 1 (default 2.0).

    Reference: COPT-MT (mis_loss_qubo_pyg).
    """
    data_list = batch.to_data_list()
    loss = 0.0

    for data in data_list:
        phi = data.x.squeeze()  # (n,) probabilities in [0, 1]

        size_term = -torch.sum(phi ** 2)

        edge_index = remove_self_loops(data.edge_index)[0]
        src, dst = edge_index[0], edge_index[1]
        # divide by 2: undirected edges appear twice in edge_index
        edge_term = penalty * torch.sum(phi[src] * phi[dst]) / 2.0

        loss += (size_term + edge_term) / data.num_nodes

    return loss / batch.size(0)
