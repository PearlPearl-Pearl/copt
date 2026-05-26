import torch
from torch_geometric.graphgym.register import register_loss
from torch_geometric.utils import remove_self_loops


@register_loss("ising_loss")
def ising_loss_pyg(batch, gamma=1000.0, **kwargs):
    """
    L = (1/B) sum_g [ -sum_{(i,j) in E} w_ij s_i s_j  +  gamma*(sum_i x_i - n/2)^2 ]

    x_i in [0,1]: node output probability.
    s_i = 2*x_i - 1: soft spin in [-1, 1].
    w_ij from edge_weight (original ±1 coupling constants); edge_attr is not
    used here because GatedGCN overwrites it with learned embeddings.
    """
    data_list = batch.to_data_list()
    loss = 0.0

    for data in data_list:
        x = data.x.squeeze()       # (n,)
        s = 2.0 * x - 1.0          # (n,)  soft spins
        n = data.num_nodes

        ei = remove_self_loops(data.edge_index)[0]
        src, dst = ei[0], ei[1]

        # Original scalar coupling weights, preserved across GatedGCN layers.
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            w = data.edge_weight.squeeze()
        elif data.edge_attr is not None:
            w = data.edge_attr.squeeze()
        else:
            w = torch.ones(src.shape[0], device=x.device)

        # Edges stored in both directions → divide by 2 to count each once.
        ising_energy = -torch.sum(w * s[src] * s[dst]) / 2.0
        balance = (x.sum() - n / 2.0) ** 2

        loss += ising_energy + gamma * balance

    return loss / batch.size(0)
