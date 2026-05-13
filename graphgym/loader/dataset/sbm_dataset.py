import numpy as np
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

from graphgym.loader.dataset.synthetic import SyntheticDataset
from utils.sbm import stochastic_block_model  # noqa: re-exported for convenience


class SBMDataset(SyntheticDataset):
    def __init__(self, name, root, transform=None, pre_transform=None):
        super().__init__('sbm', name, root, transform, pre_transform)

    def create_graph(self, idx):
        n = np.random.randint(self.params.n_min, self.params.n_max + 1)
        k = self.params.k
        p_in = self.params.p_in
        p_out = self.params.p_out

        A, labels = stochastic_block_model(n, k, p_in, p_out)

        # ensure connected (re-sample if isolated nodes appear)
        g = nx.from_numpy_array(A)
        attempts = 0
        while not nx.is_connected(g) and attempts < 10:
            A, labels = stochastic_block_model(n, k, p_in, p_out)
            g = nx.from_numpy_array(A)
            attempts += 1

        g_pyg = from_networkx(g)
        g_pyg.partition_labels = torch.tensor(labels, dtype=torch.long)
        g_pyg.num_partitions = k
        return g_pyg
