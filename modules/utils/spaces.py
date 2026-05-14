
import torch.nn as nn

from torch.optim import SGD, Adam

from modules.architecture.models import GCN, GAT, ScGCN
from modules.architecture.models_pyg import PygGCN

from graphgym.loss.copt_loss import (maxcut_loss, maxcut_mae,
                                     maxcut_loss_pyg, maxcut_mae_pyg,
                                     maxclique_loss, maxclique_loss_pyg)
from utils.norms import min_max_norm, min_max_norm_pyg

from utils.metrics import (
    maxcut_acc, maxcut_acc_pyg, maxcut_size_pyg,
    maxclique_ratio, maxclique_size_pyg, maxclique_ratio_pyg,
    plantedclique_acc_pyg,
    mds_size_pyg, mds_acc_pyg,
    mis_size_pyg, greedy_mis_size,
    gp_gnn_cut_pyg, gp_spectral_cut_pyg,
)

OPTIMIZER_DICT = {
    "sgd": SGD,
    "adam": Adam,
}

GNN_MODEL_DICT = {
    "pyg:gcn": PygGCN,
    "gcn": GCN,
    "gat": GAT,
    "scgcn": ScGCN,
}

LAST_ACTIVATION_DICT = {
    "maxcut": nn.Sigmoid(),
    "maxclique": nn.Sigmoid(),
    "mds": nn.Sigmoid(),
    "mis": nn.Sigmoid(),
    # "mds": nn.Sigmoid(),
    # "maxclique": None,
}

LAST_NORMALIZATION_DICT = {
    "maxcut": None,
    "maxclique": None,
    "mds": None,
    "mis": None,
    # "maxclique": min_max_norm_pyg,
}

LOSS_FUNCTION_DICT = {
    "maxcut": maxcut_loss_pyg,
    "maxclique": maxclique_loss_pyg,
    # "mds": mds_loss_pyg,

}
    
EVAL_FUNCTION_DICT = {
    "maxcut": {"size": maxcut_size_pyg}, # "acc": maxcut_acc_pyg},
    "maxclique": {"size": maxclique_size_pyg, 'ratio': maxclique_ratio_pyg},
    "mds": {"size": mds_size_pyg},
    "mis": {"size": mis_size_pyg, "greedy_size": greedy_mis_size},
    "plantedclique": {"acc": plantedclique_acc_pyg},
    "gp": {"gnn_cut": gp_gnn_cut_pyg, "spectral_cut": gp_spectral_cut_pyg},
}

EVAL_FUNCTION_DICT_NOLABEL = {
    "maxcut": {"size": maxcut_size_pyg},
    "maxclique": {"size": maxclique_size_pyg},
    "mds": {"size": mds_size_pyg},
    "mis": {"size": mis_size_pyg, "greedy_size": greedy_mis_size},
    "gp": {"gnn_cut": gp_gnn_cut_pyg, "spectral_cut": gp_spectral_cut_pyg},
}