from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('ising_loss_cfg')
def ising_loss_cfg(cfg):
    cfg.ising_loss = CN()
    cfg.ising_loss.gamma = 1000.0   # balance penalty weight
