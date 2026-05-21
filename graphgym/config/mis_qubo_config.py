from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('mis_qubo_cfg')
def mis_qubo_cfg(cfg):
    cfg.mis_loss_qubo = CN()
    cfg.mis_loss_qubo.penalty = 2.0
