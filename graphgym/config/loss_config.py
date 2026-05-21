from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('loss_param_cfg')
def loss_param_cfg(cfg):
    cfg.maxclique_loss = CN()
    cfg.maxclique_loss.beta = 0.1

    cfg.maxcut_loss = CN()

    cfg.mds_loss = CN()
    cfg.mds_loss.beta = 1.0

    cfg.mis_loss = CN()
    cfg.mis_loss.alpha = 1.0
    cfg.mis_loss.beta = 1.01
    # cfg.mis_loss.k = 2

    cfg.mis_loss_annealed = CN()
    cfg.mis_loss_annealed.tau = 1.0   # initial temperature (overridden by scheduler)
    cfg.mis_loss_annealed.eps = 1e-8  # numerical guard for log(0)

    cfg.plantedclique_loss = CN()

    cfg.metrics = CN()

    cfg.metrics.maxclique = CN()
    cfg.metrics.maxclique.dec_length = 300
    cfg.metrics.maxclique.num_seeds = 1

    cfg.metrics.mds = CN()
    cfg.metrics.mds.enable = True
    cfg.metrics.mds.num_seeds = 1

    cfg.metrics.mis = CN()
    cfg.metrics.mis.dec_length = 100
    cfg.metrics.maxclique.num_seeds = 1

    cfg.gp_loss = CN()
    cfg.gp_loss.alpha = 1.0    # edge term weight  — (1 - x_i·x_j)^2 on edges
    cfg.gp_loss.beta  = 1.0    # non-edge term weight — (x_i·x_j)^2 on non-edges

    cfg.gp_loss_balanced = CN()
    cfg.gp_loss_balanced.lam = 1.0   # balance coefficient — (sum x_i - n/2)^2

    cfg.metrics.gp = CN()
    cfg.metrics.gp.k = 2