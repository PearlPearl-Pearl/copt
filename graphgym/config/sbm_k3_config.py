from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('sbm_small_k3_cfg')
def sbm_small_k3_cfg(cfg):
    """SBM small with k=3 communities — separate cache from the k=2 small variant."""
    cfg.sbm.small_k3 = CN()
    cfg.sbm.small_k3.n_min = 50
    cfg.sbm.small_k3.n_max = 100
    cfg.sbm.small_k3.k = 3
    cfg.sbm.small_k3.p_in = 0.7
    cfg.sbm.small_k3.p_out = 0.1
