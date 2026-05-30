from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('weighted_gp_cfg')
def weighted_gp_cfg(cfg):
    """Top-level weighted GP dataset config — num_samples read by SyntheticDataset.process()."""
    cfg.weighted_gp = CN()
    cfg.weighted_gp.num_samples = 1000


@register_config('weighted_gp_mixed_cfg')
def weighted_gp_mixed_cfg(cfg):
    """Parameters for the 'mixed' weighted GP variant (ER + regular + spin-glass)."""
    cfg.weighted_gp.mixed = CN()
    cfg.weighted_gp.mixed.n   = 100   # nodes per graph
    cfg.weighted_gp.mixed.eps = 0.05  # balance tolerance for refinement
