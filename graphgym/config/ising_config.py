from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('ising_cfg')
def ising_cfg(cfg):
    """Top-level Ising dataset config — num_samples is read by SyntheticDataset.process()."""
    cfg.ising = CN()
    cfg.ising.num_samples = 1000


@register_config('ising_mixed_cfg')
def ising_mixed_cfg(cfg):
    """Parameters for the 'mixed' Ising variant (ER + regular + spin-glass)."""
    cfg.ising.mixed = CN()
    cfg.ising.mixed.n   = 100   # number of nodes per graph
    cfg.ising.mixed.eps = 0.05  # balance tolerance for Ising refinement


@register_config('ising_spin_glass_cfg')
def ising_spin_glass_cfg(cfg):
    """Parameters for the 'spin_glass' Ising variant (100% 5-regular ±1 weighted)."""
    cfg.ising.spin_glass = CN()
    cfg.ising.spin_glass.n   = 100
    cfg.ising.spin_glass.eps = 0.05
