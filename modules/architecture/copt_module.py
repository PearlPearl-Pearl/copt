import logging
import time
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch_geometric.graphgym import register

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict

from graphgym.loss.copt_loss import entropy
from modules.utils.spaces import OPTIMIZER_DICT, LOSS_FUNCTION_DICT, EVAL_FUNCTION_DICT, EVAL_FUNCTION_DICT_NOLABEL


class COPTModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)

        # Loss function
        loss_func = register.loss_dict[cfg.model.loss_fun]
        loss_params = cfg[cfg.model.loss_fun]
        self.loss_func = partial(loss_func, **loss_params)
        if cfg.optim.entropy.scheduler == "linear-energy":
            self.alpha = (cfg.optim.entropy.base_temp / cfg.optim.entropy.min_temp - 1) / cfg.optim.max_epoch
        elif cfg.optim.entropy.scheduler == "linear-entropy":
            self.alpha = (cfg.optim.entropy.base_temp - cfg.optim.entropy.min_temp) / cfg.optim.max_epoch

        # Eval function
        if not cfg.dataset.label:
            self.eval_func_dict = EVAL_FUNCTION_DICT_NOLABEL[cfg.train.task]
        else:
            self.eval_func_dict = EVAL_FUNCTION_DICT[cfg.train.task]
        for key, eval_func in self.eval_func_dict.items():
            if cfg.train.task in cfg.metrics:
                eval_func = partial(eval_func, **cfg.metrics[cfg.train.task])
            self.eval_func_dict[key] = eval_func

        self._test_probs = []   # accumulates batch.x across test batches
        self._current_tau = cfg.optim.entropy.base_temp  # tracked for logging

    def on_train_epoch_start(self):
        if cfg.optim.entropy.enable and cfg.model.loss_fun == "mis_loss_annealed":
            tau = cfg.optim.entropy.base_temp / (1.0 + self.alpha * self.current_epoch)
            loss_func = register.loss_dict[cfg.model.loss_fun]
            loss_params = dict(cfg[cfg.model.loss_fun])
            loss_params['tau'] = tau
            self.loss_func = partial(loss_func, **loss_params)
            self._current_tau = tau
            print(f"  [anneal] epoch {self.current_epoch:3d}  tau = {tau:.6f}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]

    def _log_probs(self, batch, split):
        # Use raw logits if available (set by Softmax activation), else fall back to batch.x
        z = getattr(batch, 'logits', None)
        x = (z if z is not None else batch.x).detach()   # (N, k) logits or probs
        for i in range(x.shape[-1] if x.dim() > 1 else 1):
            col = x[:, i] if x.dim() > 1 else x
            self.log(f"logits/mean_z{i}_{split}", col.mean(), batch_size=batch.batch_size, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"logits/std_z{i}_{split}",  col.std(),  batch_size=batch.batch_size, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, *args, **kwargs):
        batch.split = "train"
        out = self.forward(batch)
        loss = self.loss_func(batch)
        self.log("loss/train", loss, batch_size=batch.batch_size, on_step=True, prog_bar=True, logger=True)
        if cfg.train.log_probs:
            self._log_probs(batch, "train")

        if cfg.optim.entropy.enable and cfg.model.loss_fun == "mis_loss_annealed":
            # tau is already baked into loss via on_train_epoch_start; just log it
            self.log("train/tau", self._current_tau, batch_size=batch.batch_size,
                     on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif cfg.optim.entropy.enable:
            # legacy additive entropy correction for other losses
            if cfg.optim.entropy.scheduler == "linear-energy":
                tau = cfg.optim.entropy.base_temp / (1.0 + self.alpha * self.current_epoch)
            elif cfg.optim.entropy.scheduler == "linear-entropy":
                tau = cfg.optim.entropy.base_temp - self.alpha * self.current_epoch

            H = tau * entropy(out)
            self.log("loss/train-entropy", H, batch_size=batch.batch_size, on_step=True, prog_bar=True, logger=True)
            loss -= H
            self.log("loss/train-anneal-loss", loss, batch_size=batch.batch_size, on_step=True, prog_bar=True, logger=True)

        step_end_time = time.time()
        return dict(loss=loss, step_end_time=step_end_time)

    def validation_step(self, batch, *args, **kwargs):
        batch.split = "val"
        out = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        eval_dict = dict(loss=loss, step_end_time=step_end_time)
        self.log("loss/valid", loss, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        if cfg.train.log_probs:
            self._log_probs(batch, "val")
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(batch)
            eval_dict.update({eval_type: eval})
            self.log("".join([eval_type, "/valid"]), eval, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        return eval_dict

    def test_step(self, batch, *args, **kwargs):
        cfg.test = True
        out = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        eval_dict = dict(loss=loss, step_end_time=step_end_time)
        self.log("loss/test", loss, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(batch)
            eval_dict.update({eval_type: eval})
            self.log("".join([eval_type, "/test"]), eval, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        if cfg.train.log_probs:
            z = getattr(batch, 'logits', None)
            src = z if z is not None else batch.x
            self._test_probs.append(src.detach().cpu())
        return eval_dict

    def on_test_epoch_end(self):
        if not cfg.train.log_probs or not self._test_probs:
            return
        import os
        logits = torch.cat(self._test_probs)           # (total_nodes, k) raw logits in R
        out_path = os.path.join(cfg.run_dir, "probs_test.pt")
        torch.save(logits, out_path)

        print(f"\n--- GNN raw logits (test set) ---")
        print(f"  shape : {tuple(logits.shape)}")
        k = logits.shape[1] if logits.dim() > 1 else 1
        for i in range(k):
            col = logits[:, i] if logits.dim() > 1 else logits
            print(f"  z{i}  min={col.min():.4f}  max={col.max():.4f}  "
                  f"mean={col.mean():.4f}  std={col.std():.4f}")

        # Partition assignments via argmax over logits
        total = logits.shape[0]
        _MAX_SHOW = 30
        def _fmt_set(indices):
            idx = indices.tolist()
            if len(idx) <= _MAX_SHOW:
                return "{" + ", ".join(map(str, idx)) + "}"
            return "{" + ", ".join(map(str, idx[:_MAX_SHOW])) + f", ... +{len(idx)-_MAX_SHOW} more}}"

        assignments = logits.argmax(dim=-1) if logits.dim() > 1 else (logits >= 0).long()
        for part in range(k):
            idx = torch.where(assignments == part)[0]
            n = len(idx)
            print(f"  partition {part} : {n:6d} nodes  ({100*n/total:.1f}%)")
            print(f"  S{part+1} = {_fmt_set(idx)}")

        # Per-column histogram over the actual logit range
        flat = logits.flatten() if logits.dim() > 1 else logits
        lo_val, hi_val = flat.min().item(), flat.max().item()
        if lo_val == hi_val:
            hi_val = lo_val + 1.0
        counts = torch.histc(flat.float(), bins=10, min=lo_val, max=hi_val)
        step = (hi_val - lo_val) / 10
        label = "(all columns flattened)" if logits.dim() > 1 else ""
        print(f"  histogram {label}:")
        max_c = counts.max().item() or 1
        for i, c in enumerate(counts):
            lo, hi = lo_val + i * step, lo_val + (i + 1) * step
            bar = "█" * int(c.item() * 40 / max_c)
            print(f"    [{lo:+.2f} → {hi:+.2f}]  {bar} {int(c.item())}")
        print(f"  saved to: {out_path}")

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    if cfg.pretrained.dir:
        logging.info(f'Loading pretrained model from {cfg.pretrained.dir}')
        model = COPTModule.load_from_checkpoint(cfg.pretrained.dir, dim_in=dim_in, dim_out=dim_out, cfg=cfg)
    else:
        model = COPTModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
