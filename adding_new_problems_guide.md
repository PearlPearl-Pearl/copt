---
title: "How to Add a New Problem to the COPT Codebase"
subtitle: "A Step-by-Step Integration Guide"
author: "COPT Architecture Guide"
date: "May 2026"
geometry: margin=2.5cm
fontsize: 11pt
toc: true
toc-depth: 3
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{COPT — Adding New Problems}
  - \fancyhead[R]{\thepage}
  - \fancyfoot[C]{}
  - \usepackage{xcolor}
  - \definecolor{codegray}{HTML}{F5F5F5}
  - \usepackage{tcolorbox}
---

\newpage

# Overview

This guide explains how to integrate a new combinatorial optimization problem into the **COPT** codebase. COPT is a Graph Neural Network (GNN) framework built on top of PyTorch Geometric's **GraphGym**, designed to learn approximate solvers for graph-based combinatorial optimization problems (e.g., MaxCut, MaxClique, MIS, MDS).

The framework takes a problem-agnostic approach: the same GNN backbone is reused across problems, and you plug in a new problem by defining its **loss function**, **evaluation metric**, and **configuration**. No changes to the model architecture are required.

## Existing Problems (for Reference)

| Problem | Type | Loss File | Metric File | Config File |
|---|---|---|---|---|
| MaxCut | Unsupervised | `copt_loss.py:47` | `metrics.py:177` | `maxcut.yaml` |
| MaxClique | Unsupervised | `copt_loss.py:21` | `metrics.py:21` | `maxclique.yaml` |
| MIS | Unsupervised | `copt_loss.py:163` | `metrics.py:314` | `mis.yaml` |
| MDS | Unsupervised | `copt_loss.py:120` | `metrics.py:244` | `mds.yaml` |
| PlantedClique | Supervised | `copt_loss.py:113` | `metrics.py:230` | `plantedclique.yaml` |

---

\newpage

# Architecture at a Glance

Before diving in, here is how the pieces fit together:

```
main.py
  └── COPTModule (modules/architecture/copt_module.py)
        ├── model  ←─ GNN backbone (encoder → MP layers → head)
        ├── loss_func  ←─ looked up from register.loss_dict by name
        └── eval_func_dict  ←─ looked up from EVAL_FUNCTION_DICT (spaces.py)

Training loop:
  1. Forward pass: batch → model → batch.x updated with predictions
  2. Loss:         loss_func(batch)   [uses batch.x + batch.edge_index]
  3. Eval:         eval_func(batch)   [greedy decoder → solution quality]

Config:
  configs/<problem>.yaml   ←─ sets train.task and model.loss_fun
  graphgym/config/loss_config.py  ←─ registers hyperparameters into cfg
```

The model predicts a **soft assignment** `batch.x ∈ [0, 1]^N` for every node (or a single scalar for graph-level tasks). The loss function shapes this continuous relaxation toward a valid, high-quality discrete solution.

---

\newpage

# Step-by-Step Integration

There are **5 required steps** and **1 optional step** (needed only if your problem requires ground-truth labels computed from the graph).

## Step 1 — Define the Loss Function

**File to modify:** `graphgym/loss/copt_loss.py`

This is the most important step. The loss function encodes the mathematical structure of your problem as a differentiable objective over the GNN's soft predictions.

### What the function receives

```python
@register_loss("myproblem_loss")
def myproblem_loss_pyg(batch, beta=0.1):
    # batch.x         : (N, 1) float tensor — soft node assignments in [0,1]
    # batch.edge_index: (2, E) long tensor  — edge connectivity (COO format)
    # batch.batch     : (N,)  long tensor  — maps each node to its graph index
    # beta            : float              — penalty / regularisation weight
    ...
    return loss  # scalar tensor
```

### Pattern: objective + penalty

Almost every existing loss follows this pattern:

```
loss = -objective_term + beta * constraint_term
```

- **`objective_term`**: what you want to maximise (flip the sign for gradient descent).
- **`constraint_term`**: a soft penalty that pushes the solution toward feasibility.

### Example: MaxCut (simplest form)

```python
@register_loss("maxcut_loss")
def maxcut_loss_pyg(data):
    x = (data.x - 0.5) * 2          # re-center to [-1, 1]
    src, dst = data.edge_index[0], data.edge_index[1]
    # Maximise edges that cross the cut:  x_u * x_v = -1 when on opposite sides
    return torch.sum(x[src] * x[dst]) / len(data.batch.unique())
```

### Example: MaxClique (per-graph loop)

```python
@register_loss("maxclique_loss")
def maxclique_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()
    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]
        # Maximise sum of selected edges (clique must be a complete subgraph)
        loss1 = torch.sum(data.x[src] * data.x[dst])
        # Penalise non-edges among selected nodes
        loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
        loss += (- loss1 + beta * loss2) * data.num_nodes
    return loss / batch.size(0)
```

### Template for a new problem

```python
@register_loss("myproblem_loss")
def myproblem_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()
    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]

        # ── YOUR OBJECTIVE ──────────────────────────────────────────────
        objective = ...   # differentiable function of data.x, src, dst
        # ── YOUR CONSTRAINT ─────────────────────────────────────────────
        penalty = ...     # soft infeasibility measure

        loss += (- objective + beta * penalty) * data.num_nodes

    return loss / batch.size(0)
```

**Tips:**

- Use `torch_scatter.scatter` for efficient neighbourhood aggregation (see MDS loss for an example).
- Scale the loss by `data.num_nodes` to balance gradients across graphs of different sizes, then normalise by `batch.size(0)`.
- The `@register_loss("myproblem_loss")` name must **exactly match** `model.loss_fun` in your YAML config.

---

## Step 2 — Define the Evaluation Metric

**File to modify:** `utils/metrics.py`

The metric function decodes the GNN's soft predictions into a **discrete, feasible solution** and measures its quality. It is called during validation and testing.

### What the function receives

```python
def myproblem_size_pyg(batch, dec_length=300, num_seeds=1):
    # batch.x         : soft predictions after forward pass
    # batch.edge_index: graph structure
    # dec_length      : how many top-ranked nodes to consider in the decoder
    # num_seeds       : randomise starting point to escape local optima
    ...
    return torch.Tensor([metric_per_graph]).mean()
```

### Greedy decoder pattern

Most problems use a **greedy rank-and-prune** decoder:

1. Sort nodes by `batch.x` in descending order (highest confidence first).
2. Greedily add each node if it does not violate the problem constraints.
3. Return the objective value of the resulting discrete solution.

### Example skeleton

```python
def myproblem_size_pyg(batch, dec_length=300, num_seeds=1):
    data_list = batch.to_data_list()
    results = []
    for data in data_list:
        # Sort by predicted probability, descending
        order = torch.argsort(data.x.squeeze(), descending=True)
        selected = torch.zeros(data.num_nodes, dtype=torch.bool)

        for idx in order[:dec_length]:
            selected[idx] = True
            if not is_feasible(selected, data.edge_index):
                selected[idx] = False   # prune infeasible choice

        # Compute objective on final discrete solution
        results.append(compute_objective(selected, data))

    return torch.tensor(results, dtype=torch.float).mean()
```

Replace `is_feasible` and `compute_objective` with your problem's logic.

---

## Step 3 — Register Configuration Parameters

**File to modify:** `graphgym/config/loss_config.py`

All hyperparameters for the loss and metric functions must be declared here so that GraphGym can inject them via the `cfg` global object and allow YAML overrides.

```python
@register_config('loss_param_cfg')
def loss_param_cfg(cfg):
    # ... existing problem configs ...

    # ── YOUR PROBLEM ─────────────────────────────────────────────────────
    cfg.myproblem_loss = CN()
    cfg.myproblem_loss.beta = 0.1      # loss penalty weight

    cfg.metrics.myproblem = CN()
    cfg.metrics.myproblem.dec_length = 300   # greedy decoder depth
    cfg.metrics.myproblem.num_seeds = 1      # decoder restarts
```

The node names (`myproblem_loss`, `metrics.myproblem`) must match how they are referenced in the YAML config and in `copt_module.py`.

**How it flows:**

`copt_module.py` line 26–27 reads:

```python
loss_params = cfg[cfg.model.loss_fun]          # e.g. cfg["myproblem_loss"]
self.loss_func = partial(loss_func, **loss_params)
```

And lines 39–40 read:

```python
if cfg.train.task in cfg.metrics:
    eval_func = partial(eval_func, **cfg.metrics[cfg.train.task])
```

So your registered names must be consistent end-to-end.

---

## Step 4 — Register in the Problem Mappings

**File to modify:** `modules/utils/spaces.py`

This file provides two dictionaries that `COPTModule` uses to look up the evaluation functions at runtime.

```python
# At the top of the file, add your import:
from utils.metrics import myproblem_size_pyg

# Then add your problem to both dictionaries:

EVAL_FUNCTION_DICT = {
    # ... existing entries ...
    "myproblem": {"size": myproblem_size_pyg},
}

EVAL_FUNCTION_DICT_NOLABEL = {
    # ... existing entries ...
    "myproblem": {"size": myproblem_size_pyg},
}
```

Use `EVAL_FUNCTION_DICT` when your problem has ground-truth labels (`dataset.label: true`), and `EVAL_FUNCTION_DICT_NOLABEL` for unsupervised problems. Add your problem to whichever applies (or both if you want to support both modes).

> **Note:** The key `"myproblem"` must exactly match `train.task` in your YAML config.

---

## Step 5 — Create the YAML Configuration File

**File to create:** `configs/myproblem.yaml`

This is the experiment configuration. Copy an existing config (e.g., `maxcut.yaml`) and change the fields highlighted below.

```yaml
---
out_dir: results
metric_best: size          # which eval metric to track for model selection
num_workers: 4
dim_out: 1                 # 1 for node-level problems

wandb:
  use: True
  name: myproblem          # ← your problem name
  project: copt-pyg

dataset:
  format: ba               # graph generator: ba (Barabási–Albert),
                           #   rb, er (Erdős–Rényi), pc, bp
  name: small
  task: graph
  split_mode: cv-kfold-5
  split_dir: splits
  node_encoder: true
  node_encoder_name: GraphStats
  set_graph_stats: true
  graph_stats:
    - degree
    - eccentricity
    - cluster_coefficient
    - triangle_count
  multiprocessing: true
  label: false             # ← true only if you have ground-truth labels

train:
  mode: copt
  task: myproblem          # ← MUST match key in EVAL_FUNCTION_DICT
  batch_size: 256
  val_period: 1
  ckpt_period: 100
  ckpt_best: true

model:
  type: gnn
  loss_fun: myproblem_loss # ← MUST match @register_loss decorator name
  edge_decoding: dot
  graph_pooling: mean

gnn:
  head: copt_inductive_node   # node-level head; use copt_graph for graph-level
  layers_pre_mp: 4
  layers_mp: 16
  layers_post_mp: 1
  dim_inner: 256
  layer_type: gcnconv
  stage_type: skipsum
  batchnorm: true
  act: elu
  last_act: sigmoid           # output activation: sigmoid recommended for [0,1]
  dropout: 0.3
  agg: mean
  normalize_adj: false

optim:
  base_lr: 0.003
  max_epoch: 200
  optimizer: adamW
  weight_decay: 1e-5
  scheduler: cosine_with_warmup
  num_warmup_epochs: 5

# Loss hyperparameters (must match cfg.myproblem_loss in loss_config.py)
myproblem_loss:
  beta: 0.1

# Metric hyperparameters (must match cfg.metrics.myproblem in loss_config.py)
metrics:
  myproblem:
    dec_length: 300
    num_seeds: 1
```

---

## Step 6 (Optional) — Add a Data Transform for Ground-Truth Labels

**File to modify:** `graphgym/loader/master_loader.py`

Skip this step if your problem is **unsupervised** (no ground-truth labels needed, `dataset.label: false`).

If you need ground-truth solutions (e.g., for ratio metrics or supervised training), add a transform:

```python
def set_myproblem(data):
    """Compute and attach ground-truth solution to graph data object."""
    import networkx as nx
    from torch_geometric.utils import to_networkx

    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()

    # Compute ground-truth using a classical solver
    solution = my_classical_solver(g)

    # Attach to data object (accessible as data.my_solution during training)
    data.my_solution = torch.tensor(solution, dtype=torch.float)
    return data
```

Then register the transform in the loader pipeline (around line 124, inside `load_dataset_master()`):

```python
if cfg.train.task == 'myproblem':
    tf_list.append(set_myproblem)
```

---

\newpage

# Complete Checklist

Here is a summary checklist for adding a new problem named `myproblem`:

| # | Action | File | Key Names |
|---|---|---|---|
| 1 | Add loss function decorated with `@register_loss` | `graphgym/loss/copt_loss.py` | `"myproblem_loss"` |
| 2 | Add evaluation / metric function | `utils/metrics.py` | `myproblem_size_pyg` |
| 3 | Register config hyperparameters | `graphgym/config/loss_config.py` | `cfg.myproblem_loss`, `cfg.metrics.myproblem` |
| 4 | Add to eval function dictionaries | `modules/utils/spaces.py` | `"myproblem"` key |
| 5 | Create YAML experiment config | `configs/myproblem.yaml` | `train.task`, `model.loss_fun` |
| 6* | Add ground-truth data transform | `graphgym/loader/master_loader.py` | `set_myproblem` |

\* Step 6 is required only if `dataset.label: true`.

---

# Running Your New Problem

Once all steps are complete, run training with:

```bash
cd /path/to/copt
python main.py --config configs/myproblem.yaml
```

Results and checkpoints are saved to `results/myproblem/`.

---

\newpage

# Name Consistency Map

The most common integration error is a **name mismatch**. Every name used across files must be consistent:

```
  YAML: model.loss_fun: "myproblem_loss"
                │
                └──► @register_loss("myproblem_loss")      [copt_loss.py]
                         in: loss_params = cfg["myproblem_loss"]   [copt_module.py:26]

  YAML: train.task: "myproblem"
                │
                ├──► EVAL_FUNCTION_DICT["myproblem"]        [spaces.py]
                ├──► cfg.metrics["myproblem"]               [loss_config.py + copt_module.py:39]
                └──► cfg.train.task == 'myproblem'          [master_loader.py, optional]
```

If you see a `KeyError` at startup, trace it back through this map.

---

# Reference: Existing Loss Functions

## MaxCut (`maxcut_loss_pyg`, line 47)

```python
@register_loss("maxcut_loss")
def maxcut_loss_pyg(data):
    x = (data.x - 0.5) * 2
    src, dst = data.edge_index[0], data.edge_index[1]
    return torch.sum(x[src] * x[dst]) / len(data.batch.unique())
```

Objective: maximise the number of edges crossing a binary partition.

## MaxClique (`maxclique_loss_pyg`, line 21)

```python
@register_loss("maxclique_loss")
def maxclique_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()
    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]
        loss1 = torch.sum(data.x[src] * data.x[dst])
        loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
        loss += (- loss1 + beta * loss2) * data.num_nodes
    return loss / batch.size(0)
```

Objective: find the largest complete subgraph. `loss1` rewards edges within the selection; `loss2` penalises non-edges (incomplete subgraph constraint).

## MDS (`mds_loss_pyg`, line 120)

```python
@register_loss("mds_loss")
def mds_loss_pyg(data, beta=1.0):
    batch_size = data.batch.max() + 1.0
    p = data.x.squeeze()
    edge_index = remove_self_loops(data.edge_index)[0]
    row, col = edge_index[0], edge_index[1]
    loss = p.sum() + beta * (
        scatter(torch.log1p(-p)[row], index=col, reduce='sum').exp() * (1 - p)
    ).sum()
    return loss / batch_size
```

Objective: find the smallest set that dominates every node. `p.sum()` minimises set size; the scatter term penalises nodes not covered by their neighbourhood.

## MIS (`mis_loss_pyg`, line 163)

```python
@register_loss("mis_loss")
def mis_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()
    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]
        loss1 = torch.sum(data.x[src] * data.x[dst])
        loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
        loss += (- loss2 + beta * loss1) * data.num_nodes
    return loss / batch.size(0)
```

This is the **dual** of MaxClique: MIS maximises the independent set size (`-loss2`) while penalising adjacent selected nodes (`loss1`).

---

\newpage

# Quick Tips

1. **Start with an existing loss as a template.** If your problem is a maximisation problem over node sets, MIS or MaxClique are the closest templates. If it involves edge weights, start from MaxCut.

2. **Normalise by graph size.** Multiply the per-graph loss by `data.num_nodes` and divide by `batch.size(0)` to prevent gradients from being dominated by large graphs.

3. **Keep the decoder simple first.** A greedy rank-and-prune decoder (sort by `batch.x`, add while feasible) is sufficient to get a quality signal. You can refine it later.

4. **Match `last_act` to your output range.** `sigmoid` maps outputs to `[0, 1]`, which is what most loss functions here expect. If your loss needs a different range, adjust `gnn.last_act` in the YAML.

5. **Use `dataset.label: false` by default.** Unsupervised learning (pure loss minimisation) avoids the need for expensive ground-truth computation and works well for most combinatorial problems in this codebase.

6. **Check the config names before running.** The most common error is a `KeyError` from a name mismatch between `model.loss_fun` in YAML and the `@register_loss` decorator, or between `train.task` and the key in `EVAL_FUNCTION_DICT`.
