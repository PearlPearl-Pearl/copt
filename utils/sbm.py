import numpy as np


def stochastic_block_model(n, k, p_in, p_out, seed=None):
    """
    Generate a k-block Stochastic Block Model (SBM).

    Parameters
    ----------
    n : int    — number of nodes
    k : int    — number of blocks / communities
    p_in : float  — intra-block edge probability
    p_out : float — inter-block edge probability
    seed : int or None

    Returns
    -------
    A : ndarray (n, n) — symmetric adjacency matrix
    labels : ndarray (n,) — ground-truth block labels
    """
    if seed is not None:
        np.random.seed(seed)

    # balanced block sizes
    sizes = [n // k] * k
    for i in range(n % k):
        sizes[i] += 1

    labels = []
    for c in range(k):
        labels += [c] * sizes[c]
    labels = np.array(labels)

    perm = np.random.permutation(n)
    labels = labels[perm]

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if np.random.rand() < p:
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A, labels
