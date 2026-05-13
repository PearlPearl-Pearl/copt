import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans


def spectral_partition(A, k, kmeans_seed=0, n_init=10):
    """
    Spectral Graph Partitioning heuristic.

    Parameters
    ----------
    A : ndarray (n, n)   — symmetric adjacency matrix
    k : int              — number of partitions
    kmeans_seed : int    — random seed for k-means
    n_init : int         — number of k-means restarts

    Returns
    -------
    labels : ndarray (n,) — partition assignment per node (0 … k-1)
    """
    n = A.shape[0]

    # Degree matrix and unnormalised Laplacian
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A

    # k smallest eigenvectors of L (index 0 … k-1)
    # eigh returns them sorted by eigenvalue, so index 0 is the constant vector
    eigenvalues, eigenvectors = scipy.linalg.eigh(L, subset_by_index=[0, k - 1])

    # Discard u_1 (constant); use u_2 … u_k for embedding
    # For k=2 this gives a single column; for k>2 it gives k-1 columns
    U = eigenvectors[:, 1:]   # shape (n, k-1)

    if U.shape[1] == 0:
        # degenerate case: k=1, every node in the same partition
        return np.zeros(n, dtype=int)

    # k-means on the rows of U
    kmeans = KMeans(n_clusters=k, random_state=kmeans_seed, n_init=n_init)
    labels = kmeans.fit_predict(U)

    return labels


def normalised_cut(A, labels, k):
    """
    Compute the normalised cut value for a given partition.

    NCut = sum_i  cut(P_i, V\P_i) / vol(P_i)
    where vol(P_i) = sum of degrees of nodes in P_i.
    """
    ncut = 0.0
    degrees = A.sum(axis=1)
    for i in range(k):
        mask = labels == i
        vol = degrees[mask].sum()
        if vol == 0:
            continue
        cut = A[mask][:, ~mask].sum()
        ncut += cut / vol
    return ncut
