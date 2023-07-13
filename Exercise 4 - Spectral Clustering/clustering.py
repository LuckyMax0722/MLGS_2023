from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def deterministic_eigsh(A, **kwargs):
    np.random.seed(0)
    kwargs['v0'] = np.random.rand(min(A.shape))
    return eigsh(A, **kwargs)


def eigsh_help():
    help(eigsh)


def labels_to_list_of_clusters(z: np.array) -> List[List[int]]:
    """Convert predicted label vector to a list of clusters in the graph.
    This function is already implemented, nothing to do here.
    
    Parameters
    ----------
    z : np.array, shape [N]
        Predicted labels.
        
    Returns
    -------
    list_of_clusters : list of lists
        Each list contains ids of nodes that belong to the same cluster.
        Each node may appear in one and only one partition.
    
    Examples
    --------
    >>> z = np.array([0, 0, 1, 1, 0])
    >>> labels_to_list_of_clusters(z)
    [[0, 1, 4], [2, 3]]
    
    """
    return [np.where(z == c)[0] for c in np.unique(z)]


def construct_laplacian(A: sp.csr_matrix, norm_laplacian: bool) -> sp.csr_matrix:
    """Construct Laplacian of a graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    norm_laplacian : bool
        Whether to construct the normalized graph Laplacian or not.
        If True, construct the normalized (symmetrized) Laplacian, L = I - D^{-1/2} A D^{-1/2}.
        If False, construct the unnormalized Laplacian, L = D - A.
        
    Returns
    -------
    L : scipy.sparse.csr_matrix, shape [N, N]
        Laplacian of the graph.
        
    """
    ##########################################################
    # YOUR CODE HERE
    if norm_laplacian:
        D = 1 / np.sqrt(A.sum(axis=1)).A.flatten()
        D = sp.diags(D)
        L = sp.eye(A.shape[0]) - D.dot(A).dot(D)
    else:
        D = A.sum(axis=1).A.flatten()
        D = sp.diags(D)
        L = D - A

    ##########################################################
    return L


def spectral_embedding(A: sp.csr_matrix, num_clusters: int, norm_laplacian: bool) -> np.array:
    """Compute spectral embedding of nodes in the given graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    num_clusters : int
        Number of clusters to detect in the data.
    norm_laplacian : bool, default False
        Whether to use the normalized graph Laplacian or not.
        
    Returns
    -------
    embedding : np.array, shape [N, num_clusters]
        Spectral embedding for the given graph.
        Each row represents the spectral embedding of a given node.
        The rows have to be sorted in ascending order w.r.t. the corresponding eigenvalues.
    
    """
    if (A != A.T).sum() != 0:
        raise ValueError("Spectral embedding doesn't work if the adjacency matrix is not symmetric.")
    if num_clusters < 2:
        raise ValueError("The clustering requires at least two clusters.")
    if num_clusters > A.shape[0]:
        raise ValueError(f"We can have at most {A.shape[0]} clusters (number of nodes).")

    ##########################################################
    # YOUR CODE HERE
    L = construct_laplacian(A, norm_laplacian)
    _, embedding = eigsh(L, k=num_clusters, which='SM')
    ##########################################################

    return embedding


def compute_ratio_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the ratio cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    ratio_cut : float
        Value of the cut for the given partition of the graph.
        
    """

    ##########################################################
    # YOUR CODE HERE
    z_clusters = labels_to_list_of_clusters(z)
    ratio_cut = 0

    for i in z_clusters:
        z_clusters_delete = np.delete(range(A.shape[0]), i, None)
        ratio_cut =  ratio_cut + A[i][:, z_clusters_delete].sum() / len(i)

    ##########################################################
    return ratio_cut


def compute_normalized_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the normalized cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    norm_cut : float
        Value of the normalized cut for the given partition of the graph.
        
    """

    ##########################################################
    # YOUR CODE HERE
    z_clusters = labels_to_list_of_clusters(z)

    sum_of_column_A = np.squeeze(np.array(A.sum(1)))

    norm_cut = 0

    for i in z_clusters:
        z_clusters_delete = np.delete(range(A.shape[0]), i, None)
        v = 0
        v = v + sum_of_column_A[i].sum()
        if v == 0: v = 1

        norm_cut = norm_cut + A[i][:, z_clusters_delete].sum() / v

    ##########################################################
    return norm_cut
