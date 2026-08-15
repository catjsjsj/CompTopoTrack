

from typing import Optional

import numpy as np
from .data_structures import FramePairTopology


def find_competition_clusters(
    topology: FramePairTopology,
    max_component_size: Optional[int] = None,
) -> FramePairTopology:
    
    M, N = topology.adj_matrix.shape
    total_nodes = M + N

    if M == 0 or N == 0:
        topology.closure_cross_mask = np.zeros((M, N), dtype=bool)
        topology.closure_component_sizes = np.zeros((M, N), dtype=np.int32)
        topology.component_labels = np.full(total_nodes, -1, dtype=np.int32)
        topology.component_sizes = np.zeros(total_nodes, dtype=np.int32)
        return topology

    
    row_indices, col_indices = np.where(topology.adj_matrix)

    if len(row_indices) == 0:
        
        
        topology.closure_cross_mask = np.zeros((M, N), dtype=bool)
        topology.closure_component_sizes = np.zeros((M, N), dtype=np.int32)
        topology.component_labels = np.full(total_nodes, -1, dtype=np.int32)
        topology.component_sizes = np.zeros(total_nodes, dtype=np.int32)
        return topology

    
    
    right_nodes = col_indices + M
    left_nodes = row_indices

    
    edges_src = np.concatenate([left_nodes, right_nodes])
    edges_dst = np.concatenate([right_nodes, left_nodes])

    
    labels = _connected_components(total_nodes, edges_src, edges_dst)
    labels = labels.astype(np.int32, copy=False)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.int32)
    node_component_sizes = counts[labels]
    topology.component_labels = labels
    topology.component_sizes = node_component_sizes

    
    
    
    edge_labels = labels[left_nodes]
    edge_component_labels = np.unique(edge_labels)
    max_size = None if max_component_size is None or max_component_size <= 0 else int(max_component_size)

    cluster_ids = np.full((M, N), -1, dtype=np.int32)
    closure_cross_mask = np.zeros((M, N), dtype=bool)
    closure_component_sizes = np.zeros((M, N), dtype=np.int32)

    for cid in edge_component_labels:
        comp_size = int(counts[cid])
        if max_size is not None and comp_size > max_size:
            continue

        left_members = np.where(labels[:M] == cid)[0]
        right_members = np.where(labels[M:] == cid)[0]
        if len(left_members) == 0 or len(right_members) == 0:
            continue

        direct_in_component = (edge_labels == cid)
        cluster_ids[row_indices[direct_in_component], col_indices[direct_in_component]] = cid
        closure_cross_mask[np.ix_(left_members, right_members)] = True
        closure_component_sizes[np.ix_(left_members, right_members)] = comp_size

    topology.cluster_ids = cluster_ids
    topology.closure_cross_mask = closure_cross_mask
    topology.closure_component_sizes = closure_component_sizes

    
    
    
    cross_track_mask = np.zeros((M, N), dtype=bool)
    for i, j in zip(row_indices, col_indices):
        if not topology.true_link_mask[i, j]:
            cross_track_mask[i, j] = True

    topology.cross_track_mask = cross_track_mask

    return topology


def _connected_components(
    n_nodes: int,
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
) -> np.ndarray:
    
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as cc

        data = np.ones(len(edges_src), dtype=np.int32)
        adj = csr_matrix((data, (edges_src, edges_dst)), shape=(n_nodes, n_nodes))
        n_components, labels = cc(adj, directed=False)
        return labels

    except ImportError:
        return _unionfind_connected_components(n_nodes, edges_src, edges_dst)


def _unionfind_connected_components(
    n_nodes: int,
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
) -> np.ndarray:
    
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    for u, v in zip(edges_src, edges_dst):
        union(int(u), int(v))

    
    comp_id = {}
    labels = np.zeros(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        root = find(i)
        if root not in comp_id:
            comp_id[root] = len(comp_id)
        labels[i] = comp_id[root]

    return labels


def compute_transitive_competition_ratio(
    topology: FramePairTopology,
) -> float:
    
    if topology.n_clusters == 0:
        return 0.0

    M, N = topology.adj_matrix.shape

    
    
    
    valid_mask = topology.cluster_ids >= 0
    rows, cols = np.where(valid_mask)
    cids = topology.cluster_ids[rows, cols]

    from collections import defaultdict

    left_members = defaultdict(set)
    right_members = defaultdict(set)

    for i, j, cid in zip(rows, cols, cids):
        left_members[cid].add(int(i))
        right_members[cid].add(int(j))

    direct_comp = 0   
    indirect_comp = 0  

    for cid in left_members:
        lset = left_members[cid]
        rset = right_members[cid]

        for i in lset:
            for j in rset:
                if topology.true_link_mask[i, j]:
                    continue  
                if topology.adj_matrix[i, j]:
                    direct_comp += 1
                else:
                    
                    indirect_comp += 1

    total = direct_comp + indirect_comp
    if total == 0:
        return 0.0
    return indirect_comp / total
