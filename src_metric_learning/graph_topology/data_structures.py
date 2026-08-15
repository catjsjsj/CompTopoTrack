

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FramePairTopology:
    
    frame_t: int
    frame_tp1: int
    cell_ids_t: np.ndarray        
    cell_ids_tp1: np.ndarray      
    centroids_t: np.ndarray       
    centroids_tp1: np.ndarray     
    adj_matrix: np.ndarray        
    true_link_mask: np.ndarray    
    cluster_ids: np.ndarray       
    cross_track_mask: np.ndarray  
    cell_influence: np.ndarray    
    cluster_influence: np.ndarray 
    pair_weights: np.ndarray      
    closure_cross_mask: Optional[np.ndarray] = None  
    closure_component_sizes: Optional[np.ndarray] = None  
    component_labels: Optional[np.ndarray] = None  
    component_sizes: Optional[np.ndarray] = None  

    @property
    def M(self) -> int:
        return len(self.cell_ids_t)

    @property
    def N(self) -> int:
        return len(self.cell_ids_tp1)

    @property
    def neg_mask(self) -> np.ndarray:
        
        return self.adj_matrix & ~self.true_link_mask

    @property
    def closure_mask(self) -> np.ndarray:
        
        if self.closure_cross_mask is None:
            return np.zeros_like(self.adj_matrix, dtype=bool)
        return self.closure_cross_mask

    @property
    def topology_pair_mask(self) -> np.ndarray:
        
        return self.adj_matrix | self.closure_mask

    @property
    def topology_neg_mask(self) -> np.ndarray:
        
        return self.topology_pair_mask & ~self.true_link_mask

    @property
    def n_positive_pairs(self) -> int:
        return int(self.true_link_mask.sum())

    @property
    def n_negative_pairs(self) -> int:
        return int(self.neg_mask.sum())

    @property
    def n_clusters(self) -> int:
        valid = self.cluster_ids[self.cluster_ids >= 0]
        return len(np.unique(valid)) if len(valid) > 0 else 0

    @property
    def n_cross_track_pairs(self) -> int:
        return int(self.cross_track_mask.sum())

    def validate(self) -> bool:
        
        M, N = self.M, self.N
        shape_2d = (M, N)
        assert self.adj_matrix.shape == shape_2d, \
            f"adj_matrix shape {self.adj_matrix.shape} != {(M, N)}"
        assert self.true_link_mask.shape == shape_2d
        assert self.cluster_ids.shape == shape_2d
        assert self.cross_track_mask.shape == shape_2d
        assert self.cluster_influence.shape == shape_2d
        assert self.pair_weights.shape == shape_2d
        if self.closure_cross_mask is not None:
            assert self.closure_cross_mask.shape == shape_2d
        if self.closure_component_sizes is not None:
            assert self.closure_component_sizes.shape == shape_2d
        if self.component_labels is not None:
            assert self.component_labels.shape == (M + N,)
        if self.component_sizes is not None:
            assert self.component_sizes.shape == (M + N,)
        assert self.centroids_t.shape == (M, 2)
        assert self.centroids_tp1.shape == (N, 2)
        assert self.cell_influence.shape == (M + N,)
        assert len(self.cell_ids_t) == M
        assert len(self.cell_ids_tp1) == N
        
        assert np.all(self.true_link_mask <= self.adj_matrix), \
            "true_link_mask has edges outside adj_matrix"
        
        assert np.all(self.cross_track_mask <= self.adj_matrix), \
            "cross_track_mask has edges outside adj_matrix"
        return True


@dataclass
class SequenceTopology:
    
    sequence_name: str
    num_frames: int
    tau: float
    roi_row: float
    roi_col: float
    cell_influence_scores: Dict[int, float] = field(default_factory=dict)
    frame_pairs: List[FramePairTopology] = field(default_factory=list)

    @property
    def total_frame_pairs(self) -> int:
        return len(self.frame_pairs)

    @property
    def total_cells(self) -> int:
        return len(self.cell_influence_scores)

    def get_frame_pair(self, frame_t: int) -> Optional[FramePairTopology]:
        
        for fp in self.frame_pairs:
            if fp.frame_t == frame_t:
                return fp
        return None

    def validate(self) -> bool:
        
        for fp in self.frame_pairs:
            fp.validate()
        
        for i in range(len(self.frame_pairs) - 1):
            assert self.frame_pairs[i].frame_t < self.frame_pairs[i+1].frame_t, \
                "frame_pairs not sorted by frame_t"
        return True
