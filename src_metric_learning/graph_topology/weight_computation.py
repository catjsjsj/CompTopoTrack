

import os
import pickle
import hashlib
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from .data_structures import SequenceTopology
from .physical_graph import PhysicalGraphBuilder
from .competition_cluster import find_competition_clusters
from .influence_weight import compute_sequence_influence


class TopologyWeightComputer:
    

    def __init__(
        self,
        roi_row: float,
        roi_col: float,
        gamma: float = 1.0,
        alpha_max: float = 1.0,
        curriculum_epochs: int = 500,
        merge_parent_child: bool = False,
        horizon: int = 6,
        influence_type: str = "degree",
        influence_scope: str = "cluster_gated",
        gamma_direct: Optional[float] = None,
        gamma_closure: Optional[float] = None,
        degree_weight: Optional[float] = None,
        closure_max_component_size: Optional[int] = 50,
    ):
        self.roi_row = roi_row
        self.roi_col = roi_col
        self.gamma = gamma
        self.alpha_max = alpha_max
        self.curriculum_epochs = max(curriculum_epochs, 1)
        self.gamma_direct = float(0.2 if gamma_direct is None else gamma_direct)
        self.gamma_closure = float(gamma if gamma_closure is None else gamma_closure)
        self.degree_weight = float(alpha_max if degree_weight is None else degree_weight)
        self.closure_max_component_size = closure_max_component_size
        self.merge_parent_child = merge_parent_child
        self.horizon = horizon
        self.influence_type = influence_type
        valid_scopes = {"cluster_gated", "global"}
        if influence_scope not in valid_scopes:
            raise ValueError(
                f"influence_scope must be one of {sorted(valid_scopes)}, got {influence_scope}"
            )
        self.influence_scope = influence_scope
        self.graph_builder = PhysicalGraphBuilder(
            roi_row=roi_row, roi_col=roi_col,
            merge_parent_child=merge_parent_child,
        )

    
    
    

    def compute(
        self,
        org_df_cells: pd.DataFrame,
        sequence_name: str = "",
        enabled_layers: List[int] = None,
        tau: Optional[float] = None,
        parent_child_map: Optional[dict] = None,
    ) -> SequenceTopology:
        
        if enabled_layers is None:
            enabled_layers = [1, 2, 3]

        
        if parent_child_map is not None:
            from .physical_graph import remap_parent_child_ids
            org_df_cells = remap_parent_child_ids(org_df_cells, parent_child_map)

        frames = sorted(org_df_cells["frame_num"].unique())

        
        frame_pair_topologies = []
        for idx in range(len(frames) - 1):
            ft, ftp1 = frames[idx], frames[idx + 1]
            df_t = org_df_cells[org_df_cells["frame_num"] == ft]
            df_tp1 = org_df_cells[org_df_cells["frame_num"] == ftp1]
            fp = self.graph_builder.build_frame_pair(df_t, df_tp1, int(ft), int(ftp1))
            frame_pair_topologies.append(fp)

        
        if 2 in enabled_layers and len(frame_pair_topologies) > 0:
            frame_pair_topologies = [
                find_competition_clusters(
                    fp,
                    max_component_size=self.closure_max_component_size,
                )
                for fp in frame_pair_topologies
            ]

        
        if tau is None and 3 in enabled_layers:
            tau = PhysicalGraphBuilder.compute_tau([org_df_cells])
        elif tau is None:
            tau = 1.0

        
        if 3 in enabled_layers and len(frames) >= 2:
            seq_topo = compute_sequence_influence(
                org_df_cells, frame_pair_topologies,
                self.roi_row, self.roi_col, tau,
                horizon=self.horizon,
                method=self.influence_type,
                gamma=0.0,
            )
        else:
            
            seq_topo = SequenceTopology(
                sequence_name=sequence_name,
                num_frames=len(frames),
                tau=tau,
                roi_row=self.roi_row,
                roi_col=self.roi_col,
                cell_influence_scores={},
                frame_pairs=frame_pair_topologies,
            )
            for fp in frame_pair_topologies:
                fp.cell_influence = np.ones(fp.M + fp.N)
                fp.cluster_influence = np.ones((fp.M, fp.N))
                fp.cluster_influence[~fp.adj_matrix] = 0.0

        seq_topo.sequence_name = sequence_name

        
        self._recompute_weights_for_epoch(seq_topo, epoch=0, enabled_layers=enabled_layers)

        return seq_topo

    
    
    

    def update_weights_for_epoch(
        self,
        topology: SequenceTopology,
        epoch: int,
        enabled_layers: List[int] = None,
    ):
        
        if enabled_layers is None:
            enabled_layers = [1, 2, 3]
        self._recompute_weights_for_epoch(topology, epoch, enabled_layers)

    def _recompute_weights_for_epoch(
        self,
        topology: SequenceTopology,
        epoch: int,
        enabled_layers: List[int],
    ):
        
        for fp in topology.frame_pairs:
            direct = fp.adj_matrix.astype(np.float64)
            if 2 not in enabled_layers:
                fp.pair_weights = direct.copy()
                continue

            closure_mask = getattr(fp, "closure_cross_mask", None)
            if closure_mask is None:
                closure_mask = np.zeros_like(fp.adj_matrix, dtype=bool)
            closure = closure_mask.astype(np.float64)
            pair_mask = (fp.adj_matrix | closure_mask).astype(np.float64)

            weights = pair_mask * (
                1.0
                + self.gamma_direct * direct
                + self.gamma_closure * closure
            )

            if 3 in enabled_layers and self.degree_weight != 0:
                influence_t = fp.cell_influence[:fp.M]
                influence_tp1 = fp.cell_influence[fp.M:]
                pair_influence = np.outer(influence_t, influence_tp1)
                weights = weights + pair_mask * self.degree_weight * pair_influence

            
            
            weights[fp.true_link_mask] = 1.0

            fp.pair_weights = weights

        return topology

    def _curriculum_alpha(self, epoch: int, enabled_layers: List[int]) -> float:
        
        if 3 not in enabled_layers:
            return 0.0
        return self.degree_weight

    
    
    

    def save(self, topology: SequenceTopology, path: str):
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(topology, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> SequenceTopology:
        
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def cache_key(
        roi_row: float,
        roi_col: float,
        gamma: float,
        alpha_max: float,
        curriculum_epochs: int,
        merge_parent_child: bool,
        influence_type: str,
        influence_scope: str,
        df_hash: str,
        gamma_direct: float = 0.2,
        gamma_closure: float = 0.1,
        degree_weight: float = 0.1,
        closure_max_component_size: Optional[int] = 50,
    ) -> str:
        
        params = {
            "roi_row": roi_row, "roi_col": roi_col,
            "gamma": gamma, "alpha_max": alpha_max,
            "curriculum_epochs": curriculum_epochs,
            "merge_parent_child": merge_parent_child,
            "influence_type": influence_type,
            "influence_scope": influence_scope,
            "gamma_direct": gamma_direct,
            "gamma_closure": gamma_closure,
            "degree_weight": degree_weight,
            "closure_max_component_size": closure_max_component_size,
            "df_hash": df_hash,
        }
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]






def compute_topology_for_sequence(
    df: pd.DataFrame,
    roi_row: float,
    roi_col: float,
    gamma: float = 1.0,
    alpha_max: float = 1.0,
    curriculum_epochs: int = 500,
    enabled_layers: List[int] = None,
    merge_parent_child: bool = False,
    parent_child_map: Optional[dict] = None,
    horizon: int = 6,
    influence_type: str = "degree",
    influence_scope: str = "cluster_gated",
    gamma_direct: Optional[float] = None,
    gamma_closure: Optional[float] = None,
    degree_weight: Optional[float] = None,
    closure_max_component_size: Optional[int] = 50,
) -> SequenceTopology:
    
    computer = TopologyWeightComputer(
        roi_row=roi_row, roi_col=roi_col,
        gamma=gamma, alpha_max=alpha_max,
        curriculum_epochs=curriculum_epochs,
        merge_parent_child=merge_parent_child,
        horizon=horizon,
        influence_type=influence_type,
        influence_scope=influence_scope,
        gamma_direct=gamma_direct,
        gamma_closure=gamma_closure,
        degree_weight=degree_weight,
        closure_max_component_size=closure_max_component_size,
    )
    return computer.compute(
        df, enabled_layers=enabled_layers,
        parent_child_map=parent_child_map,
    )
