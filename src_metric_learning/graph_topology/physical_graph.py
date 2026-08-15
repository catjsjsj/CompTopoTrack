


import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from .data_structures import FramePairTopology


class PhysicalGraphBuilder:
    

    def __init__(
        self,
        roi_row: float,
        roi_col: float,
        merge_parent_child: bool = False,
    ):
        
        self.roi_row = roi_row
        self.roi_col = roi_col
        self.merge_parent_child = merge_parent_child

    
    
    

    def build_frame_pair(
        self,
        df_t: pd.DataFrame,
        df_tp1: pd.DataFrame,
        frame_t: int,
        frame_tp1: int,
    ) -> FramePairTopology:
        
        M = len(df_t)
        N = len(df_tp1)

        
        if M > 0:
            centroids_t = df_t[["centroid_row", "centroid_col"]].values.astype(np.float64)
        else:
            centroids_t = np.empty((0, 2), dtype=np.float64)

        if N > 0:
            centroids_tp1 = df_tp1[["centroid_row", "centroid_col"]].values.astype(np.float64)
        else:
            centroids_tp1 = np.empty((0, 2), dtype=np.float64)

        
        adj_matrix = self._compute_adjacency(centroids_t, centroids_tp1)

        
        
        
        ids_t = df_t["id"].values if M > 0 else np.array([], dtype=int)
        ids_tp1 = df_tp1["id"].values if N > 0 else np.array([], dtype=int)
        true_link_mask = self._compute_true_links(adj_matrix, ids_t, ids_tp1)

        
        topology = FramePairTopology(
            frame_t=frame_t,
            frame_tp1=frame_tp1,
            cell_ids_t=ids_t.copy(),
            cell_ids_tp1=ids_tp1.copy(),
            centroids_t=centroids_t.copy(),
            centroids_tp1=centroids_tp1.copy(),
            adj_matrix=adj_matrix,
            true_link_mask=true_link_mask,
            cluster_ids=np.full((M, N), -1, dtype=np.int32),
            cross_track_mask=np.zeros((M, N), dtype=bool),
            cell_influence=np.zeros(M + N, dtype=np.float64),
            cluster_influence=np.zeros((M, N), dtype=np.float64),
            pair_weights=np.zeros((M, N), dtype=np.float64),
            closure_cross_mask=np.zeros((M, N), dtype=bool),
            closure_component_sizes=np.zeros((M, N), dtype=np.int32),
            component_labels=np.full(M + N, -1, dtype=np.int32),
            component_sizes=np.zeros(M + N, dtype=np.int32),
        )
        return topology

    @classmethod
    def compute_roi_from_displacements(
        cls,
        org_df_cells_list: List[pd.DataFrame],
        mul_vals: Tuple[float, float] = (2.0, 2.0),
    ) -> Tuple[float, float]:
        
        all_diffs_row = []
        all_diffs_col = []

        for df in org_df_cells_list:
            diffs_r, diffs_c = cls._collect_true_link_displacements(df)
            all_diffs_row.extend(diffs_r)
            all_diffs_col.extend(diffs_c)

        if len(all_diffs_row) == 0:
            
            return cls._fallback_roi(org_df_cells_list, mul_vals)

        diffs_row = np.abs(np.array(all_diffs_row, dtype=np.float64))
        diffs_col = np.abs(np.array(all_diffs_col, dtype=np.float64))

        roi_row = float(diffs_row.max() + mul_vals[0] * diffs_row.std())
        roi_col = float(diffs_col.max() + mul_vals[1] * diffs_col.std())

        return roi_row, roi_col

    @classmethod
    def compute_tau(
        cls,
        org_df_cells_list: List[pd.DataFrame],
    ) -> float:
        
        all_distances = []
        for df in org_df_cells_list:
            diffs_r, diffs_c = cls._collect_true_link_displacements(df)
            for dr, dc in zip(diffs_r, diffs_c):
                all_distances.append(np.sqrt(dr**2 + dc**2))

        if len(all_distances) == 0:
            return 1.0
        return float(np.median(all_distances))

    
    
    

    def _compute_adjacency(
        self,
        centroids_t: np.ndarray,
        centroids_tp1: np.ndarray,
    ) -> np.ndarray:
        
        
        row_diff = np.abs(centroids_t[:, 0:1] - centroids_tp1[np.newaxis, :, 0])
        col_diff = np.abs(centroids_t[:, 1:2] - centroids_tp1[np.newaxis, :, 1])

        adj = (row_diff <= self.roi_row) & (col_diff <= self.roi_col)
        return adj

    @staticmethod
    def _compute_true_links(
        adj_matrix: np.ndarray,
        ids_t: np.ndarray,
        ids_tp1: np.ndarray,
    ) -> np.ndarray:
        
        
        id_match = ids_t[:, np.newaxis] == ids_tp1[np.newaxis, :]
        return adj_matrix & id_match

    @staticmethod
    def _collect_true_link_displacements(
        df: pd.DataFrame,
    ) -> Tuple[List[float], List[float]]:
        
        diffs_row = []
        diffs_col = []
        frames = sorted(df["frame_num"].unique())

        for cell_id in df["id"].unique():
            cell_data = df[df["id"] == cell_id].sort_values("frame_num")
            if len(cell_data) < 2:
                continue

            rows = cell_data["centroid_row"].values
            cols = cell_data["centroid_col"].values
            fns = cell_data["frame_num"].values

            for i in range(len(fns) - 1):
                
                if fns[i + 1] == fns[i] + 1:
                    diffs_row.append(float(rows[i + 1] - rows[i]))
                    diffs_col.append(float(cols[i + 1] - cols[i]))

        return diffs_row, diffs_col

    @staticmethod
    def _fallback_roi(
        org_df_cells_list: List[pd.DataFrame],
        mul_vals: Tuple[float, float],
    ) -> Tuple[float, float]:
        
        max_row = 0.0
        max_col = 0.0
        for df in org_df_cells_list:
            if "min_row_bb" in df.columns and "max_row_bb" in df.columns:
                max_row = max(
                    max_row,
                    float(np.abs(df["min_row_bb"] - df["max_row_bb"]).max()),
                )
                max_col = max(
                    max_col,
                    float(np.abs(df["min_col_bb"] - df["max_col_bb"]).max()),
                )
        roi_row = max_row * mul_vals[0] if max_row > 0 else 50.0
        roi_col = max_col * mul_vals[1] if max_col > 0 else 50.0
        return roi_row, roi_col






def remap_parent_child_ids(
    df: pd.DataFrame,
    parent_child_map: Optional[dict] = None,
) -> pd.DataFrame:
    
    if parent_child_map is None:
        return df

    df = df.copy()
    df["id"] = df["id"].replace(parent_child_map)
    return df


def build_parent_child_map_from_tracks(
    df: pd.DataFrame,
) -> dict:
    
    
    return {}
