

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .data_structures import FramePairTopology, SequenceTopology


def compute_sequence_influence(
    org_df_cells: pd.DataFrame,
    frame_pair_topologies: List[FramePairTopology],
    roi_row: float,
    roi_col: float,
    tau: Optional[float] = None,
    horizon: int = 6,
    method: str = "degree",
    gamma: float = 1.0,
) -> SequenceTopology:
    
    if method == "uniform":
        return _uniform_influence(frame_pair_topologies, roi_row, roi_col)
    if method == "degree":
        return _compute_degree_influence(frame_pair_topologies, roi_row, roi_col, gamma)
    if method == "random_walk":
        return _compute_random_walk_influence(
            org_df_cells, frame_pair_topologies, roi_row, roi_col, tau, horizon,
        )
    raise ValueError(f"Unsupported influence method: {method}")


def _uniform_influence(
    frame_pair_topologies: List[FramePairTopology],
    roi_row: float,
    roi_col: float,
) -> SequenceTopology:
    
    cell_influence_scores: Dict[int, float] = {}
    for fp in frame_pair_topologies:
        fp.cell_influence = np.ones(fp.M + fp.N, dtype=np.float64)
        fp.cluster_influence = fp.adj_matrix.astype(np.float64)
        for cid in fp.cell_ids_t:
            cell_influence_scores[int(cid)] = 1.0
        for cid in fp.cell_ids_tp1:
            cell_influence_scores[int(cid)] = 1.0

    return SequenceTopology(
        sequence_name="", num_frames=0, tau=0.0,
        roi_row=roi_row, roi_col=roi_col,
        cell_influence_scores=cell_influence_scores,
        frame_pairs=frame_pair_topologies,
    )






def _compute_degree_influence(
    frame_pair_topologies: List[FramePairTopology],
    roi_row: float,
    roi_col: float,
    gamma: float = 1.0,
) -> SequenceTopology:
    
    num_pairs = len(frame_pair_topologies)
    if num_pairs == 0:
        return _empty_fallback(frame_pair_topologies, roi_row, roi_col)

    
    
    
    
    
    cell_in: Dict[Tuple[int, int], float] = {}
    cell_out: Dict[Tuple[int, int], float] = {}

    for fp in frame_pair_topologies:
        M, N = fp.M, fp.N
        adj = fp.adj_matrix
        ft, ftp1 = int(fp.frame_t), int(fp.frame_tp1)

        
        out_deg = adj.sum(axis=1).astype(np.float64)
        for i, cid in enumerate(fp.cell_ids_t):
            cell_out[(ft, int(cid))] = float(out_deg[i])

        
        in_deg = adj.sum(axis=0).astype(np.float64)
        for j, cid in enumerate(fp.cell_ids_tp1):
            cell_in[(ftp1, int(cid))] = float(in_deg[j])

    
    
    
    
    frame_s_values: Dict[int, Dict[int, float]] = defaultdict(dict)

    for fp in frame_pair_topologies:
        ft, ftp1 = int(fp.frame_t), int(fp.frame_tp1)

        
        for cid in fp.cell_ids_t:
            cid_i = int(cid)
            in_d = cell_in.get((ft, cid_i), 0.0)
            out_d = cell_out.get((ft, cid_i), 0.0)
            if in_d == 0.0:
                in_d = 1.0  
            if out_d == 0.0:
                out_d = 1.0  
            s = np.log(1.0 + in_d) * np.log(1.0 + out_d)
            frame_s_values[ft][cid_i] = s

        
        for cid in fp.cell_ids_tp1:
            cid_i = int(cid)
            in_d = cell_in.get((ftp1, cid_i), 0.0)
            out_d = cell_out.get((ftp1, cid_i), 0.0)
            if in_d == 0.0:
                in_d = 1.0
            if out_d == 0.0:
                out_d = 1.0  
            s = np.log(1.0 + in_d) * np.log(1.0 + out_d)
            frame_s_values[ftp1][cid_i] = s

    
    cell_s_tilde: Dict[Tuple[int, int], float] = {}
    for f, cells in frame_s_values.items():
        max_s = max(cells.values()) if cells else 1.0
        if max_s > 0:
            for cid, s in cells.items():
                cell_s_tilde[(f, cid)] = s / max_s

    
    
    
    cell_influence_scores: Dict[int, float] = {}

    for fp in frame_pair_topologies:
        M, N = fp.M, fp.N
        ft, ftp1 = int(fp.frame_t), int(fp.frame_tp1)

        
        influence_t = np.array([
            cell_s_tilde.get((ft, int(cid)), 0.0)
            for cid in fp.cell_ids_t
        ], dtype=np.float64)

        
        influence_tp1 = np.array([
            cell_s_tilde.get((ftp1, int(cid)), 0.0)
            for cid in fp.cell_ids_tp1
        ], dtype=np.float64)

        fp.cell_influence = np.concatenate([influence_t, influence_tp1])

        
        fp.pair_weights = fp.adj_matrix.astype(np.float64)

        
        fp.cluster_influence = np.zeros((M, N), dtype=np.float64)
        if fp.n_clusters > 0:
            for cid_u in np.unique(fp.cluster_ids[fp.cluster_ids >= 0]):
                cm = fp.cluster_ids == cid_u
                rc = np.unique(np.where(cm)[0])
                cc = np.unique(np.where(cm)[1])
                mean_inf = np.concatenate([influence_t[rc], influence_tp1[cc]]).mean()
                fp.cluster_influence[cm] = mean_inf

        
        for i, cid in enumerate(fp.cell_ids_t):
            cid_i = int(cid)
            cell_influence_scores[cid_i] = max(
                cell_influence_scores.get(cid_i, 0.0), float(influence_t[i]),
            )
        for j, cid in enumerate(fp.cell_ids_tp1):
            cid_i = int(cid)
            cell_influence_scores[cid_i] = max(
                cell_influence_scores.get(cid_i, 0.0), float(influence_tp1[j]),
            )

    return SequenceTopology(
        sequence_name="", num_frames=0, tau=0.0,
        roi_row=roi_row, roi_col=roi_col,
        cell_influence_scores=cell_influence_scores,
        frame_pairs=frame_pair_topologies,
    )






def _compute_random_walk_influence(
    org_df_cells: pd.DataFrame,
    frame_pair_topologies: List[FramePairTopology],
    roi_row: float,
    roi_col: float,
    tau: Optional[float] = None,
    horizon: int = 6,
) -> SequenceTopology:
    
    frames = sorted(org_df_cells["frame_num"].unique())
    if len(frames) < 2:
        return _single_frame_fallback(org_df_cells, frame_pair_topologies, roi_row, roi_col)

    if tau is None:
        tau = _compute_tau_from_df(org_df_cells)

    node_list = []
    node_idx_map: Dict[Tuple[int, int], int] = {}
    for f in frames:
        fc = org_df_cells[org_df_cells["frame_num"] == f]
        for ri in fc.index:
            node_idx_map[(f, ri)] = len(node_list)
            node_list.append({"frame": f, "df_row": ri,
                              "cell_id": int(org_df_cells.loc[ri, "id"])})

    num_nodes = len(node_list)
    if num_nodes == 0:
        return _empty_fallback(frame_pair_topologies, roi_row, roi_col)

    transitions = {}
    for fi in range(len(frames) - 1):
        fc, fn = frames[fi], frames[fi + 1]
        cc = org_df_cells[org_df_cells["frame_num"] == fc]
        nc = org_df_cells[org_df_cells["frame_num"] == fn]
        if len(cc) == 0 or len(nc) == 0:
            transitions[fc] = None; continue
        ct = cc[["centroid_row", "centroid_col"]].values.astype(np.float64)
        nt = nc[["centroid_row", "centroid_col"]].values.astype(np.float64)
        rd = np.abs(ct[:, 0:1] - nt[np.newaxis, :, 0])
        cd = np.abs(ct[:, 1:2] - nt[np.newaxis, :, 1])
        dist = np.sqrt(rd ** 2 + cd ** 2)
        fm = (rd <= roi_row) & (cd <= roi_col)
        tw = np.exp(-dist / tau) * fm.astype(np.float64)
        ws = np.maximum(tw.sum(axis=1, keepdims=True), 1e-12)
        tp = tw / ws
        ni_arr = np.array([node_idx_map[(fn, ri)] for ri in nc.index])
        ci_arr = np.array([node_idx_map[(fc, ri)] for ri in cc.index])
        transitions[fc] = (tp, ci_arr, ni_arr)

    raw_inf = np.zeros(num_nodes, dtype=np.float64)
    last_frame = frames[-1]
    use_horizon = (horizon is not None and horizon > 0)

    for t_idx, t in enumerate(frames):
        target = frames[min(t_idx + horizon, len(frames) - 1)] if use_horizon else last_frame
        if target == t:
            for ni, n in enumerate(node_list):
                if n["frame"] == t:
                    raw_inf[ni] = max(raw_inf[ni], 1.0)
            continue
        R = np.zeros(num_nodes, dtype=np.float64)
        for ni, n in enumerate(node_list):
            if n["frame"] == target: R[ni] = 1.0
        ti = frames.index(target)
        for fi in range(ti - 1, t_idx - 1, -1):
            f = frames[fi]
            if transitions.get(f) is None: continue
            tp, ci_arr, ni_arr = transitions[f]
            R_curr = tp @ R[ni_arr]
            for i, nd in enumerate(ci_arr): R[nd] = R_curr[i]
        for ni, n in enumerate(node_list):
            if n["frame"] == t:
                raw_inf[ni] = max(raw_inf[ni], R[ni])

    cell_inf_scores: Dict[int, float] = {}
    for ni, n in enumerate(node_list):
        cid, s = n["cell_id"], float(raw_inf[ni])
        cell_inf_scores[cid] = max(cell_inf_scores.get(cid, 0.0), s)

    max_inf = max(cell_inf_scores.values()) if cell_inf_scores else 1.0
    if max_inf > 0:
        for cid in cell_inf_scores:
            cell_inf_scores[cid] /= max_inf

    _populate_frame_pair_influence_rw(frame_pair_topologies, cell_inf_scores)
    return SequenceTopology(
        sequence_name="", num_frames=len(frames), tau=tau,
        roi_row=roi_row, roi_col=roi_col,
        cell_influence_scores=cell_inf_scores,
        frame_pairs=frame_pair_topologies,
    )


def _populate_frame_pair_influence_rw(fps, scores):
    
    for fp in fps:
        M, N = fp.M, fp.N
        inf_t = np.array([scores.get(int(c), 0.0) for c in fp.cell_ids_t], dtype=np.float64)
        inf_tp1 = np.array([scores.get(int(c), 0.0) for c in fp.cell_ids_tp1], dtype=np.float64)
        fp.cell_influence = np.concatenate([inf_t, inf_tp1])
        fp.cluster_influence = np.zeros((M, N), dtype=np.float64)
        if fp.n_clusters > 0:
            for cid_u in np.unique(fp.cluster_ids[fp.cluster_ids >= 0]):
                cm = fp.cluster_ids == cid_u
                rc = np.unique(np.where(cm)[0])
                cc = np.unique(np.where(cm)[1])
                mean_inf = np.concatenate([inf_t[rc], inf_tp1[cc]]).mean()
                fp.cluster_influence[cm] = mean_inf






def _compute_tau_from_df(df: pd.DataFrame) -> float:
    distances = []
    for cid in df["id"].unique():
        cd = df[df["id"] == cid].sort_values("frame_num")
        if len(cd) < 2: continue
        rs, cs, fs = cd["centroid_row"].values, cd["centroid_col"].values, cd["frame_num"].values
        for i in range(len(fs) - 1):
            if fs[i + 1] == fs[i] + 1:
                distances.append(np.sqrt((rs[i+1]-rs[i])**2 + (cs[i+1]-cs[i])**2))
    return float(np.median(distances)) if distances else 1.0


def _single_frame_fallback(org_df, fps, rr, rc):
    scores = {int(r["id"]): 1.0 for _, r in org_df.iterrows()}
    for fp in fps:
        fp.cell_influence = np.ones(fp.M + fp.N)
        fp.cluster_influence = np.ones((fp.M, fp.N))
        fp.cluster_influence[~fp.adj_matrix] = 0.0
    return SequenceTopology(sequence_name="", num_frames=1, tau=1.0,
                            roi_row=rr, roi_col=rc, cell_influence_scores=scores, frame_pairs=fps)


def _empty_fallback(fps, rr, rc):
    for fp in fps:
        fp.cell_influence = np.zeros(fp.M + fp.N)
        fp.cluster_influence = np.zeros((fp.M, fp.N))
    return SequenceTopology(sequence_name="", num_frames=0, tau=1.0,
                            roi_row=rr, roi_col=rc, cell_influence_scores={}, frame_pairs=fps)
