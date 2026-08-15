

import os
import os.path as osp
import json
import torch
import pandas as pd
import numpy as np
import networkx as nx
from skimage import io
import warnings
warnings.filterwarnings("ignore")
import imageio


def resolve_association_threshold(
    path_inference_output,
    explicit_threshold=None,
    use_saved_validation_threshold=False,
):
    
    if use_saved_validation_threshold and explicit_threshold is not None:
        raise ValueError(
            "Use either an explicit threshold or the saved validation threshold, not both"
        )
    if use_saved_validation_threshold:
        metadata_path = os.path.join(
            path_inference_output, 'inference_metadata.json'
        )
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "Saved validation threshold requested but inference metadata is missing: "
                f"{metadata_path}"
            )
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get('schema_version') != 1:
            raise ValueError(
                f"Unsupported inference metadata schema in {metadata_path}: "
                f"{metadata.get('schema_version')!r}"
            )
        threshold = float(metadata['validation_probability_threshold'])
        policy = 'validation_calibrated'
    else:
        threshold = 0.5 if explicit_threshold is None else float(explicit_threshold)
        policy = 'fixed_explicit' if explicit_threshold is not None else 'fixed_0.5'
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"decision threshold must be in [0, 1], got {threshold}")
    return threshold, policy





class Postprocess(object):
    

    def __init__(self,
                 is_3d,
                 type_masks,
                 merge_operation,
                 decision_threshold,
                 path_inference_output,
                 center_coord,
                 path_seg_result,
                 directed=True,
                 
                 K_sister=1.7,
                 radius_type='equivalent_circle',
                 safety_margin=1.0,
                 fallback_radius_factor=3.0,
                 min_search_radius=5.0,
                 max_search_radius=200.0,
                 P_unassigned=400.0,
                 allow_singleton_parent=True,
                 pair_fallback_policy='independent',
                 output_dir=None,
                 one_to_one_association=True,
                 division_mode='geometric',
                 ):
        
        
        
        
        file1_new = os.path.join(path_inference_output, 'merged_edge_index.pt')
        file1_old = os.path.join(path_inference_output, 'pytorch_geometric_data.pt')
        file2 = os.path.join(path_inference_output, 'all_data_df.csv')
        file3 = os.path.join(path_inference_output, 'raw_output.pt')

        
        self.dir_result = dir_results = path_seg_result
        self.results = []
        if os.path.exists(dir_results):
            self.results = [os.path.join(dir_results, fname) for fname in sorted(os.listdir(dir_results))
                            if type_masks in fname]

        self.is_3d = is_3d
        self.center_coord = center_coord
        self.merge_operation = merge_operation
        self.decision_threshold = decision_threshold
        self.directed = directed
        self.path_inference_output = path_inference_output
        self.cols = ["child_id", "parent_id", "start_frame"]

        
        self.K_sister = K_sister
        self.radius_type = radius_type
        self.safety_margin = safety_margin
        self.fallback_radius_factor = fallback_radius_factor
        self.min_search_radius = min_search_radius
        self.max_search_radius = max_search_radius
        self.P_unassigned = P_unassigned
        self.allow_singleton_parent = allow_singleton_parent
        self.pair_fallback_policy = pair_fallback_policy
        self.output_dir = output_dir
        self.one_to_one_association = bool(one_to_one_association)
        if division_mode not in {'none', 'nearest', 'geometric'}:
            raise ValueError(
                "division_mode must be one of: none, nearest, geometric; "
                f"got {division_mode!r}"
            )
        self.division_mode = division_mode

        print(f"[Division postprocess] mode='{division_mode}', "
              f"one_to_one_association={self.one_to_one_association}")
        print(f"[Geometric parent search] K_sister={K_sister}, "
              f"radius_type='{radius_type}', safety_margin={safety_margin}, "
              f"fallback_radius_factor={fallback_radius_factor}, "
              f"min_search_radius={min_search_radius}, "
              f"max_search_radius={max_search_radius}, "
              f"P_unassigned={P_unassigned}, "
              f"allow_singleton_parent={allow_singleton_parent}, "
              f"pair_fallback_policy='{pair_fallback_policy}'")

        
        self.component_stats = {
            'total_frames_with_new_cells': 0,
            'total_components': 0,
            'total_singletons': 0,     
            'total_pairs': 0,           
            'total_complex': 0,         
            'max_component_size': 0,
            'complex_components': [],   
        }

        
        if os.path.exists(file1_new):
            
            print(f"Load merged edge index from: {file1_new}")
            self.edge_index = self._load_file(file1_new)
        elif os.path.exists(file1_old):
            
            print(f"Load edge index from PyG Data: {file1_old}")
            data = self._load_file(file1_old)
            self.edge_index = data.edge_index
        else:
            raise FileNotFoundError(
                f"Edge index file not found! Tried:\n"
                f"  - {file1_new} (new format)\n"
                f"  - {file1_old} (old format)"
            )

        self.df_preds = self._load_file(file2)     
        self.output_pred = self._load_file(file3)  

        
        self.find_connected_edges()

    def _one_to_one_connected_edges(self):
        
        hard_mask = self.outputs_hard.bool()
        candidate_edges = self.edge_index[:, hard_mask]
        if candidate_edges.numel() == 0:
            return candidate_edges

        probabilities = torch.sigmoid(self.output_pred[hard_mask]).detach().cpu().numpy()
        candidates = candidate_edges.detach().cpu().numpy()
        selected = []
        source_frames = self.df_preds.loc[candidates[0], "frame_num"].astype(int).values
        target_frames = self.df_preds.loc[candidates[1], "frame_num"].astype(int).values

        for frame in sorted(set(source_frames.tolist())):
            indices = np.where(
                (source_frames == frame) & (target_frames == frame + 1)
            )[0]
            graph = nx.Graph()
            for index in indices:
                source = int(candidates[0, index])
                target = int(candidates[1, index])
                left = ("source", source)
                right = ("target", target)
                weight = float(probabilities[index])
                if graph.has_edge(left, right):
                    graph[left][right]["weight"] = max(
                        graph[left][right]["weight"], weight
                    )
                else:
                    graph.add_edge(left, right, weight=weight)

            matching = nx.algorithms.matching.max_weight_matching(
                graph, maxcardinality=False, weight="weight"
            )
            for node_a, node_b in matching:
                if node_a[0] == "source":
                    source, target = node_a[1], node_b[1]
                else:
                    source, target = node_b[1], node_a[1]
                selected.append((source, target))

        if not selected:
            return torch.empty((2, 0), dtype=self.edge_index.dtype)
        selected.sort(
            key=lambda edge: (
                int(self.df_preds.loc[edge[0], "frame_num"]), edge[0], edge[1]
            )
        )
        return torch.tensor(selected, dtype=self.edge_index.dtype).t().contiguous()

    
    
    
    def _load_file(self, file_path):
        
        print(f"Load {file_path}")
        file_type = file_path.split('.')[-1]
        if file_type == 'csv':
            file = pd.read_csv(file_path, index_col=0)
        if file_type == 'pt':
            file = torch.load(file_path)
        return file

    def save_csv(self, df_file, file_name):
        
        full_name = os.path.join(self.path_inference_output, f"postprocess_data")
        os.makedirs(full_name, exist_ok=True)
        full_name = os.path.join(full_name, file_name)
        df_file.to_csv(full_name)

    def save_txt(self, str_txt, output_folder, file_name):
        
        full_name = os.path.join(output_folder, file_name)
        with open(full_name, "w") as text_file:
            text_file.write(str_txt)

    
    
    

    def insert_in_specific_col(self, all_frames_traject, frame_ind, curr_node, next_node):
        
        if curr_node in all_frames_traject[frame_ind, :]:
            
            flag = 0
            ind_place = np.argwhere(all_frames_traject[frame_ind, :] == curr_node)
            if frame_ind + 1 < all_frames_traject.shape[0]:
                all_frames_traject[frame_ind + 1, ind_place] = next_node
        else:
            
            flag = 1
            ind_place = np.argwhere(all_frames_traject[frame_ind, :] == -2)
            
            while ind_place.size == 0:
                new_col = -2 * np.ones((all_frames_traject.shape[0], 1),
                                       dtype=all_frames_traject.dtype)
                all_frames_traject = np.append(all_frames_traject, new_col, axis=1)
                ind_place = np.argwhere(all_frames_traject[frame_ind, :] == -2)
            ind_place = ind_place.min()
            all_frames_traject[frame_ind, ind_place] = curr_node
            if frame_ind + 1 < all_frames_traject.shape[0]:
                all_frames_traject[frame_ind + 1, ind_place] = next_node

        return flag, all_frames_traject

    def fill_first_frame(self, cell_starts):
        
        cols = ["child_id", "parent_id", "start_frame"]
        df_parent = pd.DataFrame(index=range(len(list(cell_starts))), columns=cols)
        df_parent.loc[:, ["start_frame", "parent_id"]] = 0
        df_parent.loc[:, "child_id"] = cell_starts
        return df_parent

    
    
    

    def _cell_radius(self, row):
        
        area = row.get("area", 0)
        major_axis = row.get("major_axis_length", 0)

        r_equiv = np.sqrt(area / np.pi) if area > 0 else 0.0
        r_major = major_axis / 2.0 if major_axis > 0 else 0.0

        if self.radius_type == 'equivalent_circle':
            return r_equiv
        elif self.radius_type == 'major_axis_half':
            return r_major
        elif self.radius_type == 'max':
            return max(r_equiv, r_major)
        else:
            return r_equiv

    def _cell_distance(self, df, idx_a, idx_b):
        
        if self.is_3d:
            a = df.loc[idx_a, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
            b = df.loc[idx_b, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
        else:
            a = df.loc[idx_a, ["centroid_row", "centroid_col"]].values.astype(float)
            b = df.loc[idx_b, ["centroid_row", "centroid_col"]].values.astype(float)
        return np.sqrt(((a - b) ** 2).sum())

    def _find_sister_candidates(self, df, cell_starts):
        
        n = len(cell_starts)
        sisters = {cell: [] for cell in cell_starts}

        if n < 2:
            return sisters

        
        radii = {cell: self._cell_radius(df.loc[cell]) for cell in cell_starts}

        
        for a_idx in range(n):
            cell_a = cell_starts[a_idx]
            for b_idx in range(a_idx + 1, n):
                cell_b = cell_starts[b_idx]
                d = self._cell_distance(df, cell_a, cell_b)
                threshold = self.K_sister * (radii[cell_a] + radii[cell_b])

                if d <= threshold:
                    sisters[cell_a].append(cell_b)
                    sisters[cell_b].append(cell_a)

        return sisters

    def _find_connected_components(self, sister_map, cell_starts):
        
        visited = set()
        components = []

        for cell in cell_starts:
            if cell in visited:
                continue

            
            comp = []
            queue = [cell]
            while queue:
                c = queue.pop(0)
                if c in visited:
                    continue
                visited.add(c)
                comp.append(c)
                for s in sister_map.get(c, []):
                    if s not in visited:
                        queue.append(s)

            components.append(comp)

        
        components.sort(key=len, reverse=True)
        return components

    def _compute_Rq(self, df, cell_i, cell_j):
        
        d = self._cell_distance(df, cell_i, cell_j)
        r_i = self._cell_radius(df.loc[cell_i])
        r_j = self._cell_radius(df.loc[cell_j])
        R_q = (d + r_i + r_j) / 2.0
        return R_q * self.safety_margin

    def _collect_parents_in_q(self, df, cell_i, sister_j, finish_coords, finish_node_ids):
        
        R_q = self._compute_Rq(df, cell_i, sister_j)

        if self.is_3d:
            coord_i = df.loc[cell_i, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
            coord_j = df.loc[sister_j, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
        else:
            coord_i = df.loc[cell_i, ["centroid_row", "centroid_col"]].values.astype(float)
            coord_j = df.loc[sister_j, ["centroid_row", "centroid_col"]].values.astype(float)

        q_center = (coord_i + coord_j) / 2.0
        distances_to_q = np.sqrt(((finish_coords - q_center) ** 2).sum(axis=-1))
        within_q = np.where(distances_to_q <= R_q)[0]

        return [(int(finish_node_ids[idx]), float(distances_to_q[idx]))
                for idx in within_q]

    def _compute_search_radius_from_sisters(self, df, cell, sister_candidates):
        
        return None, None  

    def _compute_search_radius_fallback(self, df, cell):
        
        r_i = self._cell_radius(df.loc[cell])
        radius = self.fallback_radius_factor * r_i
        radius = np.clip(radius, self.min_search_radius, self.max_search_radius)
        return radius, 'fallback'

    def _nearest_fallback_parent(self, df, cell, finish_coords, finish_node_ids, coord_cols):
        
        curr_coord = df.loc[cell, coord_cols].values.astype(float)
        search_radius, _ = self._compute_search_radius_fallback(df, cell)
        distances = np.sqrt(((finish_coords - curr_coord) ** 2).sum(axis=-1))
        within = distances <= search_radius
        if not within.any():
            return 0, None
        candidates = np.where(within)[0]
        nearest = candidates[np.argmin(distances[candidates])]
        return int(finish_node_ids[nearest]), float(distances[nearest])

    def _log_component_summary(self):
        
        stats = self.component_stats
        print(f"\n{'='*60}")
        print(f"[Component Summary]")
        print(f"  Frames with new cells: {stats['total_frames_with_new_cells']}")
        print(f"  Total components:      {stats['total_components']}")
        print(f"  Singletons (|C|=1):    {stats['total_singletons']}")
        print(f"  Pairs (|C|=2):         {stats['total_pairs']}")
        print(f"  Complex (|C|>=3):      {stats['total_complex']}  [ILP candidates]")
        print(f"  Max component size:    {stats['max_component_size']}")

        if stats['total_complex'] > 0:
            print("\n  COMPLEX COMPONENTS (would benefit from ILP):")
            for c in stats['complex_components']:
                print(f"    Frame {c['frame']}: |C|={c['size']}, "
                      f"cells={c['cells']}, "
                      f"finished_cells={c['n_finished']}, "
                      f"parent_candidates={c['n_parent_candidates']}")
            print(f"\n  ILP would jointly optimize {stats['total_complex']} "
                  f"non-trivial clusters")
        else:
            print(f"\n  All components are |C|<=2; greedy per-cell "
                  f"assignment is optimal, ILP not needed")

        print(f"{'='*60}\n")

    
    
    

    def _ilp_solve_component(self, df, component, sister_map,
                              finish_coords, finish_node_ids, coord_cols):
        
        
        
        pairs = []
        for i_idx, cell_i in enumerate(component):
            for cell_j in sister_map.get(cell_i, []):
                if cell_i < cell_j and cell_j in component:
                    pair = (cell_i, cell_j)
                    if pair not in pairs:
                        pairs.append(pair)

        
        
        
        pair_mothers = {}
        pair_cost = {}
        for pair in pairs:
            i, j = pair
            R_q = self._compute_Rq(df, i, j)

            
            if self.is_3d:
                ci = df.loc[i, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
                cj = df.loc[j, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
            else:
                ci = df.loc[i, ["centroid_row", "centroid_col"]].values.astype(float)
                cj = df.loc[j, ["centroid_row", "centroid_col"]].values.astype(float)
            q_center = (ci + cj) / 2.0

            
            mothers_in_q = []
            for l_idx, l in enumerate(finish_node_ids):
                l_coord = df.loc[l, coord_cols].values.astype(float)
                dist_to_q = np.sqrt(((l_coord - q_center) ** 2).sum())
                if dist_to_q <= R_q:
                    mothers_in_q.append(l)
                    pair_cost[(pair, l)] = dist_to_q

            pair_mothers[pair] = mothers_in_q

        
        all_mothers = list(set(
            l for mothers in pair_mothers.values() for l in mothers
        ))

        
        

        def generate_partitions(remaining):
            
            if not remaining:
                return [[]]  

            results = []
            first = remaining[0]
            rest = remaining[1:]

            
            for sub in generate_partitions(rest):
                results.append([('unassigned', first)] + sub)

            
            for sister in sister_map.get(first, []):
                if sister in rest:
                    new_remaining = [c for c in rest if c != sister]
                    for sub in generate_partitions(new_remaining):
                        results.append([('pair', (first, sister))] + sub)

            return results

        partitions = generate_partitions(component)

        
        best_total_cost = float('inf')
        best_assignment = None

        for partition in partitions:
            assigned_pairs = [p for t, p in partition if t == 'pair']
            unassigned_cells = [p for t, p in partition if t == 'unassigned']

            n_pairs = len(assigned_pairs)
            n_mothers = len(all_mothers)

            if n_pairs == 0:
                
                total_cost = self.P_unassigned * len(component)
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_assignment = {cell: 0 for cell in component}
                continue

            
            
            

            from itertools import permutations, combinations

            for mother_subset in combinations(range(n_mothers), min(n_pairs, n_mothers)):
                mother_indices = list(mother_subset)  
                if len(mother_indices) < n_pairs:
                    
                    
                    mother_slots = mother_indices + [None] * (n_pairs - len(mother_indices))
                else:
                    mother_slots = mother_indices

                for perm in set(permutations(mother_slots, n_pairs)):
                    
                    total_cost = 0.0
                    valid = True

                    for idx, pair in enumerate(assigned_pairs):
                        mother_slot = perm[idx]
                        if mother_slot is None:
                            
                            total_cost += self.P_unassigned * 2
                        else:
                            l = all_mothers[mother_slot]
                            cost = pair_cost.get((pair, l), float('inf'))
                            if cost == float('inf'):
                                valid = False
                                break
                            total_cost += cost

                    if not valid:
                        continue

                    
                    total_cost += self.P_unassigned * len(unassigned_cells)

                    if total_cost < best_total_cost:
                        best_total_cost = total_cost
                        best_assignment = {}
                        for cell in component:
                            best_assignment[cell] = 0  

                        for idx, pair in enumerate(assigned_pairs):
                            mother_slot = perm[idx]
                            if mother_slot is not None:
                                l = all_mothers[mother_slot]
                                best_assignment[pair[0]] = l
                                best_assignment[pair[1]] = l

        
        if best_assignment is not None:
            assigned_count = sum(1 for v in best_assignment.values() if v != 0)
            unassigned_count = len(component) - assigned_count
            print(f"      [ILP] |C|={len(component)} pairs={len(pairs)} "
                  f"mothers={len(all_mothers)} "
                  f"assigned={assigned_count} unassigned={unassigned_count} "
                  f"cost={best_total_cost:.1f}")

        return best_assignment if best_assignment is not None else {c: 0 for c in component}

    def find_parent_cell(self, frame_ind, all_frames_traject, df, num_starts, cell_starts):
        
        if self.division_mode == 'none':
            df_parent = pd.DataFrame(index=range(len(cell_starts)), columns=self.cols)
            df_parent.loc[:, "start_frame"] = frame_ind
            df_parent.loc[:, "child_id"] = cell_starts
            df_parent.loc[:, "parent_id"] = 0
            return df_parent

        if self.division_mode == 'nearest':
            return self._find_parent_cell_nearest(
                frame_ind, all_frames_traject, df, cell_starts
            )

        
        ind_place = np.argwhere(all_frames_traject[frame_ind, :] == -1)
        finish_node_ids = all_frames_traject[frame_ind - 1, ind_place].squeeze(axis=1)

        df_parent = pd.DataFrame(index=range(len(cell_starts)), columns=self.cols)
        df_parent.loc[:, "start_frame"] = frame_ind

        
        if finish_node_ids.shape[0] == 0:
            df_parent.loc[:, "child_id"] = cell_starts
            df_parent.loc[:, "parent_id"] = 0
            return df_parent

        
        if self.is_3d:
            finish_coords = df.loc[finish_node_ids,
                                   ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(float)
            coord_cols = ["centroid_depth", "centroid_row", "centroid_col"]
        else:
            finish_coords = df.loc[finish_node_ids,
                                   ["centroid_row", "centroid_col"]].values.astype(float)
            coord_cols = ["centroid_row", "centroid_col"]

        
        sister_map = self._find_sister_candidates(df, cell_starts)
        sisters_found = sum(1 for v in sister_map.values() if len(v) > 0)

        
        components = self._find_connected_components(sister_map, cell_starts)
        comp_sizes = [len(c) for c in components]
        n_singletons = sum(1 for s in comp_sizes if s == 1)
        n_pairs = sum(1 for s in comp_sizes if s == 2)
        n_complex = sum(1 for s in comp_sizes if s >= 3)

        
        self.component_stats['total_frames_with_new_cells'] += 1
        self.component_stats['total_components'] += len(components)
        self.component_stats['total_singletons'] += n_singletons
        self.component_stats['total_pairs'] += n_pairs
        self.component_stats['total_complex'] += n_complex
        self.component_stats['max_component_size'] = max(
            self.component_stats['max_component_size'], max(comp_sizes) if comp_sizes else 0)

        
        msg = (f"  Frame {frame_ind}: {num_starts} new cells, "
               f"{sisters_found} have sisters, "
               f"{len(finish_node_ids)} finished, "
               f"components: {len(components)} ({comp_sizes})")

        
        if n_complex > 0:
            
            complex_details = []
            for comp in components:
                if len(comp) >= 3:
                    
                    comp_parents = set()
                    for cell_i in comp:
                        for sister_j in sister_map.get(cell_i, []):
                            parents = self._collect_parents_in_q(
                                df, cell_i, sister_j, finish_coords, finish_node_ids)
                            for p_id, _ in parents:
                                comp_parents.add(p_id)
                    complex_details.append(
                        f"|C|={len(comp)} cells={comp} parents_in_qs={len(comp_parents)}")
                    
                    self.component_stats['complex_components'].append({
                        'frame': frame_ind,
                        'size': len(comp),
                        'cells': comp,
                        'n_finished': len(finish_node_ids),
                        'n_parent_candidates': len(comp_parents),
                    })
            msg += f"\n    COMPLEX: {' | '.join(complex_details)}"

        print(msg)

        
        for comp in components:
            if len(comp) >= 3:
                
                ilp_result = self._ilp_solve_component(
                    df, comp, sister_map, finish_coords, finish_node_ids, coord_cols)

                for cell in comp:
                    parent_id = ilp_result.get(cell, 0)
                    row_idx = cell_starts.index(cell)
                    df_parent.loc[row_idx, "child_id"] = cell
                    df_parent.loc[row_idx, "parent_id"] = parent_id

                    
                    if parent_id == 0 and self.allow_singleton_parent:
                        fallback_parent, _ = self._nearest_fallback_parent(
                            df, cell, finish_coords, finish_node_ids, coord_cols)
                        if fallback_parent != 0:
                            df_parent.loc[row_idx, "parent_id"] = fallback_parent
                            print(f"      [ILP-fallback] cell={cell} assigned via stage2")
            else:
                
                if len(comp) == 2:
                    
                    
                    
                    a, b = comp[0], comp[1]
                    row_a = cell_starts.index(a)
                    row_b = cell_starts.index(b)

                    
                    parents_in_q = {}
                    if b in sister_map.get(a, []):
                        R_q = self._compute_Rq(df, a, b)
                        coord_a = df.loc[a, coord_cols].values.astype(float)
                        coord_b = df.loc[b, coord_cols].values.astype(float)
                        q_center = (coord_a + coord_b) / 2.0

                        distances_to_q = np.sqrt(
                            ((finish_coords - q_center) ** 2).sum(axis=-1))
                        within = distances_to_q <= R_q
                        for idx in np.where(within)[0]:
                            parents_in_q[int(finish_node_ids[idx])] = float(distances_to_q[idx])

                    if len(parents_in_q) > 0:
                        
                        best_parent = min(parents_in_q, key=parents_in_q.get)
                        best_dist = parents_in_q[best_parent]
                        df_parent.loc[row_a, "child_id"] = a
                        df_parent.loc[row_a, "parent_id"] = best_parent
                        df_parent.loc[row_b, "child_id"] = b
                        df_parent.loc[row_b, "parent_id"] = best_parent
                        print(f"    [pair] cells=({a},{b}) "
                              f"R_q={R_q:.1f} mothers_in_q={len(parents_in_q)} "
                              f"-> parent={best_parent} (dist_to_midpoint={best_dist:.1f})")
                    else:
                        
                        df_parent.loc[row_a, "child_id"] = a
                        df_parent.loc[row_a, "parent_id"] = 0
                        df_parent.loc[row_b, "child_id"] = b
                        df_parent.loc[row_b, "parent_id"] = 0
                        if self.pair_fallback_policy == 'independent':
                            for cell in comp:
                                row_idx = cell_starts.index(cell)
                                fallback_parent, _ = self._nearest_fallback_parent(
                                    df, cell, finish_coords, finish_node_ids, coord_cols)
                                if fallback_parent != 0:
                                    df_parent.loc[row_idx, "parent_id"] = fallback_parent
                                    print(f"    [pair-fallback] cell={cell} assigned via stage2")
                        elif self.pair_fallback_policy == 'same_parent':
                            parent_a, dist_a = self._nearest_fallback_parent(
                                df, a, finish_coords, finish_node_ids, coord_cols)
                            parent_b, dist_b = self._nearest_fallback_parent(
                                df, b, finish_coords, finish_node_ids, coord_cols)
                            if parent_a != 0 and parent_a == parent_b:
                                df_parent.loc[row_a, "parent_id"] = parent_a
                                df_parent.loc[row_b, "parent_id"] = parent_a
                                print(f"    [pair-fallback-same] cells=({a},{b}) "
                                      f"-> parent={parent_a} "
                                      f"(dist={dist_a:.1f},{dist_b:.1f})")
                        elif self.pair_fallback_policy != 'none':
                            raise ValueError(
                                f"Unsupported pair_fallback_policy: {self.pair_fallback_policy}")
                else:
                    
                    cell = comp[0]
                    row_idx = cell_starts.index(cell)
                    df_parent.loc[row_idx, "child_id"] = cell
                    df_parent.loc[row_idx, "parent_id"] = 0
                    if self.allow_singleton_parent:
                        fallback_parent, _ = self._nearest_fallback_parent(
                            df, cell, finish_coords, finish_node_ids, coord_cols)
                        if fallback_parent != 0:
                            df_parent.loc[row_idx, "child_id"] = cell
                            df_parent.loc[row_idx, "parent_id"] = fallback_parent

        return df_parent

    def _find_parent_cell_nearest(self, frame_ind, all_frames_traject, df, cell_starts):
        
        ended_columns = np.argwhere(all_frames_traject[frame_ind, :] == -1)
        finished = all_frames_traject[frame_ind - 1, ended_columns].squeeze(axis=1)
        parent_df = pd.DataFrame(index=range(len(cell_starts)), columns=self.cols)
        parent_df.loc[:, "start_frame"] = frame_ind
        parent_df.loc[:, "child_id"] = cell_starts
        parent_df.loc[:, "parent_id"] = 0
        if finished.shape[0] == 0:
            return parent_df

        coord_cols = (
            ["centroid_depth", "centroid_row", "centroid_col"]
            if self.is_3d
            else ["centroid_row", "centroid_col"]
        )
        finished_coords = df.loc[finished, coord_cols].values.astype(float)
        for row_index, cell in enumerate(cell_starts):
            cell_coord = df.loc[cell, coord_cols].values.astype(float)
            squared_distance = ((finished_coords - cell_coord) ** 2).sum(axis=-1)
            nearest = int(np.argmin(squared_distance))
            parent_df.loc[row_index, "parent_id"] = int(finished[nearest])
        return parent_df

    def clean_repetition(self, df):
        
        all_childs = df.child_id.values
        unique_vals, count_vals = np.unique(all_childs, return_counts=True)
        prob_vals = unique_vals[count_vals > 1]  
        for prob_val in prob_vals:
            masking = df.child_id.values == prob_val
            all_apearence = df.loc[masking, :]
            start_frame = all_apearence.start_frame.min()
            end_frame = all_apearence.end_frame.max()
            df.loc[all_apearence.index[0], ["start_frame", "end_frame"]] = start_frame, end_frame
            df = df.drop(all_apearence.index[1:])

        return df.reset_index(drop=True)

    def set_all_info(self, df_parents_all, all_frames_traject):
        
        iterate_childs = df_parents_all.child_id.values
        frames_traject_same_label = all_frames_traject.copy()

        for ind, child_ind in enumerate(iterate_childs):
            
            coordinates_child = np.argwhere(all_frames_traject == child_ind)
            n_places = coordinates_child.shape[0]
            assert n_places == 1, f"Problem! find {n_places} places which the current child appears"

            coordinates_child = coordinates_child.squeeze()
            row, col = coordinates_child
            s_frame = df_parents_all.loc[ind, "start_frame"]
            assert row == s_frame, f"Problem! start frame {s_frame} is not equal to row {row}"

            
            curr_col = all_frames_traject[row:, col]
            last_ind = np.argwhere(curr_col == -1)
            if last_ind.size != 0:
                last_ind = last_ind[0].squeeze()
                curr_col = curr_col[:last_ind]
            e_frame = row + curr_col.shape[0] - 1

            df_parents_all.loc[ind, "end_frame"] = int(e_frame)
            curr_id = curr_col[-1]  
            df_parents_all.loc[ind, "child_id"] = curr_id
            
            frames_traject_same_label[row:e_frame + 1, col] = curr_id

        assert not (df_parents_all.isnull().values.any()), "Problem! dataframe contains NaN values"
        df_parents_all = self.clean_repetition(df_parents_all.astype(int))
        return df_parents_all.astype(int), frames_traject_same_label

    def df2str(self, df_track):
        
        str_track = ''
        for i in df_track.index:
            L = df_track.loc[i, "child_id"]
            B = df_track.loc[i, "start_frame"]
            E = df_track.loc[i, "end_frame"]
            P = df_track.loc[i, "parent_id"]
            str_track += f"{L} {B} {E} {P}\n"

        return str_track

    
    
    

    def merge_edges(self):
        
        in_output_pred, out_output_pred = self.match_edges()
        if self.merge_operation == 'OR' or self.merge_operation == 'AND':
            in_outputs_soft = torch.sigmoid(in_output_pred)
            in_outputs_hard = (in_outputs_soft > self.decision_threshold).int()
            out_outputs_soft = torch.sigmoid(out_output_pred)
            out_outputs_hard = (out_outputs_soft > self.decision_threshold).int()
            final_outputs_hard = np.bitwise_or(in_outputs_hard, out_outputs_hard) \
                if self.merge_operation == 'OR' \
                else np.bitwise_and(in_outputs_hard, out_outputs_hard)

        if self.merge_operation == 'AVG':
            avg_outputs_soft = torch.sigmoid(in_output_pred) + torch.sigmoid(out_output_pred)
            avg_outputs_soft = avg_outputs_soft / 2.0
            final_outputs_hard = (avg_outputs_soft > self.decision_threshold).int()

        self.outputs_hard = final_outputs_hard
        return final_outputs_hard

    def megre_match_edges(self, edge_index, output_pred):
        
        
        assert torch.all(edge_index[:, ::2] == edge_index[[1, 0], 1::2]), \
            "The results don't match!"

        
        edge_index = edge_index[:, ::2]
        in_output_pred = output_pred[::2]    
        out_output_pred = output_pred[1::2]  

        if self.merge_operation == 'OR' or self.merge_operation == 'AND':
            in_outputs_soft = torch.sigmoid(in_output_pred)
            in_outputs_hard = (in_outputs_soft > self.decision_threshold).int()
            out_outputs_soft = torch.sigmoid(out_output_pred)
            out_outputs_hard = (out_outputs_soft > self.decision_threshold).int()
            final_outputs_hard = np.bitwise_or(in_outputs_hard, out_outputs_hard) \
                if self.merge_operation == 'OR' \
                else np.bitwise_and(in_outputs_hard, out_outputs_hard)

        elif self.merge_operation == 'AVG':
            avg_outputs_soft = torch.sigmoid(in_output_pred) + torch.sigmoid(out_output_pred)
            avg_outputs_soft = avg_outputs_soft / 2.0
            final_outputs_hard = (avg_outputs_soft > self.decision_threshold).int()

        return final_outputs_hard, edge_index

    
    
    
    def find_connected_edges(self):
        
        edge_index, df, outputs = self.edge_index, self.df_preds, self.output_pred

        if not self.directed:
            
            final_outputs_hard, edge_index = self.megre_match_edges(
                edge_index.detach().clone(), outputs.detach().clone())
            self.outputs_hard = final_outputs_hard
            self.edge_index = edge_index
        else:
            
            outputs_soft = torch.sigmoid(outputs)
            self.outputs_hard = (outputs_soft > self.decision_threshold).int()

    
    
    
    def create_trajectory(self):
        
        edge_index, df, outputs_hard = self.edge_index, self.df_preds, self.outputs_hard
        self.flag_id0_terminate = False

        
        
        if self.one_to_one_association:
            connected_indices = self._one_to_one_connected_edges()
            print(
                "One-to-one association: "
                f"{int(outputs_hard.sum())} thresholded edges -> "
                f"{connected_indices.shape[1]} matched edges"
            )
        else:
            connected_indices = edge_index[:, outputs_hard.bool()]

        
        frame_nums = np.unique(df.frame_num)
        max_elements = [df.frame_num.isin([i]).sum() for i in frame_nums]
        all_frames_traject = np.zeros((frame_nums.shape[0], max(max_elements)))

        
        all_frames_traject[:, :] = -2
        all_trajectory_dict = {}
        str_track = ''
        df_parents = []

        
        for frame_ind in frame_nums:
            
            mask_frame_ind = df.frame_num.isin([frame_ind])
            nodes = df.loc[mask_frame_ind, :]
            nodes_indices = nodes.index.values

            next_frame_indices = np.array([])

            
            if frame_ind == 0:
                all_frames_traject[frame_ind, :nodes_indices.shape[0]] = nodes_indices
                df_parents.append(self.fill_first_frame(nodes_indices))

            num_starts = 0
            cell_starts = []

            for i in nodes_indices:
                if i in connected_indices[0, :]:
                    
                    ind_place = np.argwhere(connected_indices[0, :] == i)

                    if ind_place.shape[-1] > 1:
                        
                        next_frame_ind = connected_indices[1, ind_place].numpy().squeeze()
                        if self.is_3d:
                            next_frame = df.loc[next_frame_ind,
                                               ["centroid_depth", "centroid_row", "centroid_col"]].values
                            curr_node = df.loc[i,
                                              ["centroid_depth", "centroid_row", "centroid_col"]].values
                        else:
                            next_frame = df.loc[next_frame_ind,
                                               ["centroid_row", "centroid_col"]].values
                            curr_node = df.loc[i,
                                              ["centroid_row", "centroid_col"]].values

                        distance = ((next_frame - curr_node) ** 2).sum(axis=-1)
                        nearest_cell = np.argmin(distance, axis=-1)
                        next_node_ind = next_frame_ind[nearest_cell]

                    elif ind_place.shape[-1] == 1:
                        
                        next_node_ind = connected_indices[1, ind_place[0]]
                    else:
                        
                        next_node_ind = -1

                else:
                    
                    if i == 0:
                        self.flag_id0_terminate = True
                    next_node_ind = -1

                next_frame_indices = np.append(next_frame_indices, next_node_ind)

                
                start, all_frames_traject = self.insert_in_specific_col(
                    all_frames_traject, frame_ind, i, next_node_ind)
                num_starts += start

                if start == 1:
                    cell_starts.append(i)

            
            if num_starts > 0:
                df_parents.append(self.find_parent_cell(
                    frame_ind, all_frames_traject, df, num_starts, cell_starts))

            all_trajectory_dict[frame_ind] = {'from': nodes_indices, 'to': next_frame_indices}

        all_frames_traject = all_frames_traject.astype(int)

        
        df_parents_all = pd.concat(df_parents, axis=0).reset_index(drop=True)
        df_track_res, trajectory_same_label = self.set_all_info(
            df_parents_all, all_frames_traject)

        
        str_track = self.df2str(df_track_res)

        df_trajectory = pd.DataFrame(all_frames_traject)

        
        self.all_frames_traject = all_frames_traject
        self.trajectory_same_label = trajectory_same_label
        self.df_trajectory = df_trajectory
        self.df_track = df_track_res
        self.file_str = str_track

        return all_frames_traject, trajectory_same_label, df_trajectory, str_track

    
    
    

    def get_pred(self, idx):
        
        pred = None
        if len(self.results):
            im_path = self.results[idx]
            pred = io.imread(im_path)
            if self.is_3d and len(pred.shape) != 3:
                pred = np.stack(imageio.mimread(im_path))
                assert len(pred.shape) == 3, f"Expected 3d dimiension! but {pred.shape}"
        return pred

    def create_save_dir(self):
        
        if self.output_dir is not None:
            save_tra_dir = osp.abspath(self.output_dir)
        else:
            num_seq = self.dir_result.split('/')[-1][:2]
            save_tra_dir = osp.join(self.dir_result, f"../{num_seq}_RES")
        self.save_tra_dir = save_tra_dir
        os.makedirs(self.save_tra_dir, exist_ok=True)

    def save_new_pred(self, new_pred, idx):
        
        idx_str = "%03d" % idx
        file_name = f"mask{idx_str}.tif"
        full_dir = osp.join(self.save_tra_dir, file_name)
        io.imsave(full_dir, new_pred.astype(np.uint16))

    def save_trailing_empty_frames(self, first_missing_frame):
        
        if first_missing_frame >= len(self.results):
            return 0

        saved = 0
        for idx in range(first_missing_frame, len(self.results)):
            pred = self.get_pred(idx)
            if pred is None:
                raise RuntimeError(
                    f"Unable to read source segmentation frame {idx}: "
                    f"{self.results[idx]}"
                )
            if np.any(pred):
                raise RuntimeError(
                    "Tracking output ended before a non-empty segmentation frame: "
                    f"frame={idx}, path={self.results[idx]}. Refusing to replace "
                    "real detections with an empty CTC mask."
                )
            self.save_new_pred(np.zeros_like(pred), idx)
            saved += 1
        return saved

    def check_ids_consistent(self, frame_ind, pred_ids, curr_ids):
        
        predID_not_in_currID = [x for x in pred_ids if x not in curr_ids]
        currID_not_in_predID = [x for x in curr_ids if x not in pred_ids]
        flag1 = len(predID_not_in_currID) == 1 and predID_not_in_currID[0] == 0
        flag2 = len(currID_not_in_predID) == 0

        if not flag1:
            str_print = f"Frame {frame_ind}: Find segmented cell {predID_not_in_currID} without assigned labels"
            warnings.warn(str_print)

        if not flag2:
            print(f"WARNING Frame {frame_ind}: Find assigned labels {currID_not_in_predID} "
                  f"which are not appears in the final saved results")

        return flag1, predID_not_in_currID

    def fix_inconsistent(self, pred_prob_ids, pred):
        
        for id in pred_prob_ids:
            if id == 0:
                continue
            pred[pred == id] = 0
        return pred

    def fill_mask_labels(self, debug=False):
        
        self.create_save_dir()
        all_frames_traject, trajectory_same_label = \
            self.all_frames_traject, self.trajectory_same_label
        df = self.df_preds
        n_rows, n_cols = all_frames_traject.shape

        if n_rows > len(self.results):
            raise RuntimeError(
                "Tracking output contains more frames than the source segmentation: "
                f"tracking={n_rows}, segmentation={len(self.results)}"
            )

        count_diff_vals = 0

        for idx in range(n_rows):
            pred = self.get_pred(idx)           
            pred_copy = pred.copy()
            curr_row = all_frames_traject[idx, :]
            mask_id = np.bitwise_and(curr_row != -1, curr_row != -2)
            graph_ids = curr_row[mask_id]       
            graph_true_ids = trajectory_same_label[idx, mask_id]  
            mask_where = np.ones_like(pred)     
            frame_ids = []

            for id, true_id in zip(graph_ids, graph_true_ids):
                
                flag_id0 = true_id == 0
                if flag_id0:
                    if self.flag_id0_terminate:
                        new_id = trajectory_same_label.max() + 1
                        self.df_track.child_id[self.df_track.child_id == 0] = new_id
                        self.file_str = self.df2str(self.df_track)
                    else:
                        assert False, "Problem!"

                if self.is_3d:
                    
                    cell_center = df.loc[id, ["centroid_depth", "centroid_row",
                                              "centroid_col"]].values.astype(int)
                    depth_center, row_center, col_center = cell_center[0], cell_center[1], cell_center[2]
                    if self.center_coord:
                        n_depth_img, n_row_img, n_col_img = pred.shape
                        depth_center += n_depth_img // 2
                        row_center += n_row_img // 2
                        col_center += n_col_img // 2

                    
                    val = pred[depth_center, row_center, col_center]
                    if 'seg_label' in df.columns:
                        v_old = val
                        val = df.loc[id, "seg_label"]  
                        count_diff_vals += 1 if v_old != val else 0

                    
                    if val == 0:
                        if np.any(pred[depth_center - 3:depth_center + 3,
                                       row_center - 3:row_center + 3,
                                       col_center - 3:col_center + 3] != 0):
                            area = pred[depth_center - 3:depth_center + 3,
                                        row_center - 3:row_center + 3,
                                        col_center - 3:col_center + 3]
                            unique_labels, counts = np.unique(area, return_counts=True)
                            mask = unique_labels != 0
                            unique_labels = unique_labels[mask]
                            counts = counts[mask]
                            val = unique_labels[np.argmax(counts)]  
                        else:
                            print("Problem! center zero")
                            print(df.loc[id, ...].astype(int))
                            print()
                            continue
                else:
                    
                    cell_center = df.loc[id, ["centroid_row", "centroid_col"]].values.astype(int)
                    row_center, col_center = cell_center[0], cell_center[1]
                    if self.center_coord:
                        n_row_img, n_col_img = pred.shape
                        row_center += n_row_img // 2
                        col_center += n_col_img // 2

                    val = pred[row_center, col_center]
                    if 'seg_label' in df.columns:
                        v_old = val
                        val = df.loc[id, "seg_label"]
                        count_diff_vals += 1 if v_old != val else 0

                    
                    if val == 0:
                        if np.any(pred[row_center - 3:row_center + 3,
                                       col_center - 3:col_center + 3] != 0):
                            area = pred[row_center - 3:row_center + 3,
                                        col_center - 3:col_center + 3]
                            unique_labels, counts = np.unique(area, return_counts=True)
                            mask = unique_labels != 0
                            unique_labels = unique_labels[mask]
                            counts = counts[mask]
                            val = unique_labels[np.argmax(counts)]
                        else:
                            print("Problem! center zero")
                            print(df.loc[id, ...].astype(int))
                            print()
                            continue

                assert val != 0, "Problem! center zero"

                if flag_id0:
                    true_id = new_id

                
                mask_val = (pred_copy == val).copy()
                mask_curr = np.logical_and(mask_val, mask_where)
                pred_copy[mask_curr] = true_id
                mask_where = np.logical_and(np.logical_not(mask_val), mask_where)

                frame_ids.append(true_id)

            
            isOK, predID_not_in_currID = self.check_ids_consistent(
                idx, np.unique(pred_copy), frame_ids)
            if not debug:
                if not isOK:
                    pred_copy = self.fix_inconsistent(predID_not_in_currID, pred_copy)
                self.save_new_pred(pred_copy, idx)

        num_empty_frames = 0
        if not debug:
            num_empty_frames = self.save_trailing_empty_frames(n_rows)

        print(f"Number of different vals: {count_diff_vals}")
        if num_empty_frames:
            print(
                "Saved trailing empty CTC masks: "
                f"{num_empty_frames} (frames {n_rows}-{len(self.results) - 1})"
            )
        self.save_txt(self.file_str, self.save_tra_dir, 'res_track.txt')





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-modality', type=str, required=True, help='2D/3D modality')
    
    parser.add_argument('-iseg', type=str, required=True, help='segmentation output directory')
    
    parser.add_argument('-oi', type=str, required=True, help='inference output directory')
    parser.add_argument('-output_dir', type=str, default=None,
                        help='explicit CTC RES output directory')
    parser.add_argument('-decision_threshold', type=float, default=None,
                        help='explicit edge probability threshold; default 0.5')
    parser.add_argument('-use_saved_validation_threshold', action='store_true',
                        help='read validation_probability_threshold from inference_metadata.json')
    parser.add_argument('-disable_one_to_one_association', action='store_false',
                        dest='one_to_one_association', default=True,
                        help='disable framewise one-to-one association matching')
    parser.add_argument('-division_mode', type=str, default='geometric',
                        choices=['none', 'nearest', 'geometric'],
                        help='parent recovery: none, original nearest centroid, or geometric')
    
    parser.add_argument('-K_sister', type=float, default=1.7,
                        help='sister cell proximity constraint (default 1.5)')
    parser.add_argument('-radius_type', type=str, default='equivalent_circle',
                        choices=['equivalent_circle', 'major_axis_half', 'max'],
                        help='cell radius calculation method (default equivalent_circle)')
    parser.add_argument('-safety_margin', type=float, default=1.0,
                        help='R_q safety margin factor (default 1.0)')
    parser.add_argument('-fallback_radius_factor', type=float, default=3.0,
                        help='search radius factor for single-cell fallback (default 3.0)')
    parser.add_argument('-min_search_radius', type=float, default=5.0,
                        help='minimum search radius in pixels (default 5.0)')
    parser.add_argument('-max_search_radius', type=float, default=200.0,
                        help='maximum search radius in pixels (default 200.0)')
    parser.add_argument('-P_unassigned', type=float, default=400.0,
                        help='ILP penalty for unassigned cells (default 400.0)')
    parser.add_argument('-allow_singleton_parent', action='store_true', default=True,
                        help='allow singleton new tracks to attach to a finished parent')
    parser.add_argument('-disable_singleton_parent', action='store_false',
                        dest='allow_singleton_parent',
                        help='disable singleton-to-parent fallback for conservative division postprocessing')
    parser.add_argument('-pair_fallback_policy', type=str, default='independent',
                        choices=['same_parent', 'independent', 'none'],
                        help='fallback strategy when a sister pair has no parent in q_ab')

    args = parser.parse_args()

    modality = args.modality
    assert modality == '2D' or modality == '3D'

    path_inference_output = args.oi
    path_Seg_result = args.iseg

    decision_threshold, threshold_policy = resolve_association_threshold(
        path_inference_output,
        explicit_threshold=args.decision_threshold,
        use_saved_validation_threshold=args.use_saved_validation_threshold,
    )
    print(
        f"[Association threshold] policy={threshold_policy}, "
        f"probability={decision_threshold:.8g}"
    )

    is_3d = '3d' in modality.lower()
    directed = True                      
    merge_operation = 'AND'              

    
    pp = Postprocess(is_3d=is_3d,
                     type_masks='tif', merge_operation=merge_operation,
                     decision_threshold=decision_threshold,
                     path_inference_output=path_inference_output, center_coord=False,
                     directed=directed,
                     path_seg_result=path_Seg_result,
                     K_sister=args.K_sister,
                     radius_type=args.radius_type,
                     safety_margin=args.safety_margin,
                     fallback_radius_factor=args.fallback_radius_factor,
                     min_search_radius=args.min_search_radius,
                     max_search_radius=args.max_search_radius,
                     P_unassigned=args.P_unassigned,
                     allow_singleton_parent=args.allow_singleton_parent,
                     pair_fallback_policy=args.pair_fallback_policy,
                     output_dir=args.output_dir,
                     one_to_one_association=args.one_to_one_association,
                     division_mode=args.division_mode)

    
    all_frames_traject, trajectory_same_label, df_trajectory, str_track = pp.create_trajectory()

    
    pp._log_component_summary()

    
    pp.fill_mask_labels(debug=False)
