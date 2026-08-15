

import os
import os.path as osp
from collections.abc import Iterable

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data, InMemoryDataset
from hydra.utils import get_original_cwd
import warnings

from src.datamodules.datasets.motion_features import relative_motion_features





class CellTrackDataset(InMemoryDataset):
    

    def __init__(self,
                 num_frames,
                 type_file,
                 dirs_path,
                 main_path,
                 edge_feat_embed_dict,
                 normalize_all_cols,
                 mul_vals=[2, 2, 2],
                 produce_gt='simple',
                 split='train',
                 exp_name='',
                 overlap=1,
                 jump_frames=1,
                 filter_edges=False,
                 save_stats=False,
                 directed=True,
                 same_frame=True,
                 next_frame=True,
                 separate_models=False,
                 one_hot_label=True,
                 self_loop=True,
                 normalize=True,
                 debug_visualization=False,
                 which_preprocess='MinMax',
                 drop_feat=[],
                 roi_type='bb_roi',
                 motion_scale=None,
                 ):
        
        
        main_path = os.path.join(get_original_cwd(), main_path) if main_path.startswith('./') else main_path

        
        
        self.separate_models = separate_models
        self.save_stats = save_stats
        self.mul_vals = mul_vals
        self.roi_type = roi_type
        self.motion_scale = None if motion_scale is None else {
            key: float(value) for key, value in dict(motion_scale).items()
        }

        
        flag_2d = '2D' in exp_name
        flag_3d = '3D' in exp_name
        assert not (flag_2d and flag_3d), \
            "Please provide experiment name with only one detailed dimension (e.g. 2D/3D)"
        assert flag_2d or flag_3d, \
            "Please provide experiment name with detailed dimension (e.g. 2D/3D)"
        self.is_3d = flag_3d and (not flag_2d)

        
        self.filter_edges = filter_edges

        
        self.debug_visualization = debug_visualization
        self.drop_feat = list(drop_feat)
        self.normalize = normalize
        self.which_preprocess = which_preprocess

        
        self.edge_feat_embed_dict = edge_feat_embed_dict

        
        self.normalize_all_cols = normalize_all_cols

        
        self.produce_gt = produce_gt
        self.one_hot_label = one_hot_label

        
        self.modes = ["train", "valid", "test"]
        self.type_file = type_file
        self.same_frame = same_frame
        self.next_frame = next_frame
        self.self_loop = self_loop
        self.overlap = overlap
        self.directed = directed
        self.num_frames = num_frames
        self.jump_frames = jump_frames

        
        self.train_seq_len_check = []
        self.dirs_path = dirs_path
        for k, v_list in dirs_path.items():
            for ind, val in enumerate(v_list):
                
                self.dirs_path[k][ind] = osp.join(main_path, val)
                
                self.fill_seq_list(self.dirs_path[k][ind])

        if self.jump_frames > 1:
            print(f"Pay attention! using {jump_frames} jump_frames can make problem in mitosis edges!")

        self.exp_name = exp_name

        
        
        self.all_paths = {}
        for key, mul_path in dirs_path.items():
            if mul_path is None:
                continue
            if isinstance(mul_path, str):
                
                root = self.dirs_path[split]
                curr_path = osp.join(osp.join(mul_path, "processed"), self.exp_name)
                self.all_paths[key] = curr_path
                os.makedirs(curr_path, exist_ok=True)
            elif isinstance(mul_path, Iterable):
                
                root = self.dirs_path[split][0]
                self.all_paths[key] = []
                for path in mul_path:
                    curr_path = osp.join(osp.join(path, "processed"), self.exp_name)
                    self.all_paths[key] += [curr_path]
                    os.makedirs(curr_path, exist_ok=True)
            else:
                assert False, "Can't handle the object type that was inserted for the directory path"

        
        
        
        
        super(CellTrackDataset, self).__init__(root)
        index = self.modes.index(split)
        file_name = self.processed_paths[0].split('/')[-1]
        mul_path = self.all_paths[split]
        if isinstance(mul_path, str):
            read_path = osp.join(mul_path, file_name)
        else:
            read_path = osp.join(mul_path[0], file_name)
        self.data, self.slices = torch.load(read_path)

    
    
    
    @property
    def raw_file_names(self):
        
        return []

    @property
    def processed_file_names(self):
        
        is_directed = 'Directed' if self.directed else 'UnDirected'
        is_norm = f'{self.which_preprocess}_normalized' if self.normalize else 'NotNormalized'

        return [f"./{self.exp_name}/{is_norm}Data_{is_directed}Graph_{self.num_frames}Frames.pt"]

    def download(self):
        
        pass

    
    
    
    def fill_seq_list(self, paths):
        
        curr_dir = os.path.join(paths, self.type_file)
        files = [osp.join(curr_dir, f_name) for f_name in sorted(os.listdir(curr_dir)) if
                 self.type_file in f_name]
        num_files = len(files)

        for ind in range(0, num_files, self.overlap):
            
            if ind + self.num_frames > num_files:
                break
        self.train_seq_len_check.append(ind)

    
    
    
    def true_links(self, df_data):
        
        link_edges = []
        
        for id in np.unique(df_data.id.values):
            mask_id = df_data.id.isin([id])         
            nodes = df_data.index[mask_id].values    
            frames = df_data.frame_num[mask_id].values  

            for ind_node in range(0, nodes.shape[0] - 1):
                
                if frames[ind_node] + self.jump_frames == frames[ind_node + 1]:
                    link_edges.append([nodes[ind_node], nodes[ind_node + 1]])
                    if not self.directed:
                        
                        link_edges.append([nodes[ind_node + 1], nodes[ind_node]])

        return link_edges

    
    
    
    def filter_by_roi(self, df_data_curr, df_data_next):
        
        
        cols = ["centroid_row", "centroid_col"]
        if self.is_3d:
            cols.append("centroid_depth")
        df_data_curr_ceter, df_data_next_ceter = df_data_curr.loc[:, cols], df_data_next.loc[:, cols]

        curr_list = []

        for ind in df_data_curr_ceter.index.values:
            
            row_coord, col_coord = df_data_curr_ceter.centroid_row[ind], df_data_curr_ceter.centroid_col[ind]

            
            max_row, min_row = row_coord + self.curr_roi['row'], row_coord - self.curr_roi['row']
            max_col, min_col = col_coord + self.curr_roi['col'], col_coord - self.curr_roi['col']

            
            row_vals, col_vals = df_data_next_ceter.centroid_row.values, df_data_next_ceter.centroid_col.values
            mask_row = np.bitwise_and(min_row <= row_vals, row_vals <= max_row)
            mask_col = np.bitwise_and(min_col <= col_vals, col_vals <= max_col)
            mask_all = np.bitwise_and(mask_row, mask_col)

            
            if self.is_3d:
                depth_coord = df_data_curr_ceter.centroid_depth[ind]
                max_depth, min_depth = depth_coord + self.curr_roi['depth'], depth_coord - self.curr_roi['depth']
                depth_vals = df_data_next_ceter.centroid_depth.values
                mask_depth = np.bitwise_and(min_depth <= depth_vals, depth_vals <= max_depth)
                mask_all = np.bitwise_and(mask_all, mask_depth)

            
            next_indices = df_data_next_ceter.index[mask_all].values
            curr_indices = np.ones_like(next_indices) * ind
            curr_list += np.concatenate((curr_indices[:, None], next_indices[:, None]), -1).tolist()

        return curr_list

    
    
    
    def same_next_links(self, df_data, link_edges):
        
        same_next_edge_index = []
        
        
        
        
        true_link_set = {tuple(edge) for edge in link_edges}
        iter_frames = np.unique(df_data.frame_num.values)

        for loop_ind, frame_ind in enumerate(iter_frames[:-1]):
            
            mask_frame = df_data.frame_num.isin([frame_ind])
            nodes = df_data.index[mask_frame].values.tolist()

            
            if self.same_frame:
                if self.self_loop:
                    
                    same_next_edge_index += [list(tup) for tup in itertools.product(nodes, nodes)]
                else:
                    
                    same_next_edge_index += [list(tup) for tup in itertools.product(nodes, nodes)
                                             if tup[0] != tup[1]]

            
            if self.next_frame:
                if frame_ind != iter_frames[-1]:
                    
                    mask_next_frame = df_data.frame_num.isin([iter_frames[loop_ind + 1]])
                    next_nodes = df_data.index[mask_next_frame].values.tolist()

                    if self.filter_edges:
                        
                        curr_list = self.filter_by_roi(
                            df_data.loc[mask_frame, :], df_data.loc[mask_next_frame, :])
                        
                        curr_list = [edge for edge in curr_list if tuple(edge) not in true_link_set]
                    else:
                        
                        curr_list = [list(tup) for tup in itertools.product(nodes, next_nodes)
                                     if tup not in true_link_set]

                    if not self.directed:
                        
                        curr_list_opposite = [pairs[::-1] for pairs in curr_list]
                        curr_list = list(itertools.chain.from_iterable(
                            zip(curr_list, curr_list_opposite)))

                    same_next_edge_index += curr_list

        return same_next_edge_index

    
    
    
    def iterator_gt_creator(self, df_data):
        
        frames = np.unique(df_data.frame_num)
        gt = []
        for ind in range(frames.shape[0] - 1):
            curr_frame = frames[ind]
            next_frame = frames[ind + 1]
            mask_frames = df_data.frame_num.isin([curr_frame, next_frame])
            gt.append(self.create_gt(df_data[mask_frames], curr_frame, next_frame))
        return torch.cat(gt, axis=0)

    def create_gt(self, df_data, curr_frame, next_frame):
        
        start_frame_mask = df_data.frame_num.isin([curr_frame])
        next_frame_mask = df_data.frame_num.isin([next_frame])

        start_frame_ids = df_data.id.loc[start_frame_mask].values
        next_frame_ids = df_data.id.loc[next_frame_mask].reset_index().drop(['index'], axis=1)

        
        num_classes = next_frame_ids.index[-1] + 2
        next_frame_ids = next_frame_ids.values.squeeze()

        gt_list = []
        for id in start_frame_ids:
            if np.sum(id == next_frame_ids):
                
                gt_list.append((next_frame_ids == id).astype(int).argmax() + 1)
            else:
                
                gt_list.append(0)

        y = torch.tensor(gt_list)
        if self.one_hot_label:
            y = one_hot(y, num_classes=num_classes).flatten()
        return y

    
    
    
    def preprocess(self, dropped_df):
        
        array = dropped_df.values
        if self.normalize:
            array = self.normalize_array(array)
        return array

    def normalize_array(self, array):
        
        if self.which_preprocess == 'MinMax':
            scaler = MinMaxScaler()
        elif self.which_preprocess == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        if self.separate_models:
            
            array = scaler.fit_transform(array)
        else:
            
            array[:, self.normalize_cols] = scaler.fit_transform(array[:, self.normalize_cols])
        return array

    
    
    
    def edge_feat_embedding(self, x, edge_index):
        
        src, trg = edge_index
        sub_x = x[src] - x[trg]
        abs_sub = np.abs(sub_x)
        
        
        res = abs_sub ** 2 if self.edge_feat_embed_dict['p'] == 2 else abs_sub

        
        if self.edge_feat_embed_dict['normalized_features']:
            res = self.normalize_array(res)
        return res

    
    
    
    def bb_roi(self, df_data):
        
        if self.is_3d:
            cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb',
                    'min_depth_bb', 'max_depth_bb']
        else:
            cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb']

        bb_feat = df_data.loc[:, cols]
        max_row = np.abs(bb_feat.min_row_bb.values - bb_feat.max_row_bb.values).max()
        max_col = np.abs(bb_feat.min_col_bb.values - bb_feat.max_col_bb.values).max()

        self.curr_roi = {'row': max_row * self.mul_vals[0], 'col': max_col * self.mul_vals[1]}
        if self.is_3d:
            max_depth = np.abs(bb_feat.min_depth_bb.values - bb_feat.max_depth_bb.values).max()
            self.curr_roi['depth'] = max_depth * self.mul_vals[2]

    def move_roi(self, df_data, curr_dir):
        
        
        if self.is_3d:
            cols = ['centroid_row', 'centroid_col', 'centroid_depth']
            cols_new = ['diff_row', 'diff_col', 'diff_depth']
        else:
            cols = ['centroid_row', 'centroid_col']
            cols_new = ['diff_row', 'diff_col']

        df_stats = pd.DataFrame(columns=['id'] + cols_new)
        counter = 0

        
        for id in np.unique(df_data.id):
            mask_id = df_data.id.values == id
            df_id = df_data.loc[mask_id, ['frame_num'] + cols]
            for i in range(df_id.shape[0] - 1):
                if not (i + self.jump_frames < df_id.shape[0]):
                    break
                curr_frame_ind = df_id.iloc[i, 0]
                next_frame_ind = df_id.iloc[i + self.jump_frames, 0]

                
                if curr_frame_ind + self.jump_frames != next_frame_ind:
                    continue

                
                diff = df_id.iloc[i, 1:].values - df_id.iloc[i + 1, 1:].values
                df_stats.loc[counter, 'id'] = id
                df_stats.loc[counter, cols_new] = np.abs(diff)
                counter += 1

        
        if self.save_stats:
            path = osp.join(curr_dir, "stats")
            os.makedirs(path, exist_ok=True)
            path = osp.join(path, "df_movement_stats.csv")
            df_stats.to_csv(path)

        
        diff_row = np.abs(df_stats.diff_row.values)
        diff_col = np.abs(df_stats.diff_col.values)
        self.curr_roi = {'row': diff_row.max() + self.mul_vals[0] * diff_row.std(),
                         'col': diff_col.max() + self.mul_vals[1] * diff_col.std()}
        if self.is_3d:
            diff_depth = np.abs(df_stats.diff_depth.values)
            self.curr_roi['depth'] = diff_depth.max() + self.mul_vals[2] * diff_depth.std()

    def find_roi(self, files, curr_dir):
        
        temp_data = [pd.read_csv(file) for file in files]
        df_data = pd.concat(temp_data, axis=0).reset_index(drop=True)
        if self.roi_type == "move_roi":
            self.move_roi(df_data, curr_dir)
        else:
            self.bb_roi(df_data)  

    
    
    
    def create_graph(self, curr_dir, mode):
        
        data_list = []
        drop_col_list = ['id']      
        is_first_time = True

        
        files = [osp.join(curr_dir, f_name) for f_name in sorted(os.listdir(curr_dir)) if
                 self.type_file in f_name]
        num_files = len(files)

        
        print(f"Estimating ROI from {num_files} CSV frames in {curr_dir}", flush=True)
        self.find_roi(files, curr_dir)

        
        if self.num_frames == 'all':
            num_frames = num_files
        elif isinstance(self.num_frames, int):
            num_frames = self.num_frames
        else:
            assert False, f"The provided num_frames {type(self.num_frames)} variable type is not supported"

        total_windows = max(0, 1 + (num_files - num_frames) // self.overlap)
        progress_step = max(1, total_windows // 10)
        skipped_empty_windows = 0
        skipped_edgeless_windows = 0
        print(
            f"Start graph construction: {curr_dir}; window={num_frames}, "
            f"overlap={self.overlap}, total_windows={total_windows}",
            flush=True,
        )

        
        for window_index, ind in enumerate(range(0, num_files, self.overlap), start=1):
            
            if ind + num_frames > num_files:
                break

            
            temp_data = [pd.read_csv(files[ind_tmp])
                         for ind_tmp in range(ind, ind + num_frames, self.jump_frames)]
            df_data = pd.concat(temp_data, axis=0).reset_index(drop=True)
            if df_data.empty:
                skipped_empty_windows += 1
                continue

            
            link_edges = self.true_links(df_data)
            connected_edges = len(link_edges)

            
            if self.same_frame or self.next_frame:
                link_edges += self.same_next_links(df_data, link_edges)

            
            
            
            if not link_edges:
                skipped_edgeless_windows += 1
                continue

            
            edge_index = [torch.tensor([lst], dtype=torch.long) for lst in link_edges]
            edge_index = torch.cat(edge_index, dim=0).t().contiguous()
            motion_edge_feat = torch.from_numpy(relative_motion_features(
                df_data,
                edge_index,
                self.motion_scale or self.curr_roi,
                is_3d=self.is_3d,
                jump_frames=self.jump_frames,
            ))

            
            
            connected_index = torch.zeros(len(link_edges))
            connected_index[:connected_edges] = 1

            if self.produce_gt == 'simple':
                edge_label = connected_index
            else:
                
                edge_label = self.iterator_gt_creator(
                    df_data.reset_index()[['index', 'id', 'frame_num']])

            
            if not ('id' in drop_col_list) and 'id' in df_data.columns:
                drop_col_list.append('id')
                warnings.warn("Find the id label as part of the features and dropped it, please be aware")
            if not ('seg_label' in drop_col_list) and 'seg_label' in df_data.columns:
                drop_col_list.append('seg_label')
                warnings.warn("Find the seg label as part of the features and dropped it, please be aware")

            
            dropped_df = df_data.drop(drop_col_list, axis=1)
            for feat in self.drop_feat:
                if feat in dropped_df.columns:
                    dropped_df = dropped_df.drop([feat], axis=1)

            
            if is_first_time:
                is_first_time = False
                if self.normalize_all_cols:
                    
                    self.normalize_cols = np.ones((dropped_df.shape[-1]), dtype=bool)
                else:
                    
                    self.normalize_cols = np.array(
                        ['feat' != name_col[:len('feat')] for name_col in dropped_df.columns])

                if self.separate_models:
                    
                    
                    self.separate_cols = np.array(
                        ['feat' != name_col[:len('feat')] for name_col in dropped_df.columns])

            
            if not self.separate_models:
                
                x = self.preprocess(dropped_df)
                if self.edge_feat_embed_dict['use_normalized_x']:
                    edge_feat = self.edge_feat_embedding(x, edge_index)
                else:
                    edge_feat = self.edge_feat_embedding(dropped_df.values, edge_index)
                x = torch.FloatTensor(x)
                edge_feat = torch.FloatTensor(edge_feat)

                if torch.any(x.isnan()) or torch.any(edge_feat.isnan()):
                    assert False, "inputs contain nan values"

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_label=edge_label,
                    edge_feat=edge_feat,
                    motion_edge_feat=motion_edge_feat,
                )
            else:
                
                if not self.edge_feat_embed_dict['use_normalized_x']:
                    
                    x = torch.FloatTensor(self.preprocess(dropped_df.loc[:, self.separate_cols]))
                    x_2 = torch.FloatTensor(dropped_df.loc[:, np.logical_not(self.separate_cols)].values)
                    edge_feat = self.edge_feat_embedding(dropped_df.values, edge_index)
                else:
                    
                    x = self.preprocess(dropped_df.loc[:, self.separate_cols])
                    x_2 = dropped_df.loc[:, np.logical_not(self.separate_cols)].values
                    edge_feat = self.edge_feat_embedding(
                        np.concatenate((x, x_2), axis=-1), edge_index)
                    x = torch.FloatTensor(x)
                    x_2 = torch.FloatTensor(x_2)
                edge_feat = torch.FloatTensor(edge_feat)
                
                data = Data(
                    x=x,
                    x_2=x_2,
                    edge_index=edge_index,
                    edge_label=edge_label,
                    edge_feat=edge_feat,
                    motion_edge_feat=motion_edge_feat,
                )

            
            data_list.append(data)
            if window_index % progress_step == 0 or window_index == total_windows:
                print(
                    f"Built graph window {window_index}/{total_windows}: "
                    f"nodes={data.num_nodes}, edges={data.edge_index.shape[1]}",
                    flush=True,
                )

        print(
            f"Num of produced graphs is {len(data_list)} "
            f"(skipped empty={skipped_empty_windows}, "
            f"edgeless={skipped_edgeless_windows})",
            flush=True,
        )
        return data_list

    
    
    
    def process(self):
        
        for ind_mode, mode in enumerate(self.modes):
            if not (mode in self.dirs_path.keys()):
                continue

            curr_dir = self.dirs_path[mode]

            if isinstance(curr_dir, str):
                
                curr_dir = osp.join(curr_dir, self.type_file)  
                data_list = self.create_graph(curr_dir, mode)
                print(f"Finish process {curr_dir} ({mode})")

                file_name = self.processed_paths[0].split('/')[-1]
                write_path = osp.join(self.all_paths[mode][0], file_name)

                print(f"Processed Data is saved to {write_path}")
                torch.save(self.collate(data_list), write_path)

            elif isinstance(curr_dir, Iterable):
                
                data_list = []
                for dir_path in curr_dir:
                    curr_dir_path = osp.join(dir_path, self.type_file)
                    data_list += self.create_graph(curr_dir_path, mode)
                    print(f"Finish process {curr_dir_path} ({mode})")

                    file_name = self.processed_paths[0].split('/')[-1]

                write_path = osp.join(self.all_paths[mode][0], file_name)
                print(f"Processed Data is save to {write_path}")
                torch.save(self.collate(data_list), write_path)

            else:
                assert False, "Can't handle the object type that was inserted for the directory path"
