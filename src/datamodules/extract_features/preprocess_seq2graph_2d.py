

import os
import os.path as op
import shutil
from PIL import Image
import torch
import numpy as np
import pandas as pd
import cv2
import warnings
import re
from torch.utils.data import Dataset
from skimage.measure import regionprops
from hydra.utils import get_original_cwd, to_absolute_path


from src_metric_learning.modules.resnet_2d.resnet import set_model_architecture, MLP





class TestDataset(Dataset):
    

    def __init__(self,
                 path: str,
                 path_masks: str,
                 path_result: str,
                 type_img: str,
                 sec_path,
                 frame_start=None,
                 max_frames=None,
                 ):
        
        
        path = os.path.join(get_original_cwd(), path) if path.startswith('./') else path
        path_masks = os.path.join(get_original_cwd(), path_masks) if path_masks.startswith('./') else path_masks
        path_result = os.path.join(get_original_cwd(), path_result) if path_result.startswith('./') else path_result

        self.path = path
        self.sec_path = sec_path
        self.path_result = path_result
        type_masks = type_img
        dir_img = path
        dir_masks = path_masks
        dir_results = path_result

        
        assert os.path.exists(dir_img), f"Image paths ({dir_img}) is not exist, please fix it!"
        assert os.path.exists(dir_masks), f"Masks paths ({dir_masks}) is not exist, please fix it!"
        assert os.path.exists(dir_results), f"Result paths ({dir_results}) is not exist, please fix it!"

        
        self.images = []
        if os.path.exists(dir_img):
            self.images = [os.path.join(dir_img, fname) for fname in sorted(os.listdir(dir_img))
                           if type_img in fname]

        
        self.masks = []
        if os.path.exists(dir_masks):
            self.masks = [os.path.join(dir_masks, fname) for fname in sorted(os.listdir(dir_masks))
                          if type_masks in fname]

        
        self.results = []
        if os.path.exists(dir_results):
            self.results = [os.path.join(dir_results, fname) for fname in sorted(os.listdir(dir_results))
                           if type_masks in fname]

        if frame_start is not None or max_frames is not None:
            self._restrict_frame_window(frame_start, max_frames)

    @staticmethod
    def _frame_number(path):
        match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
        if match is None:
            raise ValueError(f"Cannot parse frame number from {path}")
        return int(match.group(1))

    def _restrict_frame_window(self, frame_start, max_frames):
        
        if not self.images or not self.masks or not self.results:
            raise ValueError("Frame-window extraction requires images, TRA masks, and SEG masks.")

        start = 0 if frame_start is None else int(frame_start)
        selected = [self._frame_number(path) for path in self.images
                    if self._frame_number(path) >= start]
        if max_frames is not None:
            selected = selected[:int(max_frames)]
        if not selected:
            raise ValueError(f"No frames remain after frame_start={frame_start}, max_frames={max_frames}.")

        selected_set = set(selected)
        self.images = [p for p in self.images if self._frame_number(p) in selected_set]
        self.masks = [p for p in self.masks if self._frame_number(p) in selected_set]
        self.results = [p for p in self.results if self._frame_number(p) in selected_set]
        if not (len(self.images) == len(self.masks) == len(self.results) == len(selected)):
            raise ValueError("Selected image, TRA, and SEG frame counts are not aligned.")

    def __getitem__(self, idx):
        
        
        assert len(self.images) or len(self.images), "both directories are empty, please fix it!"

        
        im_path, image = None, None
        if len(self.images):
            im_path = self.images[idx]
            image = np.array(Image.open(im_path))

        
        mask_path, mask = None, None
        if len(self.masks):
            mask_path = self.masks[idx]
            mask = np.array(Image.open(mask_path))

        
        result_path, result = None, None
        if len(self.results):
            result_path = self.results[idx]
            result = np.array(Image.open(result_path))

        
        
        flag = True
        if im_path is not None:
            flag = False
            im_num = im_path.split(".")[-2][-3:]
        if mask_path is not None:
            flag = False
            mask_num = mask_path.split(".")[-2][-3:]
        if result_path is not None:
            flag = False
            result_num = result_path.split(".")[-2][-3:]

        
        if flag:
            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

        return image, mask, result, im_path, mask_path, result_path

    def __len__(self):
        
        return len(self.images)

    def padding(self, img):
        
        
        
        
        desired_size_row = max(int(self.roi_model['row']), int(img.shape[0]))
        desired_size_col = max(int(self.roi_model['col']), int(img.shape[1]))
        delta_row = desired_size_row - img.shape[0]
        delta_col = desired_size_col - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2
        image = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top,
                                   pad_left, delta_col - pad_left,
                                   cv2.BORDER_CONSTANT, value=self.pad_value)
        return image

    def prepare_metric_patch(self, bbox, img, seg_mask, ind,
                             normalize_type='MinMaxCell'):
        
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb].copy()
        msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        img_patch[msk_patch] = self.pad_value
        img_patch = img_patch.astype(np.float32)
        if normalize_type == 'regular':
            output = self.padding(img_patch) / self.max_img
        elif normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.min_cell) \
                                       / (self.max_cell - self.min_cell)
            img_patch[not_msk_patch] = np.clip(img_patch[not_msk_patch], 0.0, 1.0)
            output = self.padding(img_patch)
        else:
            raise ValueError(f"Unsupported normalization type: {normalize_type}")
        return torch.from_numpy(output).float()

    def extract_freature_metric_learning(self, bbox, img, seg_mask, ind,
                                         normalize_type='MinMaxCell'):
        
        img = self.prepare_metric_patch(
            bbox, img, seg_mask, ind, normalize_type=normalize_type,
        ).to(self.device)
        with torch.no_grad():
            
            
            embedded_img = self.embedder(self.trunk(img[None, None, ...]))

        
        return embedded_img.cpu().numpy().squeeze()

    def preprocess_basic_features(self, path_to_write):
        
        
        cols = ["id",
                "frame_num",
                "area",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "major_axis_length", "minor_axis_length",
                "max_intensity", "mean_intensity", "min_intensity"
                ]

        
        for ind_data in range(self.__len__()):
            
            img, mask, result, im_path, mask_path, result_path = self[ind_data]

            
            im_num, mask_num = im_path.split(".")[-2][-3:], mask_path.split(".")[-2][-3:]
            result_num = result_path.split(".")[-2][-3:]

            
            assert im_num == mask_num, \
                f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            assert im_num == result_num, \
                f"Image number ({im_num}) is not equal to result number ({result_num})"

            rows = []
            
            
            
            for properties in regionprops(result, img):
                id_res = int(properties.label)
                min_row, min_col, max_row, max_col = properties.bbox
                local_tra = mask[min_row:max_row, min_col:max_col]
                overlap_labels = local_tra[properties.image]
                overlap_labels = overlap_labels[overlap_labels != 0].astype(np.int64)
                res_label = (
                    int(np.bincount(overlap_labels).argmax())
                    if overlap_labels.size else 0
                )

                
                if res_label == 0:
                    warnings.warn(f"Pay Attention! there is no result for {id_res}!")
                    continue

                rows.append({
                    "id": res_label,
                    "frame_num": int(im_num),
                    "area": properties.area,
                    "min_row_bb": min_row,
                    "min_col_bb": min_col,
                    "max_row_bb": max_row,
                    "max_col_bb": max_col,
                    "centroid_row": np.int16(np.round(properties.centroid[0])),
                    "centroid_col": np.int16(np.round(properties.centroid[1])),
                    "major_axis_length": properties.major_axis_length,
                    "minor_axis_length": properties.minor_axis_length,
                    "max_intensity": properties.max_intensity,
                    "mean_intensity": properties.mean_intensity,
                    "min_intensity": properties.min_intensity,
                })

            df = pd.DataFrame(rows, columns=cols)

            
            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")

            
            
            sub_dir = op.join(self.path.split("/")[-2], self.sec_path)
            full_dir = op.join(path_to_write, sub_dir)
            full_dir = op.join(full_dir, "csv")
            os.makedirs(to_absolute_path(full_dir), exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            df.to_csv(to_absolute_path(file_path), index=False)

        print(f"files were saved to : {full_dir}")

    def preprocess_features_metric_learning(
            self, path_to_write, dict_path, image_batch_size=32):
        
        if image_batch_size < 1:
            raise ValueError(
                f"image_batch_size must be positive, got {image_batch_size}"
            )

        
        dict_params = torch.load(dict_path, map_location="cpu")

        
        
        
        sequence_stats = dict_params.get("normalization_by_sequence", {}).get(
            str(self.sec_path)
        )
        if sequence_stats is None:
            sequence_stats = dict_params.get("normalization_fallback")
        if sequence_stats is not None:
            self.min_cell = float(sequence_stats["min_cell"])
            self.max_cell = float(sequence_stats["max_cell"])
        else:
            index = int(self.sec_path) - 1
            if index >= len(dict_params["min_cell"]):
                raise ValueError(
                    "Model has no normalization statistics for sequence "
                    f"{self.sec_path}. Re-export the checkpoint with "
                    "normalization_fallback for cross-sequence evaluation."
                )
            self.min_cell = dict_params['min_cell'][index]
            self.max_cell = dict_params['max_cell'][index]
        if not self.max_cell > self.min_cell:
            raise ValueError(
                f"Invalid normalization range: {self.min_cell}..{self.max_cell}"
            )
        self.roi_model = dict_params['roi']
        self.pad_value = dict_params['pad_value']

        
        model_name = dict_params['model_name']
        mlp_dims = dict_params['mlp_dims']
        mlp_normalized_features = dict_params['mlp_normalized_features']

        
        trunk_state_dict = dict_params['trunk_state_dict']
        embedder_state_dict = dict_params['embedder_state_dict']

        
        trunk = set_model_architecture(model_name)
        trunk.load_state_dict(trunk_state_dict)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trunk = trunk.to(self.device)
        self.trunk.eval()  

        
        embedder = MLP(mlp_dims, normalized_feat=mlp_normalized_features)  
        embedder.load_state_dict(embedder_state_dict)
        
        self.embedder = embedder.to(self.device)
        self.embedder.eval()  
        print(f"Feature extraction device: {self.device}")
        print(f"Feature extraction image batch size: {image_batch_size}")

        
        cols = ["id", "seg_label",
                "frame_num",
                "area",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "major_axis_length", "minor_axis_length",
                "max_intensity", "mean_intensity", "min_intensity"
                ]

        
        cols_resnet = [f'feat_{i}' for i in range(mlp_dims[-1])]
        cols += cols_resnet

        
        
        sub_dir = op.join(self.path.split("/")[-2], self.sec_path)
        full_dir = op.join(path_to_write, sub_dir, "csv")
        os.makedirs(to_absolute_path(full_dir), exist_ok=True)

        
        for ind_data in range(self.__len__()):
            
            img, mask, result, im_path, mask_path, result_path = self[ind_data]

            
            im_num, mask_num = im_path.split(".")[-2][-3:], mask_path.split(".")[-2][-3:]
            result_num = result_path.split(".")[-2][-3:]

            
            assert im_num == mask_num, \
                f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            assert im_num == result_num, \
                f"Image number ({im_num}) is not equal to result number ({result_num})"

            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            if os.path.exists(to_absolute_path(file_path)):
                print(f"Skip existing feature CSV: frame {im_num}")
                continue

            rows = []
            patches_by_shape = {}
            for properties in regionprops(result, img):
                id_res = int(properties.label)
                min_row, min_col, max_row, max_col = properties.bbox
                local_tra = mask[min_row:max_row, min_col:max_col]
                overlap_labels = local_tra[properties.image]
                overlap_labels = overlap_labels[overlap_labels != 0].astype(np.int64)
                res_label = (
                    int(np.bincount(overlap_labels).argmax())
                    if overlap_labels.size else 0
                )

                
                if res_label == 0:
                    warnings.warn(f"Pay Attention! there is no result for {id_res}!")
                    continue

                row_index = len(rows)
                rows.append({
                    "id": res_label,
                    "seg_label": id_res,
                    "frame_num": int(im_num),
                    "area": properties.area,
                    "min_row_bb": min_row,
                    "min_col_bb": min_col,
                    "max_row_bb": max_row,
                    "max_col_bb": max_col,
                    "centroid_row": np.int16(np.round(properties.centroid[0])),
                    "centroid_col": np.int16(np.round(properties.centroid[1])),
                    "major_axis_length": properties.major_axis_length,
                    "minor_axis_length": properties.minor_axis_length,
                    "max_intensity": properties.max_intensity,
                    "mean_intensity": properties.mean_intensity,
                    "min_intensity": properties.min_intensity,
                })
                patch = self.prepare_metric_patch(properties.bbox, img, result, id_res)
                patches_by_shape.setdefault(tuple(patch.shape), []).append((row_index, patch))

            feature_values = [None] * len(rows)
            with torch.no_grad():
                for shape_entries in patches_by_shape.values():
                    for start in range(0, len(shape_entries), image_batch_size):
                        batch_entries = shape_entries[start:start + image_batch_size]
                        batch = torch.stack([entry[1] for entry in batch_entries])[:, None].to(self.device)
                        embedded = self.embedder(self.trunk(batch)).cpu().numpy()
                        for (row_index, _), values in zip(batch_entries, embedded):
                            feature_values[row_index] = values

            for row, values in zip(rows, feature_values):
                row.update({column: value for column, value in zip(cols_resnet, values)})
            df = pd.DataFrame(rows, columns=cols)

            
            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")

            
            df.to_csv(to_absolute_path(file_path), index=False)
            print(f"Saved feature CSV: frame {im_num} ({ind_data + 1}/{self.__len__()})")

        print(f"files were saved to : {full_dir}")





def  create_csv(input_images, input_masks, input_seg,
               input_model, output_csv, basic=False,
               sequences=['01', '02'], seg_dir='_ST/SEG',
               frame_start=None, max_frames=None,
               image_batch_size=32,
               ):
    
    dict_path = input_model
    path_output = output_csv
    path_Seg_result = input_seg

    
    for seq in sequences:
        
        curr_img_path = os.path.join(input_images, seq)
        curr_msk_path = os.path.join(input_masks, seq + "_GT/TRA")
        curr_seg_path = os.path.join(path_Seg_result, seq + seg_dir)

        
        ds = TestDataset(
            path=curr_img_path,
            path_masks=curr_msk_path,
            path_result=curr_seg_path,
            type_img="tif",
            sec_path=seq,
            frame_start=frame_start,
            max_frames=max_frames)

        
        if basic:
            
            ds.preprocess_basic_features(path_to_write=path_output)
        else:
            
            ds.preprocess_features_metric_learning(
                path_to_write=path_output,
                dict_path=dict_path,
                image_batch_size=image_batch_size,
            )





if __name__ == "__main__":
    import argparse

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ii', type=str, required=True,
                        help='input images directory')
    parser.add_argument('-imsk', type=str, required=True,
                        help='input TRA masks directory')
    parser.add_argument('-iseg', type=str, required=True,
                        help='input segmentation directory')
    parser.add_argument('-im', type=str, required=True,
                        help='metric learning model params directory')
    parser.add_argument('-sd', type=str, default=None,
                        help='segmentation directory name')
    parser.add_argument('-seq', type=str, default=None, nargs="*",
                        help='sequences list of strings')
    parser.add_argument('-oc', type=str, required=True,
                        help='output csv directory')

    args = parser.parse_args()

    
    input_images = args.ii
    input_masks = args.imsk
    input_segmentation = args.iseg
    input_model = args.im
    output_csv = args.oc
    seg_dir = args.sd
    sequences = args.seq

    
    create_csv(input_images, input_segmentation, input_model, output_csv,
               sequences, seg_dir)
