

import os
import glob
import json
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    _proxy_val = os.environ.get(_proxy_key, "")
    if "<" in _proxy_val or ">" in _proxy_val:
        os.environ.pop(_proxy_key, None)

from src.models.celltrack_plmodel import CellTrackLitModel
from graph_dataset_inference import CellTrackDataset
import warnings
warnings.filterwarnings("ignore")





def predict(
    ckpt_path,
    path_csv_output,
    num_seq,
    intra_window_weight=0.0,
    delta_t=1,
    edge_threshold=0.05,
):
    

    CKPT_PATH = ckpt_path
    path_output = path_csv_output

    
    folder_path = CKPT_PATH
    for i in range(2):
        folder_path = folder_path[:folder_path.rfind('/')]

    config_path = os.path.join(folder_path, '.hydra/config.yaml')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    print(f"load model from: {CKPT_PATH}")

    
    data_yaml = config['datamodule']

    
    trained_model = CellTrackLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)
    print(trained_model.hparams)

    validation_logit_threshold = float(
        trained_model.validation_logit_threshold.detach().cpu().item()
    )
    validation_probability_threshold = float(
        torch.sigmoid(torch.tensor(validation_logit_threshold)).item()
    )
    print(
        "Frozen validation decision threshold: "
        f"logit={validation_logit_threshold:.6g}, "
        f"probability={validation_probability_threshold:.6g}"
    )

    trained_model.eval()
    trained_model.freeze()

    
    training_num_frames = data_yaml['dataset_params'].get('num_frames', 10)
    if training_num_frames == 'all' or not isinstance(training_num_frames, int):
        training_num_frames = 10  
    print(f"Using training window size: num_frames = {training_num_frames}")

    
    num_frames = training_num_frames

    
    
    csv_dir = os.path.join(path_output, num_seq + "_CSV", "csv")
    all_csv_files = sorted(glob.glob(os.path.join(csv_dir, "frame_*.csv")))

    if len(all_csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(all_csv_files)} CSV files in {csv_dir}")

    
    global_df_list = [pd.read_csv(f) for f in all_csv_files]
    all_data_df = pd.concat(global_df_list, ignore_index=True)

    
    global_id_map = {}
    for global_idx, (_, row) in enumerate(all_data_df.iterrows()):
        key = (int(row.frame_num), int(row.seg_label))
        global_id_map[key] = global_idx

    print(f"Global ID map: {len(global_id_map)} cells across {len(all_csv_files)} frames")

    
    
    data_yaml['dataset_params']['num_frames'] = num_frames
    data_yaml['dataset_params']['main_path'] = path_output

    second_path = num_seq
    data_yaml['dataset_params']['dirs_path']['test'] = [second_path + "_CSV"]

    
    
    data_train: CellTrackDataset = CellTrackDataset(
        **data_yaml['dataset_params'], split='test'
    )
    data_list, df_list = data_train.all_data['test']

    print(f"Built {len(data_list)} window graphs (num_frames={num_frames})")

    
    
    
    sp_weights = defaultdict(float)
    sp_accum = defaultdict(float)

    overlap = data_yaml['dataset_params'].get('overlap', 1)
    total_frames = len(all_csv_files)

    for window_idx, (graph, df_window) in enumerate(zip(data_list, df_list)):
        
        x, x2, edge_index, edge_feat = (
            graph.x, graph.x_2, graph.edge_index, graph.edge_feat
        )
        motion_edge_feat = getattr(graph, "motion_edge_feat", None)
        outputs = trained_model(
            (x, x2),
            edge_index,
            edge_feat.float(),
            motion_edge_feat.float() if motion_edge_feat is not None else None,
        )

        
        probs = torch.sigmoid(outputs)

        
        
        
        window_start_frame = int(df_window.frame_num.min())
        t_middle = window_start_frame + (num_frames - 1) / 2.0

        
        for e in range(edge_index.shape[1]):
            local_src = int(edge_index[0, e])
            local_trg = int(edge_index[1, e])
            prob = probs[e].item()

            
            src_frame = int(df_window.iloc[local_src].frame_num)
            trg_frame = int(df_window.iloc[local_trg].frame_num)

            
            dt = trg_frame - src_frame
            if dt <= 0 or dt > delta_t:
                continue

            
            if prob < edge_threshold:
                continue

            
            src_seg = int(df_window.iloc[local_src].seg_label)
            trg_seg = int(df_window.iloc[local_trg].seg_label)
            global_src = global_id_map.get((src_frame, src_seg))
            global_trg = global_id_map.get((trg_frame, trg_seg))

            if global_src is None or global_trg is None:
                continue

            
            
            ddt = src_frame - t_middle
            window_weight = np.exp(-intra_window_weight * ddt * ddt)

            key = (global_src, global_trg)
            sp_weights[key] += window_weight * prob
            sp_accum[key] += window_weight

        if (window_idx + 1) % 50 == 0 or window_idx == 0:
            print(f"  Window {window_idx + 1}/{len(data_list)} "
                  f"(frames {window_start_frame}-{window_start_frame + num_frames - 1}), "
                  f"unique edges so far: {len(sp_weights)}")

    print(f"Total unique edges after merging: {len(sp_weights)}")

    
    
    
    merged_edges = []
    merged_logits = []
    edge_counts = []  

    for (global_src, global_trg), weighted_sum in sp_weights.items():
        total_weight = sp_accum[(global_src, global_trg)]
        avg_prob = weighted_sum / total_weight

        
        avg_prob = np.clip(avg_prob, 1e-7, 1.0 - 1e-7)
        
        logit = np.log(avg_prob / (1.0 - avg_prob))

        merged_edges.append([global_src, global_trg])
        merged_logits.append(logit)
        edge_counts.append(int(total_weight))  

    print(f"Merged edges: {len(merged_edges)}")
    print(f"Avg occurrences per edge: {np.mean(edge_counts):.1f} "
          f"(min={np.min(edge_counts)}, max={np.max(edge_counts)})")

    
    data_path = os.path.join(path_output, second_path) + '_RES_inference'
    print(f"save path : {data_path}")
    os.makedirs(data_path, exist_ok=True)

    edge_index_tensor = torch.tensor(merged_edges, dtype=torch.long).t().contiguous()
    logits_tensor = torch.tensor(merged_logits, dtype=torch.float32)

    file1 = os.path.join(data_path, 'merged_edge_index.pt')
    file2 = os.path.join(data_path, 'all_data_df.csv')
    file3 = os.path.join(data_path, 'raw_output.pt')
    file4 = os.path.join(data_path, 'inference_metadata.json')

    print(f"Save inference files: \n - {file1} \n - {file2} \n - {file3}")
    torch.save(edge_index_tensor, file1)
    all_data_df.to_csv(file2)
    torch.save(logits_tensor, file3)
    with open(file4, 'w') as handle:
        json.dump(
            {
                "schema_version": 1,
                "checkpoint": str(Path(CKPT_PATH).resolve()),
                "validation_logit_threshold": validation_logit_threshold,
                "validation_probability_threshold": validation_probability_threshold,
                "edge_prefilter_probability": float(edge_threshold),
                "threshold_application": (
                    "compare merged raw probability against the frozen validation "
                    "probability threshold during postprocessing"
                ),
            },
            handle,
            indent=2,
        )

    
    print(f"\n===== Inference Summary =====")
    print(f"Total frames: {total_frames}")
    print(f"Window size: {num_frames}")
    print(f"Number of windows: {len(data_list)}")
    print(f"Total cells (global): {len(all_data_df)}")
    print(f"Total unique edges: {len(merged_edges)}")
    print(f"Edge occurrence: mean={np.mean(edge_counts):.1f}, "
          f"median={np.median(edge_counts):.1f}, "
          f"min={np.min(edge_counts)}, max={np.max(edge_counts)}")
    print(f"Boundary edges (< {num_frames} occurrences): "
          f"{sum(1 for c in edge_counts if c < num_frames)}")
    print(f"Inference metadata: {file4}")





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', type=str, required=True,
                        help='model params full path (.ckpt)')
    parser.add_argument('-ns', type=str, required=True,
                        help='sequence identifier used in <id>_CSV')
    parser.add_argument('-oc', type=str, required=True,
                        help='output csv directory')

    args = parser.parse_args()

    model_path = args.mp
    num_seq = args.ns
    output_csv = args.oc

    if (
        not num_seq
        or num_seq in {'.', '..'}
        or '/' in num_seq
        or '\\' in num_seq
    ):
        raise ValueError(f"Invalid sequence identifier: {num_seq!r}")

    predict(
        ckpt_path=model_path,
        path_csv_output=output_csv,
        num_seq=num_seq,
        intra_window_weight=0.0,   
        delta_t=1,                  
        edge_threshold=0.05,        
    )
