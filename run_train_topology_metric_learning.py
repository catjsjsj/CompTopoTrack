

import os, sys, json, random, torch
import numpy as np
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)




CONFIG = {
    "seed": 17,
    
    "model_name": "resnet18",
    "embedding_dim": 128,
    "normalized_feat": True,

    
    "lr_trunk": 1e-5,
    "lr_embedder": 1e-4,
    "weight_decay": 1e-4,

    
    "num_epochs": 50,
    "effective_batch_size": 2,
    "frame_pairs_per_epoch": 0,
    "validation_pairs_per_sequence": 0,
    "image_forward_batch_size": 64,
    "activation_checkpoint": True,
    "amp": True,
    "val_interval": 1,
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.0,

    
    "enabled_layers": [1],
    "same_frame_cluster_contrast": False,
    "same_frame_cluster_weight": 0.0,

    
    "gamma": 0.1,
    "gamma_direct": 0.1,
    "gamma_closure": 0.1,
    "degree_weight": 0.0,
    "closure_max_component_size": 50,
    "alpha_max": 0.0,
    "curriculum_epochs": 500,
    "influence_type": "degree",
    "influence_scope": "global",
    "horizon": 6,
    "merge_parent_child": False,

    
    "ms_alpha": 2.0,
    "ms_beta": 50.0,
    "ms_base": 0.5,

    
    "roi_type": "bb_roi",
    "roi_mul_vals": (1.0, 1.0),

    
    "data_dir_img": f"{BASE}/data/CTC/Training/Fluo-N2DH-SIM+_train",
    "data_dir_mask": f"{BASE}/data/CTC/Training/Fluo-N2DH-SIM+_train",
    "subdir_mask": "GT/TRA",
    "dir_csv": f"{BASE}/data/basic_features_sim_plus/Fluo-N2DH-SIM+_train",
    "num_sequences": 1,
    "sequences_names": ["01"],
    "start_index": 1,
    "frame_start": None,
    "max_frames": None,
    "train_val_test_split": [45, 20, 0],
    "deviation": "with_overlap",
    "explicit_dataset_splits": None,
    "normalization_policy": "per_sequence_with_source_fallback",

    
    "exp_name": "sim_plus_metric",
    "output_root": f"{BASE}/outputs/metric",
}


def main():
    cfg = CONFIG

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        os.path.abspath(cfg.get("output_root", os.path.join(BASE, "outputs/metric"))),
        f"{cfg['exp_name']}_{timestamp}",
    )
    model_dir = os.path.join(exp_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    print(f"Output: {exp_dir}")
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    
    from src_metric_learning.Data.dataset_2D import ImgDataset

    data_cfg = {
        "pad_value": 0, "norm_value": 0, "normalize_type": "MinMaxCell",
        "train_val_test_split": cfg["train_val_test_split"],
        "deviation": cfg["deviation"],
        "data_dir_img": cfg["data_dir_img"],
        "data_dir_mask": cfg["data_dir_mask"],
        "subdir_mask": cfg["subdir_mask"],
        "dir_csv": cfg["dir_csv"],
        "curr_seq": 1, "num_sequences": cfg["num_sequences"],
        "sequences_names": cfg.get("sequences_names"),
        "start_index": cfg.get("start_index", 1),
        "frame_start": cfg.get("frame_start"),
        "max_frames": cfg.get("max_frames"),
        "type_img": "tif",
    }
    print("Loading datasets...")
    explicit_splits = cfg.get("explicit_dataset_splits")
    if explicit_splits:
        datasets = {}
        for split_name, type_data in (("train", "train"), ("valid", "valid"), ("test", "test")):
            split_cfg = explicit_splits[split_name]
            split_data_cfg = dict(data_cfg)
            split_data_cfg.update(split_cfg)
            split_data_cfg["num_sequences"] = len(split_data_cfg["sequences_names"])
            datasets[split_name] = ImgDataset(**split_data_cfg, type_data=type_data)
        train_ds, val_ds, test_ds = datasets["train"], datasets["valid"], datasets["test"]
    else:
        train_ds = ImgDataset(**data_cfg, type_data="train")
        val_ds   = ImgDataset(**data_cfg, type_data="valid")
        test_ds  = ImgDataset(**data_cfg, type_data="test")
    if cfg.get("normalization_policy") == "source_global":
        source_min = min(float(value) for value in train_ds.min_cell)
        source_max = max(float(value) for value in train_ds.max_cell)
        for dataset in (train_ds, val_ds, test_ds):
            dataset.min_cell = [source_min] * dataset.num_sequences
            dataset.max_cell = [source_max] * dataset.num_sequences
        print(
            "Using source-global MinMaxCell normalization: "
            f"min={source_min:g}, max={source_max:g}"
        )
    print(f"Train: {len(train_ds)} cells, Val: {len(val_ds)} cells")

    
    from src_metric_learning.graph_topology.physical_graph import PhysicalGraphBuilder
    roi_type = cfg.get("roi_type", "bb_roi")
    if roi_type == "move_roi":
        roi_row, roi_col = PhysicalGraphBuilder.compute_roi_from_displacements(
            train_ds.org_df_cells, mul_vals=cfg["roi_mul_vals"])
    else:
        roi_row = train_ds.curr_roi["row"] * cfg["roi_mul_vals"][0]
        roi_col = train_ds.curr_roi["col"] * cfg["roi_mul_vals"][1]
    print(f"ROI ({roi_type}): row={roi_row:.1f}, col={roi_col:.1f}")

    
    from src_metric_learning.graph_topology.weight_computation import TopologyWeightComputer
    computer = TopologyWeightComputer(
        roi_row=roi_row, roi_col=roi_col,
        gamma=cfg["gamma"], alpha_max=cfg["alpha_max"],
        curriculum_epochs=cfg["curriculum_epochs"],
        merge_parent_child=cfg["merge_parent_child"],
        horizon=cfg["horizon"],
        influence_type=cfg.get("influence_type", "degree"),
        influence_scope=cfg.get("influence_scope", "cluster_gated"),
        gamma_direct=cfg.get("gamma_direct"),
        gamma_closure=cfg.get("gamma_closure"),
        degree_weight=cfg.get("degree_weight"),
        closure_max_component_size=cfg.get("closure_max_component_size", 50),
    )
    def compute_topologies(dataset, label):
        topologies = []
        for seq_idx in range(dataset.num_sequences):
            topo = computer.compute(dataset.org_df_cells[seq_idx],
                                    enabled_layers=topology_layers)
            if topology_layers != sorted(set(cfg["enabled_layers"])):
                computer.update_weights_for_epoch(topo, epoch=0,
                                                   enabled_layers=cfg["enabled_layers"])
            print(f"{label} seq {seq_idx}: {topo.total_frame_pairs} frame pairs, tau={topo.tau:.2f}")
            topologies.append(topo)
        return topologies

    train_topos = []
    topology_layers = sorted(set(cfg["enabled_layers"]) |
                             ({2} if cfg.get("same_frame_cluster_contrast", False) else set()))
    train_topos = compute_topologies(train_ds, "Train")
    if explicit_splits:
        val_topos = compute_topologies(val_ds, "Valid")
        test_topos = compute_topologies(test_ds, "Test")
    else:
        val_topos = train_topos
        test_topos = train_topos

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src_metric_learning.modules.resnet_2d.resnet import set_model_architecture, MLP
    import torch.nn as nn

    class EmbedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = set_model_architecture(cfg["model_name"])
            self.embedder = MLP(
                [self.trunk.input_features_fc_layer, cfg["embedding_dim"]],
                normalized_feat=cfg["normalized_feat"])
        def forward(self, x):
            return self.embedder(self.trunk(x))

    model = EmbedModel().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.trunk.parameters(),    "lr": cfg["lr_trunk"]},
        {"params": model.embedder.parameters(), "lr": cfg["lr_embedder"]},
    ], weight_decay=cfg["weight_decay"])

    
    from src_metric_learning.Data.frame_pair_loader import FramePairImageLoader
    train_loader = FramePairImageLoader(train_ds)
    val_loader = FramePairImageLoader(val_ds)
    test_loader = FramePairImageLoader(test_ds)
    from src_metric_learning.losses.weighted_ms_loss import WeightedMultiSimilarityLoss
    from src_metric_learning.trainers.topology_aware_trainer import TopologyAwareTrainer
    from src_metric_learning.trainers.frame_pair_validator import FramePairValidator

    validation_pair_cap = cfg.get("validation_pairs_per_sequence", 0)
    val_validator = FramePairValidator(
        val_loader, val_topos, device,
        max_pairs_per_sequence=validation_pair_cap,
    )
    test_validator = FramePairValidator(
        test_loader, test_topos, device,
        max_pairs_per_sequence=validation_pair_cap,
    )
    loss_fn = WeightedMultiSimilarityLoss(
        alpha=cfg["ms_alpha"], beta=cfg["ms_beta"], base=cfg["ms_base"])

    trainer = TopologyAwareTrainer(
        model=model, optimizer=optimizer, device=device,
        topology_computer=computer, train_topologies=train_topos,
        val_validator=val_validator, loss_fn=loss_fn,
        frame_pair_loader=train_loader,
        num_epochs=cfg["num_epochs"],
        effective_batch_size=cfg["effective_batch_size"],
        enabled_layers=cfg["enabled_layers"],
        same_frame_cluster_contrast=cfg.get("same_frame_cluster_contrast", False),
        same_frame_cluster_weight=cfg.get("same_frame_cluster_weight", 0.1),
        early_stop_patience=cfg.get("early_stop_patience", 0),
        early_stop_min_delta=cfg.get("early_stop_min_delta", 0.0),
        amp=cfg.get("amp", False),
        frame_pairs_per_epoch=cfg.get("frame_pairs_per_epoch", 0),
        model_folder=model_dir,
        val_interval=cfg["val_interval"],
    )

    print(f"\nTraining: {cfg['num_epochs']} epochs, "
          f"{len(trainer._frame_pair_list)} frame pairs, "
          f"layers={cfg['enabled_layers']}, "
          f"same_frame_cluster={cfg.get('same_frame_cluster_contrast', False)} "
          f"(weight={cfg.get('same_frame_cluster_weight', 0.0)})")
    trainer.train()

    
    best_ckpt = os.path.join(model_dir, "model_best.pth")
    sd = torch.load(best_ckpt, map_location="cpu")["model_state_dict"] \
         if os.path.exists(best_ckpt) else model.state_dict()

    
    
    model.load_state_dict(sd)
    test_result = test_validator.evaluate(model)
    print(
        "Test: "
        f"MRR={test_result['mrr']:.4f}, "
        f"P@1={test_result['precision_at_1']:.4f} "
        f"({test_result['n_anchors']} anchors)"
    )

    trunk_sd = {k.replace("trunk.", ""): v for k, v in sd.items()
                if k.startswith("trunk.")}
    emb_sd   = {k.replace("embedder.", ""): v for k, v in sd.items()
                if k.startswith("embedder.")}

    tmp_trunk = set_model_architecture(cfg["model_name"])
    sequence_names = (
        explicit_splits["train"]["sequences_names"]
        if explicit_splits else cfg.get("sequences_names")
    )
    if sequence_names is None:
        if cfg["num_sequences"] == 1:
            sequence_names = [f"{int(data_cfg['curr_seq']):02d}"]
        else:
            sequence_names = [
                f"{index:02d}"
                for index in range(
                    int(cfg.get("start_index", 1)),
                    int(cfg.get("start_index", 1)) + int(cfg["num_sequences"]),
                )
            ]
    normalization_by_sequence = {
        str(seq): {
            "min_cell": float(train_ds.min_cell[index]),
            "max_cell": float(train_ds.max_cell[index]),
        }
        for index, seq in enumerate(sequence_names)
    }
    if cfg.get("normalization_policy") == "source_global":
        normalization_by_sequence = {}
    source_min = min(float(value) for value in train_ds.min_cell)
    source_max = max(float(value) for value in train_ds.max_cell)

    all_params = {
        "min_cell": train_ds.min_cell, "max_cell": train_ds.max_cell,
        "normalization_by_sequence": normalization_by_sequence,
        "normalization_fallback": {
            "min_cell": source_min,
            "max_cell": source_max,
            "policy": "training_source_global_range",
        },
        "training_sequences": list(sequence_names),
        "pad_value": train_ds.pad_value, "roi": train_ds.curr_roi,
        "model_name": cfg["model_name"],
        "mlp_dims": [tmp_trunk.input_features_fc_layer, cfg["embedding_dim"]],
        "mlp_normalized_features": cfg["normalized_feat"],
        "trunk_state_dict": trunk_sd, "embedder_state_dict": emb_sd,
    }
    torch.save(all_params, os.path.join(exp_dir, "all_params.pth"))

    summary = {
        "exp_dir": exp_dir,
        "model_dir": model_dir,
        "all_params": os.path.join(exp_dir, "all_params.pth"),
        "best_mrr": float(trainer.best_val_mrr),
        "best_val_result": trainer.best_val_result,
        "early_stop_triggered": bool(trainer.early_stop_triggered),
        "early_stop_reason": trainer.early_stop_reason,
        "test_result": test_result,
        "config": cfg,
    }
    with open(os.path.join(exp_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {exp_dir}")
    print(f"  Best MRR: {trainer.best_val_mrr:.4f}")
    return summary

if __name__ == "__main__":
    main()
