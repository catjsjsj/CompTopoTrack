

import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple
import numpy as np
import tqdm

from ..graph_topology.weight_computation import TopologyWeightComputer
from ..graph_topology.data_structures import FramePairTopology, SequenceTopology
from ..losses.weighted_ms_loss import WeightedMultiSimilarityLoss


class TopologyAwareTrainer:
    

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        topology_computer: TopologyWeightComputer,
        train_topologies: List[SequenceTopology],
        val_dataset: Optional[Dataset] = None,
        val_validator = None,
        loss_fn: Optional[nn.Module] = None,
        frame_pair_loader=None,
        num_epochs: int = 100,
        effective_batch_size: int = 4,
        model_folder: str = "./saved_models",
        enabled_layers: List[int] = None,
        val_interval: int = 5,
        same_frame_cluster_contrast: bool = False,
        same_frame_cluster_weight: float = 0.1,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        amp: bool = False,
        frame_pairs_per_epoch: int = 0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.topology_computer = topology_computer
        self.train_topologies = train_topologies
        self.val_dataset = val_dataset
        self.val_validator = val_validator  
        self.loss_fn = loss_fn or WeightedMultiSimilarityLoss()
        self.frame_pair_loader = frame_pair_loader
        self.num_epochs = num_epochs
        self.effective_batch_size = effective_batch_size
        self.model_folder = model_folder
        self.enabled_layers = enabled_layers or [1, 2, 3]
        self.val_interval = val_interval
        self.same_frame_cluster_contrast = same_frame_cluster_contrast
        self.same_frame_cluster_weight = float(same_frame_cluster_weight)
        self.early_stop_patience = int(early_stop_patience or 0)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.amp = bool(amp and device.type == "cuda")
        self.frame_pairs_per_epoch = int(frame_pairs_per_epoch or 0)
        if self.frame_pairs_per_epoch < 0:
            raise ValueError("frame_pairs_per_epoch must be non-negative")
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.best_val_mrr = 0.0  
        self.best_val_result = None
        self.metric_history = []
        self.current_epoch = 0
        self.global_step = 0
        self.early_stop_triggered = False
        self.early_stop_reason = ""

        
        self._frame_pair_list = self._collect_frame_pairs()

    
    
    

    def train(self):
        
        os.makedirs(self.model_folder, exist_ok=True)
        checks_without_improvement = 0

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            
            if 3 in self.enabled_layers:
                for topo in self.train_topologies:
                    self.topology_computer.update_weights_for_epoch(
                        topo, epoch, self.enabled_layers,
                    )

            
            self.model.train()
            total_loss = 0.0
            n_pairs = 0
            available_pair_count = len(self._frame_pair_list)
            if 0 < self.frame_pairs_per_epoch < available_pair_count:
                indices = np.random.choice(
                    available_pair_count,
                    size=self.frame_pairs_per_epoch,
                    replace=False,
                )
                indices = np.random.permutation(indices)
            else:
                indices = np.random.permutation(available_pair_count)

            pbar = tqdm.tqdm(
                range(0, len(indices), self.effective_batch_size),
                desc=f"Epoch {epoch}",
            )
            for batch_start in pbar:
                batch_end = min(batch_start + self.effective_batch_size,
                                len(indices))
                batch_indices = indices[batch_start:batch_end]

                batch_loss = 0.0
                valid_pairs = 0
                self.optimizer.zero_grad()

                for idx in batch_indices:
                    seq_idx, _fp_idx, fp = self._frame_pair_list[idx]
                    loss = self._train_on_frame_pair(seq_idx, fp)
                    if loss is not None:
                        
                        
                        self.grad_scaler.scale(loss).backward()
                        batch_loss += float(loss.detach().item())
                        valid_pairs += 1

                if valid_pairs == 0:
                    continue

                self.grad_scaler.unscale_(self.optimizer)
                for group in self.optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.div_(valid_pairs)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                avg_loss = batch_loss / valid_pairs
                total_loss += batch_loss
                n_pairs += valid_pairs
                self.global_step += 1

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "alpha": f"{self._current_alpha():.2f}",
                })

            avg_epoch_loss = total_loss / max(n_pairs, 1)
            print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")
            epoch_record = {
                "epoch": int(epoch),
                "avg_loss": float(avg_epoch_loss),
                "n_pairs": int(n_pairs),
                "available_frame_pairs": int(available_pair_count),
                "sampled_frame_pairs": int(len(indices)),
                "alpha": float(self._current_alpha()),
                "global_step": int(self.global_step),
            }
            improved = False
            checked_validation = False

            
            if self.val_validator is not None and epoch % self.val_interval == 0:
                checked_validation = True
                val_result = self.val_validator.evaluate(self.model)
                mrr = val_result['mrr']
                p1 = val_result['precision_at_1']
                print(f"Epoch {epoch}: MRR={mrr:.4f}, P@1={p1:.4f} "
                      f"({val_result['n_anchors']} anchors)")
                epoch_record.update({
                    "mrr": float(mrr),
                    "precision_at_1": float(p1),
                    "mean_rank": float(val_result.get("mean_rank", 0.0)),
                    "n_anchors": int(val_result.get("n_anchors", 0)),
                    "n_skipped": int(val_result.get("n_skipped", 0)),
                })
                if mrr > self.best_val_mrr + self.early_stop_min_delta:
                    self.best_val_mrr = mrr
                    self.best_val_result = dict(epoch_record)
                    self._save_checkpoint("best")
                    improved = True
            elif self.val_dataset is not None and epoch % self.val_interval == 0:
                checked_validation = True
                
                val_acc = self._validate_knn()
                print(f"Epoch {epoch}: val_precision@1={val_acc:.4f}")
                epoch_record.update({
                    "mrr": float(val_acc),
                    "precision_at_1": float(val_acc),
                })
                if val_acc > self.best_val_mrr + self.early_stop_min_delta:
                    self.best_val_mrr = val_acc
                    self.best_val_result = dict(epoch_record)
                    self._save_checkpoint("best")
                    improved = True

            if checked_validation and self.early_stop_patience > 0:
                if improved:
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                epoch_record["early_stop_checks_without_improvement"] = int(
                    checks_without_improvement
                )

            self.metric_history.append(epoch_record)
            self._write_metric_history()

            if (
                checked_validation
                and self.early_stop_patience > 0
                and checks_without_improvement >= self.early_stop_patience
            ):
                best_epoch = (
                    self.best_val_result.get("epoch")
                    if self.best_val_result is not None else "NA"
                )
                self.early_stop_triggered = True
                self.early_stop_reason = (
                    f"MRR did not improve by at least {self.early_stop_min_delta:g} "
                    f"for {checks_without_improvement} validation checks; "
                    f"best_epoch={best_epoch}, best_mrr={self.best_val_mrr:.4f}"
                )
                print(f"Early stopping: {self.early_stop_reason}")
                break

        self._save_checkpoint("final")
        self._write_metric_history()

    
    
    

    def _train_on_frame_pair(
        self, seq_idx: int, fp: FramePairTopology,
    ) -> Optional[torch.Tensor]:
        
        result = self._get_frame_pair_embeddings(seq_idx, fp)
        if result is None:
            return None

        if len(result) == 2:
            
            anchor_emb, cand_emb = result
            valid_t = np.ones(fp.M, dtype=bool)
            valid_tp1 = np.ones(fp.N, dtype=bool)
        else:
            
            anchor_emb, cand_emb, valid_t, valid_tp1 = result

        if anchor_emb.shape[0] == 0 or cand_emb.shape[0] == 0:
            return None

        
        pos_mask = torch.from_numpy(
            fp.true_link_mask[valid_t][:, valid_tp1],
        ).to(self.device)
        cross_frame_neg = fp.topology_neg_mask if 2 in self.enabled_layers else fp.neg_mask
        neg_mask = torch.from_numpy(
            cross_frame_neg[valid_t][:, valid_tp1],
        ).to(self.device)
        weights = torch.from_numpy(
            fp.pair_weights[valid_t][:, valid_tp1],
        ).to(self.device).float()

        if pos_mask.sum() == 0 and neg_mask.sum() == 0:
            return None

        loss = self.loss_fn(anchor_emb, cand_emb, pos_mask, neg_mask, weights)

        if self.same_frame_cluster_contrast and self.same_frame_cluster_weight > 0:
            same_frame_loss = self._same_frame_cluster_loss(
                anchor_emb, cand_emb, fp, valid_t, valid_tp1,
            )
            if same_frame_loss is not None:
                loss = loss + self.same_frame_cluster_weight * same_frame_loss

        return loss

    def _same_frame_cluster_loss(
        self,
        anchor_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        fp: FramePairTopology,
        valid_t: np.ndarray,
        valid_tp1: np.ndarray,
    ) -> Optional[torch.Tensor]:
        
        losses = []

        neg_t = self._same_side_cluster_neg_mask(
            fp.cluster_ids, fp.cell_ids_t, valid_t, side="left",
        )
        if neg_t is not None and neg_t.any():
            neg_t_t = torch.from_numpy(neg_t).to(self.device)
            pos_t_t = torch.zeros_like(neg_t_t, dtype=torch.bool)
            weights_t = torch.ones_like(neg_t_t, dtype=anchor_emb.dtype)
            losses.append(self.loss_fn(anchor_emb, anchor_emb, pos_t_t, neg_t_t, weights_t))

        neg_tp1 = self._same_side_cluster_neg_mask(
            fp.cluster_ids, fp.cell_ids_tp1, valid_tp1, side="right",
        )
        if neg_tp1 is not None and neg_tp1.any():
            neg_tp1_t = torch.from_numpy(neg_tp1).to(self.device)
            pos_tp1_t = torch.zeros_like(neg_tp1_t, dtype=torch.bool)
            weights_tp1 = torch.ones_like(neg_tp1_t, dtype=cand_emb.dtype)
            losses.append(self.loss_fn(cand_emb, cand_emb, pos_tp1_t, neg_tp1_t, weights_tp1))

        if not losses:
            return None
        return torch.stack(losses).mean()

    @staticmethod
    def _same_side_cluster_neg_mask(
        cluster_ids: np.ndarray,
        cell_ids: np.ndarray,
        valid: np.ndarray,
        side: str,
    ) -> Optional[np.ndarray]:
        
        if cluster_ids.size == 0 or not np.any(cluster_ids >= 0):
            return None

        n_cells = len(cell_ids)
        full_mask = np.zeros((n_cells, n_cells), dtype=bool)

        for cid in np.unique(cluster_ids[cluster_ids >= 0]):
            rows, cols = np.where(cluster_ids == cid)
            members = np.unique(rows if side == "left" else cols)
            if len(members) < 2:
                continue
            full_mask[np.ix_(members, members)] = True

        if not full_mask.any():
            return None

        
        
        np.fill_diagonal(full_mask, False)
        different_track = cell_ids[:, np.newaxis] != cell_ids[np.newaxis, :]
        full_mask &= different_track

        filtered = full_mask[valid][:, valid]
        return filtered if filtered.size > 0 else None

    def _get_frame_pair_embeddings(self, seq_idx: int, fp: FramePairTopology):
        
        if self.frame_pair_loader is not None:
            if hasattr(self.frame_pair_loader, "get_frame_pair_embeddings"):
                return self.frame_pair_loader.get_frame_pair_embeddings(
                    self.model, fp, seq_idx, self.device, training=True,
                )
            result = self.frame_pair_loader.load_frame_pair(fp, seq_idx)
            if result is None:
                return None
            images_t, images_tp1, valid_t, valid_tp1 = result

            
            all_images = torch.cat([images_t, images_tp1], dim=0).to(self.device)
            all_emb = self.model(all_images)

            anchor_emb = all_emb[:len(images_t)]
            cand_emb = all_emb[len(images_t):]
            return anchor_emb, cand_emb, valid_t, valid_tp1

        return None, None

    
    
    

    def _validate_knn(self) -> float:
        
        if self.val_dataset is None:
            return 0.0

        self.model.eval()
        all_embeddings = []
        all_labels = []

        
        _ = len(self.val_dataset)  
        loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, num_workers=0,
        )

        with torch.no_grad():
            for batch in loader:
                images, labels = batch
                images = images.to(self.device)
                emb = self.model(images)
                all_embeddings.append(emb.cpu())
                all_labels.append(labels)

        if len(all_embeddings) == 0:
            self.model.train()
            return 0.0

        all_emb = torch.cat(all_embeddings, dim=0)
        all_lbl = torch.cat(all_labels, dim=0).squeeze()
        all_emb = F.normalize(all_emb, p=2, dim=1)

        
        n = min(len(all_emb), 500)
        indices = np.random.choice(len(all_emb), n, replace=False)
        emb_subset = all_emb[indices].to(self.device)
        lbl_subset = all_lbl[indices]

        sim = emb_subset @ emb_subset.T  

        correct = 0
        for i in range(n):
            sim_i = sim[i].clone()
            sim_i[i] = -float("inf")
            nn_idx = sim_i.argmax().item()
            if lbl_subset[i] == lbl_subset[nn_idx]:
                correct += 1

        self.model.train()
        return correct / n

    
    
    

    def _collect_frame_pairs(self) -> List[Tuple[int, int, FramePairTopology]]:
        
        pairs = []
        for seq_idx, topo in enumerate(self.train_topologies):
            for fp_idx, fp in enumerate(topo.frame_pairs):
                pairs.append((seq_idx, fp_idx, fp))
        return pairs

    def _current_alpha(self) -> float:
        return self.topology_computer._curriculum_alpha(
            self.current_epoch, self.enabled_layers,
        )

    def _save_checkpoint(self, tag: str, mrr: float = None):
        if mrr is None:
            mrr = self.best_val_mrr
        fname = f"model_{tag}_e{self.current_epoch:04d}_mrr{mrr:.4f}.pth"
        path = os.path.join(self.model_folder, fname)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_mrr": self.best_val_mrr,
            "mrr": mrr,
        }, path)
        
        if tag == "best":
            best_path = os.path.join(self.model_folder, "model_best.pth")
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "best_val_mrr": self.best_val_mrr,
            }, best_path)
        print(f"  Saved: {fname}")

    def _write_metric_history(self):
        
        exp_dir = os.path.dirname(self.model_folder)
        if not exp_dir or exp_dir == self.model_folder:
            exp_dir = self.model_folder
        os.makedirs(exp_dir, exist_ok=True)

        json_path = os.path.join(exp_dir, "training_metrics.json")
        with open(json_path, "w") as f:
            json.dump({
                "best_val_mrr": float(self.best_val_mrr),
                "best_val_result": self.best_val_result,
                "early_stop_triggered": bool(self.early_stop_triggered),
                "early_stop_reason": self.early_stop_reason,
                "history": self.metric_history,
            }, f, indent=2)

        if not self.metric_history:
            return

        csv_path = os.path.join(exp_dir, "training_metrics.csv")
        fieldnames = sorted({k for row in self.metric_history for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metric_history)

    def save_model_params(self, path: str, **extra_params):
        
        params = dict(extra_params)
        params["model_state_dict"] = self.model.state_dict()
        torch.save(params, path)
        print(f"  Saved all_params to: {path}")
