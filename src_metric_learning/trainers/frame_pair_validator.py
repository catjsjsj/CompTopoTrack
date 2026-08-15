

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from ..graph_topology.data_structures import SequenceTopology, FramePairTopology


class FramePairValidator:
    

    def __init__(self, val_loader, val_topologies: List[SequenceTopology],
                 device: torch.device, max_pairs_per_sequence: int = 0):
        self.loader = val_loader
        self.topologies = val_topologies
        self.device = device
        self.max_pairs_per_sequence = int(max_pairs_per_sequence or 0)
        if self.max_pairs_per_sequence < 0:
            raise ValueError("max_pairs_per_sequence must be non-negative")

        
        self._frame_pairs: List[Tuple[int, FramePairTopology]] = []
        for seq_idx, topo in enumerate(val_topologies):
            frame_pairs = topo.frame_pairs
            if 0 < self.max_pairs_per_sequence < len(frame_pairs):
                indices = np.linspace(
                    0, len(frame_pairs) - 1,
                    num=self.max_pairs_per_sequence,
                    dtype=int,
                )
                frame_pairs = [frame_pairs[index] for index in np.unique(indices)]
            for fp in frame_pairs:
                self._frame_pairs.append((seq_idx, fp))

    
    
    

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> dict:
        
        model.eval()

        all_ranks = []
        n_skipped = 0

        for seq_idx, fp in self._frame_pairs:
            ranks = self._evaluate_frame_pair(model, seq_idx, fp)
            if ranks is None:
                n_skipped += 1
                continue
            all_ranks.extend(ranks)

        model.train()

        if not all_ranks:
            return {
                'mrr': 0.0, 'precision_at_1': 0.0,
                'n_anchors': 0, 'n_skipped': n_skipped, 'mean_rank': float('inf')
            }

        ranks_arr = np.array(all_ranks, dtype=np.float64)
        return {
            'mrr': float((1.0 / ranks_arr).mean()),
            'precision_at_1': float((ranks_arr == 1).mean()),
            'n_anchors': len(ranks_arr),
            'n_skipped': n_skipped,
            'mean_rank': float(ranks_arr.mean()),
            'evaluated_frame_pairs': len(self._frame_pairs),
        }

    
    
    

    def _evaluate_frame_pair(
        self, model, seq_idx: int, fp: FramePairTopology
    ) -> Optional[List[int]]:
        
        
        
        if hasattr(self.loader, "get_frame_pair_embeddings"):
            result = self.loader.get_frame_pair_embeddings(
                model, fp, seq_idx, self.device, training=False,
            )
            if result is None:
                return None
            anchor_emb, cand_emb, valid_t, valid_tp1 = result
            M_val = anchor_emb.shape[0]
            N_val = cand_emb.shape[0]
        else:
            result = self.loader.load_frame_pair(fp, seq_idx)
            if result is None:
                return None

            images_t, images_tp1, valid_t, valid_tp1 = result
            M_val = images_t.shape[0]
            N_val = images_tp1.shape[0]
            if M_val == 0 or N_val == 0:
                return None

            all_images = torch.cat([images_t, images_tp1], dim=0).to(self.device)
            all_emb = model(all_images)
            anchor_emb = all_emb[:M_val]
            cand_emb = all_emb[M_val:]

        if M_val == 0 or N_val == 0:
            return None

        
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
        cand_emb = F.normalize(cand_emb, p=2, dim=1)
        sim = anchor_emb @ cand_emb.T  

        
        pos_mask = fp.true_link_mask[valid_t][:, valid_tp1]  
        adj_mask = fp.adj_matrix[valid_t][:, valid_tp1]      

        
        pos_mask_t = torch.from_numpy(pos_mask).to(self.device)
        adj_mask_t = torch.from_numpy(adj_mask).to(self.device)

        ranks = []
        for i in range(M_val):
            
            cand_mask = adj_mask_t[i]
            if not cand_mask.any():
                continue  

            
            pos_cols = torch.where(pos_mask_t[i])[0]
            if len(pos_cols) == 0:
                continue  

            pos_col = pos_cols[0]
            pos_sim = sim[i, pos_col]

            
            candidate_sims = sim[i][cand_mask]
            
            candidate_filtered = candidate_sims.clone()
            
            pos_in_candidates = (torch.where(cand_mask)[0] == pos_col).nonzero()
            if len(pos_in_candidates) > 0:
                pos_idx_in_cands = pos_in_candidates[0].item()
                candidate_filtered[pos_idx_in_cands] = -float('inf')

            
            n_better = (candidate_filtered > pos_sim).sum().item()
            rank = n_better + 1
            ranks.append(rank)

        return ranks if ranks else None


def create_val_validator(val_ds, val_topologies, device):
    
    from ..Data.frame_pair_loader import FramePairImageLoader
    val_loader = FramePairImageLoader(val_ds)
    return FramePairValidator(val_loader, val_topologies, device)
