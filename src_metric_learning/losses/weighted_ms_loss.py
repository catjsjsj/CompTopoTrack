

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedMultiSimilarityLoss(nn.Module):
    

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        device = anchor_embeddings.device
        M, D = anchor_embeddings.shape
        N = candidate_embeddings.shape[0]

        
        anchor_emb = F.normalize(anchor_embeddings, p=2, dim=1)
        cand_emb = F.normalize(candidate_embeddings, p=2, dim=1)

        
        sim_matrix = anchor_emb @ cand_emb.T  

        
        has_explicit_weights = weights is not None
        if not has_explicit_weights:
            log_weights = torch.zeros((M, N), device=device, dtype=sim_matrix.dtype)
            zero_weight_mask = torch.zeros((M, N), device=device, dtype=torch.bool)
        else:
            
            w_tensor = weights.to(device=device, dtype=sim_matrix.dtype)
            
            w_clamped = w_tensor.clamp(min=1e-12)
            log_weights = torch.log(w_clamped)
            
            zero_weight_mask = (w_tensor < 1e-12)

        
        
        
        
        pos_exp = self.alpha * (self.base - sim_matrix) + log_weights

        
        pos_exp = pos_exp.masked_fill(~pos_mask, float("-inf"))
        
        if has_explicit_weights:
            pos_exp = pos_exp.masked_fill(zero_weight_mask & pos_mask, float("-inf"))

        
        zeros = torch.zeros(M, 1, device=device, dtype=sim_matrix.dtype)
        pos_for_lse = torch.cat([pos_exp, zeros], dim=1)  
        pos_loss = (1.0 / self.alpha) * torch.logsumexp(pos_for_lse, dim=1)  

        
        has_pos = pos_mask.any(dim=1)
        pos_loss = pos_loss.masked_fill(~has_pos, 0.0)

        
        
        neg_exp = self.beta * (sim_matrix - self.base) + log_weights

        
        neg_exp = neg_exp.masked_fill(~neg_mask, float("-inf"))
        
        if has_explicit_weights:
            neg_exp = neg_exp.masked_fill(zero_weight_mask & neg_mask, float("-inf"))

        neg_for_lse = torch.cat([neg_exp, zeros], dim=1)  
        neg_loss = (1.0 / self.beta) * torch.logsumexp(neg_for_lse, dim=1)  

        
        has_neg = neg_mask.any(dim=1)
        neg_loss = neg_loss.masked_fill(~has_neg, 0.0)

        
        
        valid_anchors = has_pos | has_neg
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss_per_anchor = pos_loss + neg_loss  
        loss = total_loss_per_anchor[valid_anchors].mean()

        return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, base={self.base}"
