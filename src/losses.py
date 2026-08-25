"""
VoiceShield Neural Step 3: Numerically Stable Binary Focal Loss.
Penalizes hard edge-cases and suppresses easy examples for robust audio deepfake detection:
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """
    Numerically stable Binary Focal Loss operating directly on raw, unnormalized logits.
    
    Args:
        gamma: Focusing parameter (default: 2.0) that decreases loss contribution from easy examples.
        alpha: Class weighting factor (default: 0.25) to balance false positive and negative rates.
        reduction: Specifies reduction to apply to output: 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            logits: Predicted raw logits of shape (B,) or (B, 1).
            targets: Binary ground truth labels (0.0 or 1.0) of identical or broadcastable shape.

        Returns:
            Computed Focal Loss scalar or tensor.
        """
        # Flatten tensors to 1D
        logits = logits.view(-1).float()
        targets = targets.view(-1).float()

        # Numerically stable Binary Cross Entropy with Logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability calculation p = sigmoid(logits)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        # p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)

        # alpha_t: class weighting
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        # Focal loss per sample
        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")


class TriModalMultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for VoiceShield Tri-Modal Architecture.
    Combines:
      1. Binary Focal Loss on Fused Prediction Head (gamma=2.0, alpha=0.25)
      2. Auxiliary Focal Losses on Acoustic & Physics heads
      3. Contrastive Cosine Margin Loss on latent embeddings
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        aux_weight: float = 0.20,
        contrastive_weight: float = 0.15,
        margin: float = 0.50,
    ) -> None:
        super().__init__()
        self.focal_loss = BinaryFocalLossWithLogits(gamma=focal_gamma, alpha=focal_alpha)
        self.aux_weight = aux_weight
        self.contrastive_weight = contrastive_weight
        self.margin = margin

    def forward(
        self,
        model_outputs: dict,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # targets: [B] or [B, 1]
        targets_flat = targets.view(-1).float()

        # 1. Primary Fused Focal Loss
        loss_fused = self.focal_loss(model_outputs["fused_logit"], targets_flat)

        # 2. Auxiliary Losses
        loss_aux_a = self.focal_loss(model_outputs["aux_acoustic_logit"], targets_flat)
        loss_aux_p = self.focal_loss(model_outputs["aux_physics_logit"], targets_flat)

        total_loss = loss_fused + self.aux_weight * (loss_aux_a + loss_aux_p)

        # 3. Contrastive Margin Loss on Fused Embeddings (if batch size > 1)
        if "fused_embedding" in model_outputs and model_outputs["fused_embedding"].shape[0] > 1:
            emb = model_outputs["fused_embedding"]
            emb_norm = F.normalize(emb, p=2, dim=-1)
            sim_matrix = torch.matmul(emb_norm, emb_norm.T)  # [B, B]

            # Pairwise target match matrix (1 if same class, 0 if different)
            target_col = targets_flat.unsqueeze(1)
            match_matrix = (target_col == target_col.T).float()

            # Pull same-class together, push different-class beyond margin
            pos_loss = (1.0 - sim_matrix) * match_matrix
            neg_loss = torch.clamp(sim_matrix - self.margin, min=0.0) * (1.0 - match_matrix)
            contrastive_loss = (pos_loss + neg_loss).mean()

            total_loss = total_loss + self.contrastive_weight * contrastive_loss

        return total_loss

