"""
VoiceShield SOTA Neural Audio Deepfake & Spoofing Detector.
Architecture:
  1. Pretrained Speech Representation Backbone (Wav2Vec 2.0 / WavLM / Sinc-CNN)
  2. Multi-Head Self-Attention Temporal Pooling
  3. Dense Non-Linear Projection (768 -> 256) with LayerNorm, Dropout, & GELU
  4. Calibrated Binary Spoof Logit Classification Head
"""

import math
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Multi-Head Attention Temporal Pooling
# =============================================================================
class MultiHeadAttentionPooling(nn.Module):
    """
    Learned multi-head attention pooling over frame-level feature sequences.
    Transforms sequence [Batch, Frames, Dim] -> [Batch, Dim].
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        q = self.query_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        k = self.key_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T, d]
        v = self.value_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.matmul(attn_weights, v)  # [B, H, T, d]
        pooled = pooled.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]

        # Mean aggregate across temporal dimension
        pooled_out = self.out_proj(pooled.mean(dim=1))  # [B, D]
        return pooled_out


# =============================================================================
# 2. Lightweight Multi-Scale Temporal CNN Backbone (Offline / Fast Fallback)
# =============================================================================
class LightweightSpeechBackbone(nn.Module):
    """
    High-speed raw waveform encoder with multi-scale 1D convolutions & residual blocks.
    Outputs temporal frame sequence of dimension 768.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, embed_dim, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        h = F.gelu(self.bn4(self.conv4(h)))
        return h.transpose(1, 2)  # [B, T_frames, 768]


# =============================================================================
# 3. End-to-End Neural VoiceShield Architecture
# =============================================================================
class VoiceShieldNeuralDetector(nn.Module):
    """
    End-to-End Neural Classifier for Audio Deepfakes & Synthetic Voice Spoofing.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/wav2vec2-base",
        freeze_backbone_epochs: int = 3,
        dropout: float = 0.30,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.embed_dim = 768

        # 1. Backbone Initialization with graceful fallback
        self.use_hf_backbone = False
        if backbone_name == "lightweight":
            self.backbone = LightweightSpeechBackbone(embed_dim=self.embed_dim)
            self.use_hf_backbone = False
            print("[*] Initialized native high-speed LightweightSpeechBackbone.")
        else:
            try:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(backbone_name)
                self.use_hf_backbone = True
                print(f"[*] Loaded pretrained foundation speech backbone: {backbone_name}")
            except Exception as e:
                print(f"[*] Pretrained HF model unavailable ({e}). Using native high-speed acoustic backbone.")
                self.backbone = LightweightSpeechBackbone(embed_dim=self.embed_dim)
                self.use_hf_backbone = False

        # 2. Multi-Head Temporal Attention Pooling
        self.attn_pool = MultiHeadAttentionPooling(embed_dim=self.embed_dim, num_heads=4)

        # 3. Downstream Non-Linear Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, 1),  # Output binary logit
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def freeze_backbone(self) -> None:
        """Freezes backbone weights to warmup downstream attention and classifier heads."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes backbone for full end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Raw waveform tensor of shape [B, T] or [B, 1, T] in float32.
        Returns:
            Logits of shape [B].
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        x = x.to(self.device)

        if self.use_hf_backbone:
            # Hugging Face Feature Extraction
            outputs = self.backbone(x)
            hidden_states = outputs.last_hidden_state  # [B, T_frames, 768]
        else:
            hidden_states = self.backbone(x)  # [B, T_frames, 768]

        # Temporal Attention Pooling
        pooled = self.attn_pool(hidden_states)  # [B, 768]

        # Classification Logit
        logits = self.head(pooled).squeeze(-1)  # [B]
        return logits

    @torch.inference_mode()
    def forward_waveform(self, audio: Union[torch.Tensor, Any]) -> float:
        """
        Inference entrypoint for a single 1D audio waveform array.
        Returns calibrated spoof probability in [0.0, 1.0].
        """
        res = self.predict_waveform(audio)
        return res["spoof_probability"]

    @torch.inference_mode()
    def predict_waveform(self, waveform: Union[torch.Tensor, Any], device: Optional[str] = None) -> Dict[str, Any]:
        """
        Inference entrypoint returning calibrated probabilities and raw logit.
        """
        self.eval()
        dev = torch.device(device) if device else self.device
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        waveform = waveform.to(dev)
        logit_t = self.forward(waveform)
        logit_val = float(logit_t.item() if logit_t.numel() == 1 else logit_t[0].item())
        spoof_p = float(torch.sigmoid(torch.tensor(logit_val)).item())
        human_p = float(1.0 - spoof_p)

        return {
            "spoof_probability": round(spoof_p, 4),
            "human_probability": round(human_p, 4),
            "raw_logit": round(logit_val, 4),
        }

    def save_checkpoint(
        self,
        filepath: str = "models/voiceshield_neural_best.pt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves weights and metadata dictionary."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "model_state_dict": self.state_dict(),
            "backbone_name": self.backbone_name,
            "use_hf_backbone": self.use_hf_backbone,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)
        print(f" • Saved neural checkpoint to: {filepath}")

    def load_checkpoint(self, filepath: str = "models/voiceshield_neural_best.pt") -> Dict[str, Any]:
        """Loads weights and metadata dictionary."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        self.load_state_dict(state, strict=False)
        self.eval()
        return checkpoint.get("metadata", {})


# Alias for explicit naming compatibility
VoiceShieldNeuralClassifier = VoiceShieldNeuralDetector
