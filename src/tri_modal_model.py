"""
VoiceShield Tri-Modal Architecture: Unified Acoustic + Physics + NLP Intent Defense Engine.
Fuses three independent feature modalities:
  - Branch A: Self-Supervised Foundation Acoustic Transformer (256-dim embedding e_a)
  - Branch B: Higher-Order Bispectral Physics & Phase Coupling (64-dim embedding e_p)
  - Branch C: Real-Time NLP Intent & Voice Phishing Intelligence (64-dim embedding e_nlp)
Through a Gated Multimodal Unit (GMU) with adaptive quality-aware attention weighting.
"""

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.calibration import calibrate_risk
from src.neural_model import LightweightSpeechBackbone, MultiHeadAttentionPooling
from src.nlp_intent import (
    NLP_EMBED_DIM,
    NLP_RAW_FEATURE_DIM,
    NLPIntentProjector,
    score_text_intent_heuristics,
)
from src.physics_dsp import (
    PHYSICS_EMBED_DIM,
    PHYSICS_RAW_DIM,
    PhysicsEmbeddingProjector,
    extract_raw_physics_vector,
)

ACOUSTIC_EMBED_DIM: int = 256


class GatedMultimodalUnit(nn.Module):
    """
    Dynamically weights Acoustic, Physics, and NLP intent modalities using input gating.
    """

    def __init__(self, embed_dim: int = ACOUSTIC_EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Project 64-dim physics and NLP embeddings to matched 256-dim space
        self.proj_physics = nn.Linear(PHYSICS_EMBED_DIM, embed_dim)
        self.proj_nlp = nn.Linear(NLP_EMBED_DIM, embed_dim)

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        e_a: torch.Tensor,
        e_p: torch.Tensor,
        e_nlp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # e_a: [B, 256], e_p: [B, 64], e_nlp: [B, 64]
        h_p = F.gelu(self.proj_physics(e_p))
        h_nlp = F.gelu(self.proj_nlp(e_nlp))

        concat_all = torch.cat([e_a, h_p, h_nlp], dim=-1)  # [B, 768]
        gate_logits = self.gate_net(concat_all)  # [B, 3]
        gates = F.softmax(gate_logits, dim=-1)  # [B, 3]

        g_a = gates[:, 0:1]
        g_p = gates[:, 1:2]
        g_nlp = gates[:, 2:3]

        fused = g_a * e_a + g_p * h_p + g_nlp * h_nlp  # [B, 256]
        return fused, gates


class VoiceShieldTriModalEngine(nn.Module):
    """
    Enterprise Tri-Modal Deepfake & Voice Fraud Defense Architecture.
    """

    def __init__(
        self,
        backbone_name: str = "lightweight",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Branch A: Acoustic Backbone & Pooling
        if backbone_name == "lightweight":
            self.acoustic_backbone = LightweightSpeechBackbone(embed_dim=768)
            self.use_hf = False
        else:
            try:
                from transformers import AutoModel
                self.acoustic_backbone = AutoModel.from_pretrained(backbone_name)
                self.use_hf = True
            except Exception:
                self.acoustic_backbone = LightweightSpeechBackbone(embed_dim=768)
                self.use_hf = False

        self.acoustic_pool = MultiHeadAttentionPooling(embed_dim=768, num_heads=4)
        self.acoustic_projector = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, ACOUSTIC_EMBED_DIM),
            nn.GELU(),
            nn.LayerNorm(ACOUSTIC_EMBED_DIM),
        )

        # 2. Branch B: Higher-Order DSP Physics
        self.physics_projector = PhysicsEmbeddingProjector(
            in_dim=PHYSICS_RAW_DIM, embed_dim=PHYSICS_EMBED_DIM
        )

        # 3. Branch C: Real-Time NLP Intent
        self.nlp_projector = NLPIntentProjector(
            in_dim=NLP_RAW_FEATURE_DIM, embed_dim=NLP_EMBED_DIM
        )

        # 4. Gated Multimodal Unit (GMU)
        self.gmu = GatedMultimodalUnit(embed_dim=ACOUSTIC_EMBED_DIM)

        # 5. Output Heads
        self.fused_classifier = nn.Sequential(
            nn.Linear(ACOUSTIC_EMBED_DIM, 128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 1),
        )

        # Auxiliary Heads
        self.aux_acoustic_head = nn.Linear(ACOUSTIC_EMBED_DIM, 1)
        self.aux_physics_head = nn.Linear(PHYSICS_EMBED_DIM, 1)
        self.aux_nlp_head = nn.Linear(NLP_EMBED_DIM, 1)

        self.to(self.device)

    def forward(
        self,
        waveform: torch.Tensor,
        physics_vector: Optional[torch.Tensor] = None,
        nlp_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # waveform: [B, 48000]
        B = waveform.shape[0]

        if physics_vector is None:
            physics_vector = torch.zeros((B, PHYSICS_RAW_DIM), device=waveform.device)
        if nlp_vector is None:
            nlp_vector = torch.zeros((B, NLP_RAW_FEATURE_DIM), device=waveform.device)

        # Branch A: Acoustic
        if self.use_hf:
            h_seq = self.acoustic_backbone(waveform).last_hidden_state
        else:
            h_seq = self.acoustic_backbone(waveform)

        pooled = self.acoustic_pool(h_seq)
        e_a = self.acoustic_projector(pooled)  # [B, 256]

        # Branch B: Physics
        e_p = self.physics_projector(physics_vector)  # [B, 64]

        # Branch C: NLP
        e_nlp = self.nlp_projector(nlp_vector)  # [B, 64]

        # Fusion
        e_fused, gates = self.gmu(e_a, e_p, e_nlp)

        # Logits
        fused_logit = self.fused_classifier(e_fused).squeeze(-1)
        aux_a_logit = self.aux_acoustic_head(e_a).squeeze(-1)
        aux_p_logit = self.aux_physics_head(e_p).squeeze(-1)
        aux_nlp_logit = self.aux_nlp_head(e_nlp).squeeze(-1)

        return {
            "fused_logit": fused_logit,
            "aux_acoustic_logit": aux_a_logit,
            "aux_physics_logit": aux_p_logit,
            "aux_nlp_logit": aux_nlp_logit,
            "fused_embedding": e_fused,
            "acoustic_embedding": e_a,
            "modality_gates": gates,
        }

    @torch.inference_mode()
    def predict_tri_modal(
        self,
        audio: np.ndarray,
        transcription: str = "",
        sr: int = 16000,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end tri-modal inference across acoustic, physics, and NLP intent.
        """
        self.eval()

        # 1. Prepare raw audio tensor (3.0s = 48,000 samples)
        if len(audio) < 48000:
            pad = np.pad(audio, (0, 48000 - len(audio)))
            wav_tensor = torch.from_numpy(pad).unsqueeze(0).float().to(self.device)
        else:
            wav_tensor = torch.from_numpy(audio[:48000]).unsqueeze(0).float().to(self.device)

        # 2. Extract physics vector (32-dim)
        phys_arr = extract_raw_physics_vector(audio, sr=sr)
        phys_tensor = torch.from_numpy(phys_arr).unsqueeze(0).float().to(self.device)

        # 3. Extract NLP Intent vector (16-dim)
        nlp_res = score_text_intent_heuristics(transcription)
        nlp_arr = nlp_res["raw_feature_vector"]
        nlp_tensor = torch.from_numpy(nlp_arr).unsqueeze(0).float().to(self.device)

        # 4. Forward Pass
        out = self.forward(wav_tensor, phys_tensor, nlp_tensor)

        fused_prob = float(torch.sigmoid(out["fused_logit"][0]).item())
        acoustic_prob = float(torch.sigmoid(out["aux_acoustic_logit"][0]).item())
        physics_prob = float(torch.sigmoid(out["aux_physics_logit"][0]).item())
        nlp_prob = max(nlp_res["fraud_intent_score"], float(torch.sigmoid(out["aux_nlp_logit"][0]).item()))

        # Combined Threat Score (0 - 100)
        combined_score = int(round(fused_prob * 100))
        acoustic_score = int(round(acoustic_prob * 100))
        physics_score = int(round(physics_prob * 100))
        nlp_score = int(round(nlp_prob * 100))

        # Risk Band Mapping
        calib = calibrate_risk(fused_prob, fused_prob)

        return {
            "combined_risk_score": combined_score,
            "risk_band": calib["risk_band_label"],
            "risk_band_key": calib["risk_band"],
            "badge_class": calib["badge_class"],
            "fused_spoof_prob": round(fused_prob, 4),
            "breakdown": {
                "neural_acoustic_score": acoustic_score,
                "dsp_phase_anomaly": physics_score,
                "nlp_intent_threat": nlp_score,
            },
            "nlp_threat_categories": nlp_res["threat_categories"],
            "transcription": transcription,
            "modality_weights": {
                "acoustic_weight": round(float(out["modality_gates"][0, 0].item()), 3),
                "physics_weight": round(float(out["modality_gates"][0, 1].item()), 3),
                "nlp_weight": round(float(out["modality_gates"][0, 2].item()), 3),
            },
        }
