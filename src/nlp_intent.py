"""
VoiceShield Tri-Modal Branch C: Real-Time NLP Intent & Voice Phishing Intelligence.
Analyzes transcribed conversational speech for social engineering markers:
  1. Urgency & Coercion (e.g. account suspension, immediate payment threats)
  2. Credential & OTP Solicitation (e.g. asking for 2FA tokens, PINs, passwords)
  3. Fraudulent Financial Routing (e.g. unauthorized wire transfers, crypto demands)
  4. Authority Impersonation (e.g. IT support, bank manager, IRS/law enforcement)
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NLP_RAW_FEATURE_DIM: int = 16
NLP_EMBED_DIM: int = 64

# Social Engineering & Phishing Intent Lexicon Rules
URGENCY_KEYWORDS = [
    "immediately", "urgent", "suspend", "suspended", "arrest", "warrant",
    "expire", "expired", "penalty", "emergency", "within 10 minutes", "right now",
    "lawsuit", "legal action", "frozen", "deactivated", "compromised"
]

CREDENTIAL_KEYWORDS = [
    "password", "passcode", "otp", "one time password", "pin", "verification code",
    "security code", "2fa", "two factor", "cvv", "card number", "social security",
    "ssn", "login credentials", "verify your account", "confirm your identity"
]

FINANCIAL_ROUTING_KEYWORDS = [
    "wire transfer", "transfer funds", "gift card", "bitcoin", "crypto",
    "western union", "bank routing", "send money", "unauthorized charge",
    "refund", "overpayment", "escrow", "direct deposit"
]

AUTHORITY_IMPERSONATION_KEYWORDS = [
    "internal revenue service", "irs", "fbi", "police department", "it support",
    "helpdesk", "fraud department", "security team", "bank manager", "compliance officer",
    "executive office", "chief executive", "system administrator"
]


def score_text_intent_heuristics(text: str) -> Dict[str, Any]:
    """
    Computes rule-based and density-based social engineering risk metrics from raw transcribed text.
    """
    if not text or len(text.strip()) == 0:
        return {
            "fraud_intent_score": 0.0,
            "threat_categories": [],
            "urgency_score": 0.0,
            "credential_solicitation_score": 0.0,
            "financial_routing_score": 0.0,
            "authority_impersonation_score": 0.0,
            "raw_feature_vector": np.zeros(NLP_RAW_FEATURE_DIM, dtype=np.float32),
        }

    clean_text = text.lower()
    words = re.findall(r"\w+", clean_text)
    total_words = max(1, len(words))

    # Keyword matching densities
    urgency_hits = sum(1 for kw in URGENCY_KEYWORDS if kw in clean_text)
    cred_hits = sum(1 for kw in CREDENTIAL_KEYWORDS if kw in clean_text)
    fin_hits = sum(1 for kw in FINANCIAL_ROUTING_KEYWORDS if kw in clean_text)
    auth_hits = sum(1 for kw in AUTHORITY_IMPERSONATION_KEYWORDS if kw in clean_text)

    urgency_score = float(np.clip(urgency_hits * 0.45, 0.0, 1.0))
    cred_score = float(np.clip(cred_hits * 0.55, 0.0, 1.0))
    fin_score = float(np.clip(fin_hits * 0.45, 0.0, 1.0))
    auth_score = float(np.clip(auth_hits * 0.45, 0.0, 1.0))

    # Combined Social Engineering Risk Score with synergistic boost
    base_risk = 0.40 * cred_score + 0.35 * urgency_score + 0.15 * fin_score + 0.10 * auth_score
    active_threat_count = sum(1 for s in [urgency_score, cred_score, fin_score, auth_score] if s > 0.25)
    synergy_boost = 0.15 if active_threat_count >= 2 else 0.0
    combined_risk = float(np.clip(base_risk + synergy_boost, 0.0, 1.0))

    threat_categories = []
    if cred_score > 0.3:
        threat_categories.append("Credential / OTP Harvesting")
    if urgency_score > 0.3:
        threat_categories.append("Urgency & Coercion Pressure")
    if fin_score > 0.3:
        threat_categories.append("Fraudulent Financial Routing")
    if auth_score > 0.3:
        threat_categories.append("Authority / Executive Impersonation")

    # Build 16-dim raw feature vector
    raw_vector = np.array([
        combined_risk,
        urgency_score,
        cred_score,
        fin_score,
        auth_score,
        min(1.0, total_words / 50.0),
        min(1.0, urgency_hits / 5.0),
        min(1.0, cred_hits / 5.0),
        min(1.0, fin_hits / 5.0),
        min(1.0, auth_hits / 5.0),
        1.0 if ("click" in clean_text or "link" in clean_text) else 0.0,
        1.0 if ("do not hang up" in clean_text or "stay on the line" in clean_text) else 0.0,
        0.0, 0.0, 0.0, 0.0
    ], dtype=np.float32)

    return {
        "fraud_intent_score": round(combined_risk, 4),
        "threat_categories": threat_categories,
        "urgency_score": round(urgency_score, 4),
        "credential_solicitation_score": round(cred_score, 4),
        "financial_routing_score": round(fin_score, 4),
        "authority_impersonation_score": round(auth_score, 4),
        "raw_feature_vector": raw_vector,
    }


class NLPIntentProjector(nn.Module):
    """
    Neural projection layer mapping 16-dim semantic intent features to 64-dim NLP embedding space.
    """

    def __init__(self, in_dim: int = NLP_RAW_FEATURE_DIM, embed_dim: int = NLP_EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 16)
        return self.net(x)
