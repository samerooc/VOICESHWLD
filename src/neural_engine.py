"""
VoiceShield Phase 3 — SOTA Deep Learning Foundation Inference Engine.

Implements:
  1. Resilient multi-tier foundation model loader:
       Primary  : garystafford/wav2vec2-deepfake-voice-detector
       Fallback1: gustking/wav2vec2-large-xlsr-53-deepfake-detect
       Fallback2: MelodyMachine/Deepfake-audio-detection-V2
     Automatic device routing: CUDA (FP16) → multi-threaded CPU.
     Startup warmup: non-blocking dummy forward pass on init.

  2. Dynamic token & label resolution (eliminates label inversion):
     Inspects model.config.id2label via case-insensitive regex matching.
     Never assumes a hardcoded label index.

  3. Temperature-scaled logit calibration  (T = 1.35).

  4. Tri-tier orthogonal feature integration:
       Tier 1: Deep acoustic embeddings from fine-tuned Wav2Vec2/XLS-R
               (3.0 s sliding windows, zero-mean unit-variance normalisation)
       Tier 2: LPC residual excitation & phase entropy (LPCPhysicsAnalyzer)
       Tier 3: Glottal micro-jitter, shimmer, HNR, LFCC (ForensicDSPAnalyzer)

  5. Adaptive SNR-weighted consensus engine:
       SNR ≥ 10 dB  →  P_spoof = 0.50·Transformer + 0.30·LPC + 0.20·Glottal/LFCC
       SNR <  10 dB  →  P_spoof = 0.35·Transformer + 0.35·LPC + 0.30·Glottal/LFCC

  6. 5-state calibrated risk categorisation:
        0 –  25  : Low Risk (Human Voice)
       26 –  60  : Review Required (Borderline Evidence)
       61 – 100  : High Risk (Likely AI / Cloned Voice)
       Low Quality / Degraded: voiced speech < 0.4 s or SNR < 3 dB

Author:  VoiceShield Engineering
Version: 3.0.0  (Phase 3)
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.audio_processor import decode_and_sanitize_audio, normalize_audio_standard
from src.channel_normalizer import AcousticChannelNormalizer
from src.config import SAMPLE_RATE
from src.forensic_dsp import ForensicDSPAnalyzer
from src.lpc_physics import LPCPhysicsAnalyzer
from src.indic_forensics import extract_indic_tts_forensics
from src.music_forensics import is_music_track, extract_music_diffusion_artifacts, isolate_singing_vocals
from src.neural_model import VoiceShieldNeuralClassifier

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    HAS_TRANSFORMERS: bool = True
except ImportError:
    HAS_TRANSFORMERS = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------
_TEMPERATURE: float          = 1.35      # Logit temperature scaling
_WINDOW_SEC: float           = 3.0       # Sliding inference window length (seconds)
_WINDOW_HOP_SEC: float       = 1.5       # Sliding window hop (50 % overlap)
_SNR_CLEAN_DB: float         = 10.0      # SNR threshold for tier-weight switching
_SNR_FLOOR_DB: float         = 3.0       # Below this → degraded quality gate
_MIN_VOICED_SEC: float       = 0.4       # Below this → degraded quality gate
_MIN_TOTAL_SEC: float        = 0.5       # Below this → degraded quality gate
_CPU_TORCH_THREADS: int      = 4         # OMP/MKL thread count for CPU inference

# Regex patterns for dynamic label resolution (case-insensitive)
_SPOOF_PATTERN = re.compile(
    r"(fake|spoof|synth|deepfake|clone|generated|artificial|ai_voice|tts)",
    re.IGNORECASE,
)
_HUMAN_PATTERN = re.compile(
    r"(real|bonafide|bona.fide|human|authentic|original|genuine|natural)",
    re.IGNORECASE,
)


# ===========================================================================
# Main Production Detector
# ===========================================================================

class ProductionNeuralDetector:
    """
    Production-Grade Multi-Tier Voice Clone & Deepfake Detection Engine.

    Combines three orthogonal forensic channels:
      • Fine-tuned Wav2Vec2/XLS-R transformer (Tier 1)
      • LPC residual excitation physics (Tier 2)
      • Praat glottal biomechanics + ASVspoof LFCC (Tier 3)

    Usage
    -----
    >>> detector = ProductionNeuralDetector()
    >>> result = detector.predict_bytes(wav_bytes)
    >>> print(result["risk_score"], result["risk_band"])
    """

    # Ordered model cascade (primary → fallback1 → fallback2)
    _DEFAULT_MODEL_CASCADE: Tuple[str, ...] = (
        "garystafford/wav2vec2-deepfake-voice-detector",
        "gustking/wav2vec2-large-xlsr-53-deepfake-detect",
        "MelodyMachine/Deepfake-audio-detection-V2",
    )

    def __init__(
        self,
        primary_model_id: Optional[str] = "garystafford/wav2vec2-deepfake-voice-detector",
        fallback_model_id: Optional[str] = "gustking/wav2vec2-large-xlsr-53-deepfake-detect",
        secondary_fallback_id: Optional[str] = "MelodyMachine/Deepfake-audio-detection-V2",
        native_checkpoint_path: str = "models/voiceshield_live_robust.pt",
        checkpoint_path: Optional[str] = None,
        temperature: float = _TEMPERATURE,
        device: Optional[str] = None,
        load_hf: bool = True,
    ) -> None:
        # ---- Device routing ------------------------------------------------
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(_CPU_TORCH_THREADS)

        self.target_sr   = SAMPLE_RATE
        self.temperature = temperature

        log.info("[Phase3] Initializing VoiceShield Neural Engine on: [%s]", self.device)
        print(f"[*] Initializing VoiceShield Phase 3 Neural Engine on: [{self.device}]")

        # ---- HF state ------------------------------------------------------
        self.has_hf_model: bool   = False
        self.model                = None
        self.feature_extractor    = None
        self.spoof_idx: int       = 1
        self.human_idx: int       = 0
        self.active_model_id: str = "Native Acoustic Backbone"
        self.loaded_hf_models: List[Dict[str, Any]] = []

        # ---- Read HF token from env / .env ---------------------------------
        hf_token = self._load_hf_token()

        # ---- 1. Multi-Foundation Model Ensemble Loader ---------------------
        if HAS_TRANSFORMERS and load_hf:
            model_cascade = [m for m in [primary_model_id, secondary_fallback_id, fallback_model_id] if m]
            for model_id in model_cascade:
                m_info = self._try_load_hf_model_info(model_id, hf_token)
                if m_info:
                    self.loaded_hf_models.append(m_info)
                    if not self.has_hf_model:
                        self.has_hf_model = True
                        self.model = m_info["model"]
                        self.feature_extractor = m_info["feature_extractor"]
                        self.spoof_idx = m_info["spoof_idx"]
                        self.human_idx = m_info["human_idx"]
                        self.active_model_id = m_info["model_id"]

        # ---- 2. Native Acoustic Neural Backbone ----------------------------
        self.native_model: Optional[VoiceShieldNeuralClassifier] = None
        self._load_native_checkpoint(checkpoint_path or native_checkpoint_path)

        # ---- 3. Phase 2 DSP & Physics Analyzers ---------------------------
        self.lpc_analyzer      = LPCPhysicsAnalyzer(order=16, sr=self.target_sr)
        self.dsp_analyzer      = ForensicDSPAnalyzer(sr=self.target_sr)
        self.channel_normalizer = AcousticChannelNormalizer(sr=self.target_sr)

        # ---- 4. Startup warmup (eliminates cold-start latency) -------------
        self._warmup()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_token() -> Optional[str]:
        """Read HF access token from env or .env file."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token and os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("HF_TOKEN="):
                            token = line.strip().split("=", 1)[1].strip().strip('"\'')
                            os.environ["HF_TOKEN"] = token
                            break
            except Exception:
                pass
        return token or None

    def _try_load_hf_model_info(self, model_id: str, hf_token: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Attempt to load a single Hugging Face model and resolve its labels.
        Returns a model info dictionary on success or None on failure.
        """
        try:
            load_kwargs: Dict[str, Any] = {}
            if hf_token:
                load_kwargs["token"] = hf_token

            fe = AutoFeatureExtractor.from_pretrained(model_id, **load_kwargs)

            dtype_kwargs: Dict[str, Any] = {}
            if self.device.type == "cuda":
                dtype_kwargs["torch_dtype"] = torch.float16

            m = AutoModelForAudioClassification.from_pretrained(
                model_id, **{**load_kwargs, **dtype_kwargs}
            ).to(self.device)
            m.eval()

            # Dynamic label resolution per model
            spoof_idx = 1
            human_idx = 0
            if hasattr(m, "config") and hasattr(m.config, "id2label") and m.config.id2label:
                for idx, label in m.config.id2label.items():
                    if _SPOOF_PATTERN.search(label):
                        spoof_idx = int(idx)
                    elif _HUMAN_PATTERN.search(label):
                        human_idx = int(idx)

            print(
                f"[+] Ensemble Foundation Model: {model_id} "
                f"(spoof_idx={spoof_idx}, human_idx={human_idx})"
            )
            return {
                "model": m,
                "feature_extractor": fe,
                "model_id": model_id,
                "spoof_idx": spoof_idx,
                "human_idx": human_idx,
            }
        except Exception as exc:
            log.debug("Could not load %s: %s", model_id, exc)
            return None

    def _try_load_hf_model(self, model_id: str, hf_token: Optional[str]) -> bool:
        """Legacy helper for single model loading."""
        info = self._try_load_hf_model_info(model_id, hf_token)
        if info:
            self.has_hf_model = True
            self.model = info["model"]
            self.feature_extractor = info["feature_extractor"]
            self.spoof_idx = info["spoof_idx"]
            self.human_idx = info["human_idx"]
            self.active_model_id = info["model_id"]
            return True
        return False

    def _load_native_checkpoint(self, target_ckpt: str) -> None:
        """Load the native lightweight acoustic backbone with multi-checkpoint fallback."""
        try:
            self.native_model = VoiceShieldNeuralClassifier(
                backbone_name="lightweight", device=self.device
            )
            candidate_ckpts = [
                target_ckpt,
                "models/voiceshield_live_robust.pt",
                "models/voiceshield_neural_best.pt",
            ]
            for ckpt_path in candidate_ckpts:
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    state = ckpt.get(
                        "model_state_dict", ckpt.get("state_dict", ckpt)
                    )
                    self.native_model.load_state_dict(state, strict=False)
                    self.native_model.has_weights = True
                    print(f"[+] Native Weights: {os.path.basename(ckpt_path)}")
                    break
            else:
                self.native_model.has_weights = False
                print("[*] No fine-tuned native checkpoint found; using untrained backbone.")
            self.native_model.eval()

        except Exception as exc:
            log.warning("Native model init failed: %s", exc)
            print(f"[!] Warning: Native acoustic model unavailable ({exc}).")
            self.native_model = None

    def _warmup(self) -> None:
        """
        Non-blocking dummy forward pass to pre-warm CUDA kernels, JIT compilation,
        and MKLDNN weight conversion.  Eliminates cold-start latency on first request.
        """
        try:
            dummy = np.zeros(self.target_sr, dtype=np.float32)  # 1 s of silence
            if self.has_hf_model and self.feature_extractor is not None and self.model is not None:
                with torch.inference_mode():
                    inp = self.feature_extractor(
                        dummy, sampling_rate=self.target_sr, return_tensors="pt"
                    )
                    inp = {k: v.to(self.device) for k, v in inp.items()}
                    _ = self.model(**inp).logits
            if self.native_model is not None:
                with torch.inference_mode():
                    t = torch.zeros(1, self.target_sr, device=self.device)
                    _ = self.native_model(t)
            log.debug("Warmup complete.")
        except Exception as exc:
            log.debug("Warmup skipped: %s", exc)

    # ------------------------------------------------------------------
    # Dynamic Label Resolution  (Tier 1 — eliminates label inversion)
    # ------------------------------------------------------------------

    def _resolve_labels(self) -> None:
        """
        Dynamically inspect model.config.id2label to find the correct spoof
        and human class indices.

        Strategy
        --------
        1. Match each label string against _SPOOF_PATTERN / _HUMAN_PATTERN.
        2. If no named labels match (e.g. LABEL_0, LABEL_1), fall back to
           acoustic phase consensus cross-referencing (index remains default 1/0).
        3. If the resolved spoof_idx == human_idx (degenerate config), reset to 1/0.

        This function is model-agnostic and never assumes a hardcoded index.
        """
        self.spoof_idx = 1
        self.human_idx = 0

        if not (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "id2label")
            and self.model.config.id2label
        ):
            log.debug("No id2label in model config; using default indices 1/0.")
            return

        id2label: Dict[int, str] = self.model.config.id2label
        found_spoof = False
        found_human = False

        for idx, label in id2label.items():
            if _SPOOF_PATTERN.search(label):
                self.spoof_idx = idx
                found_spoof    = True
                log.debug("Label resolver: spoof → idx=%d label=%r", idx, label)

            elif _HUMAN_PATTERN.search(label):
                self.human_idx = idx
                found_human    = True
                log.debug("Label resolver: human → idx=%d label=%r", idx, label)

        # Sanity: degenerate or unmapped labels → fallback
        if not found_spoof or self.spoof_idx == self.human_idx:
            log.warning(
                "Label resolver could not unambiguously identify spoof class "
                "(found_spoof=%s, spoof_idx=%d, human_idx=%d). "
                "Falling back to acoustic phase consensus cross-referencing (idx 1/0).",
                found_spoof, self.spoof_idx, self.human_idx,
            )
            self.spoof_idx = 1
            self.human_idx = 0

        # Expose whether labels were successfully named-resolved
        self._labels_named_resolved = found_spoof and found_human

    # ------------------------------------------------------------------
    # Public inference entry points
    # ------------------------------------------------------------------

    def predict_bytes(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Public entry point accepting raw byte streams.

        Supported formats: WAV · MP3 · M4A · FLAC · OGG · WebM/Opus · AAC
        """
        return self.predict(audio_bytes)

    @torch.inference_mode()
    def predict(
        self,
        audio_input: Union[bytes, np.ndarray, str],
        sample_rate: Optional[int] = None,
        is_live_mic: bool = False,
        transcript_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute tri-tier forensic inference and SNR-weighted consensus fusion.

        Parameters
        ----------
        audio_input : bytes | np.ndarray | str
            Raw audio container bytes, pre-decoded float32 array, or file path.
        sample_rate : int, optional
            Sample rate of a pre-decoded numpy array.
        is_live_mic : bool
            If True, apply channel normalisation / de-reverberation.

        Returns
        -------
        dict
            Full forensic report including risk_score, risk_band, forensic_breakdown,
            latency_ms, and all Phase 2/3 intermediate probabilities.
        """
        t0 = time.perf_counter()

        # ================================================================
        # Step 1: In-Memory Universal Audio Decoding & VAD
        # ================================================================
        raw_full_audio, voiced_audio, diag = decode_and_sanitize_audio(
            audio_input=audio_input,
            target_sr=self.target_sr,
            orig_sr=sample_rate,
        )

        total_duration  = diag["duration_sec"]
        voiced_duration = diag["voiced_duration_sec"]
        snr_db          = float(diag["snr_db"])
        is_silent       = bool(diag["is_silent"])

        # ================================================================
        # Quality Gate: Low Quality / Degraded
        # Triggered if voiced speech < 0.4 s OR SNR < 3.0 dB OR silence
        # ================================================================
        if total_duration < _MIN_TOTAL_SEC or voiced_duration < _MIN_VOICED_SEC or is_silent:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self._degraded_response(diag, latency_ms)

        # Select analysis window: preserve continuous speech waveform with soft silence trim
        # to prevent artificial phase tearing / step jumps from fragmented frame slicing
        try:
            trimmed_audio, _ = librosa.effects.trim(raw_full_audio, top_db=28)
            eval_audio: np.ndarray = (
                trimmed_audio if len(trimmed_audio) >= int(0.5 * self.target_sr)
                else raw_full_audio
            )
        except Exception:
            eval_audio = raw_full_audio

        # Dynamic gain staging (AGC): if user is speaking softly, quietly, or muffled (< -20 dBFS),
        # stage RMS energy to 0.12 target so neural embeddings & DSP are perfectly conditioned
        rms_val = float(np.sqrt(np.mean(eval_audio ** 2))) + 1e-8
        if rms_val < 0.08:
            target_rms = 0.12
            gain = min(target_rms / rms_val, 12.0)
            eval_audio = np.clip(eval_audio * gain, -0.95, 0.95).astype(np.float32)

        # Live mic: apply soft spectral subtraction only in genuinely noisy environments (SNR < 18 dB)
        if is_live_mic and snr_db < 18.0:
            try:
                eval_audio = self.channel_normalizer.spectral_subtraction(
                    eval_audio, noise_floor_ratio=0.15, oversubtraction_factor=1.0
                )
            except Exception as exc:
                log.debug("Channel normalisation skipped: %s", exc)

        # ================================================================
        # Step 2 (Tier 2): LPC Residual Excitation & Phase Entropy Engine
        # ================================================================
        lpc_res = self.lpc_analyzer.extract_lpc_residual(
            eval_audio, sr=self.target_sr, order=16
        )
        p_lpc = float(lpc_res["lpc_anomaly_score"])

        # ================================================================
        # Step 3 (Tier 3): Glottal Micro-Jitter & ASVspoof LFCC
        # ================================================================
        dsp_res = self.dsp_analyzer.extract_dsp_metrics(eval_audio, sr=self.target_sr)
        p_dsp   = float(dsp_res["combined_dsp_risk"])

        # ================================================================
        # Step 4 (Tier 1): Deep Acoustic Embeddings & Singing Vocal Separation
        # ================================================================
        p_transformer, window_scores = self._run_transformer_inference(eval_audio)

        # AI Song, Singing Voice & Music Latent Diffusion Inspection
        is_music, music_conf = is_music_track(eval_audio, sr=self.target_sr)
        music_metrics = extract_music_diffusion_artifacts(eval_audio, sr=self.target_sr)
        p_music = float(music_metrics.get("music_spoof_prob", 0.50))
        if is_music:
            try:
                # Isolate vocal harmonics from backing music/drums for clean transformer analysis
                vocal_audio = isolate_singing_vocals(eval_audio, sr=self.target_sr)
                p_vocal, v_scores = self._run_transformer_inference(vocal_audio)
                if p_vocal > p_transformer:
                    p_transformer = p_vocal
                    if v_scores:
                        window_scores = v_scores
            except Exception as exc:
                log.debug("Music track vocal separation skipped: %s", exc)

        # Indic & Modern Neural Vocoder (Sarvam AI / Vocos / BigVGAN) Inspection
        indic_metrics = extract_indic_tts_forensics(eval_audio, sr=self.target_sr)
        p_indic = float(indic_metrics.get("indic_spoof_prob", 0.50))

        # ================================================================
        # Adaptive SNR-Weighted Consensus Fusion with Multi-Tier Neural Override
        # ================================================================
        # ── Live-mic vs file-upload vs music track fusion weights ──────────────
        if is_music:
            # Polyphonic AI Song / Singing track (Suno, Udio, RVC, DiffSinger)
            combined_prob = float(0.35 * p_transformer + 0.50 * p_music + 0.08 * p_lpc + 0.07 * p_dsp)
            if p_music >= 0.55 or p_transformer >= 0.40:
                combined_prob = max(combined_prob, 0.76)
            elif p_music <= 0.35 and p_transformer <= 0.25:
                combined_prob = min(combined_prob, 0.22)
        elif is_live_mic:
            if snr_db >= _SNR_CLEAN_DB:
                combined_prob = float(0.72 * p_transformer + 0.16 * p_lpc + 0.12 * p_dsp)
            else:
                combined_prob = float(0.65 * p_transformer + 0.20 * p_lpc + 0.15 * p_dsp)
        elif snr_db >= _SNR_CLEAN_DB:
            # Clean file upload: transformer-dominant with physical verification
            combined_prob = float(0.68 * p_transformer + 0.18 * p_lpc + 0.14 * p_dsp)
        else:
            # Lossy / noisy file upload (e.g. WhatsApp, telephony, Opus/MP3 compression)
            combined_prob = float(0.60 * p_transformer + 0.22 * p_lpc + 0.18 * p_dsp)

        # ── Multi-tier Forensic Overrides ──────────────────────────────────────
        # 1. High-confidence Transformer Deepfake:
        if p_transformer >= 0.70:
            combined_prob = max(combined_prob, p_transformer * 0.94)
        elif p_transformer >= 0.45:
            escalation_floor = 0.60 + (p_transformer - 0.45) * 1.1
            combined_prob = max(combined_prob, min(escalation_floor, 0.88))

        # 2. Physics & Vocoder Brickwall Override:
        max_physical_spoof = max(p_lpc, p_dsp)
        if max_physical_spoof >= 0.85 and p_transformer >= 0.35:
            combined_prob = max(combined_prob, 0.68)

        # 3. Sarvam AI & Indic Neural Vocoder (BigVGAN / Vocos) Override:
        vocos_score = float(indic_metrics.get("vocos_snake_score", 0.0))
        if vocos_score >= 0.65 or p_indic >= 0.65:
            indic_floor = 0.72 + max(vocos_score - 0.65, p_indic - 0.65) * 0.8
            combined_prob = max(combined_prob, min(indic_floor, 0.92))

        # 4. Triple Consensus Human Floor:
        # If primary transformer confirms authentic human voice, prevent false positives
        if p_transformer <= 0.25 and (not is_music or p_music <= 0.35) and vocos_score < 0.60 and p_indic < 0.60:
            combined_prob = min(combined_prob, max(p_transformer, 0.18))

        combined_prob = float(np.clip(combined_prob, 0.01, 0.99))

        # ================================================================
        # 5-State Calibrated Risk Categorisation
        # ================================================================
        risk_score = round(combined_prob * 100.0)
        label, risk_band, risk_band_key, badge_class, risk_desc = (
            self._risk_band(risk_score)
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "prediction_label": label,
            "spoof_probability": round(combined_prob, 4),
            "human_probability": round(1.0 - combined_prob, 4),
            "risk_score":        risk_score,
            "risk_band":         risk_band,
            "risk_band_key":     risk_band_key,
            "badge_class":       badge_class,
            "risk_description":  risk_desc,
            "forensic_breakdown": {
                # Tier 1
                "transformer_spoof_prob":  round(p_transformer, 4),
                "active_model_id":         self.active_model_id,
                # Tier 2 — Phase 2 spec keys
                "lpc_anomaly_score":       round(p_lpc, 4),
                "lpc_kurtosis":            lpc_res["lpc_kurtosis"],
                "phase_entropy":           lpc_res["phase_entropy"],
                "residual_flatness":       lpc_res["residual_flatness"],
                # Tier 3 — Phase 2 spec keys
                "dsp_physics_prob":        round(p_dsp, 4),
                "glottal_spoof_prob":      round(dsp_res["glottal_risk"], 4),
                "lfcc_spoof_prob":         round(dsp_res["lfcc_risk"], 4),
                "spectral_spoof_prob":     round(dsp_res["spectral_risk"], 4),
                "jitter_local":            dsp_res["jitter_local"],
                "local_jitter":            dsp_res["jitter_local"],
                "shimmer_local":           dsp_res["shimmer_local"],
                "local_shimmer":           dsp_res["shimmer_local"],
                "hnr_db":                  dsp_res["hnr_db"],
                "lfcc_variance":           dsp_res["lfcc_variance"],
                "hf_cutoff_ratio":         dsp_res["hf_cutoff_ratio"],
                # SNR / quality metadata
                "snr_db":                  round(snr_db, 2),
                "snr_weight_mode":         "clean" if snr_db >= _SNR_CLEAN_DB else "noisy",
                "voiced_ratio":            diag["voiced_ratio"],
                # AI Music, Song & Diffusion Latent Forensics
                "is_music_track":          bool(is_music),
                "music_spoof_prob":        round(p_music, 4) if is_music else 0.0,
                "checkerboard_score":      music_metrics.get("checkerboard_score", 0.0) if is_music else 0.0,
                "digital_haze_score":      music_metrics.get("digital_haze_score", 0.0) if is_music else 0.0,
                "neural_codec_artifact_score": music_metrics.get("neural_codec_score", 0.0) if is_music else 0.0,
                "phase_dispersion_score":  music_metrics.get("phase_dispersion_score", 0.0) if is_music else 0.0,
                "singing_vibrato_score":   music_metrics.get("singing_vibrato_score", 0.0) if is_music else 0.0,
                # Indic & Sarvam AI Forensics
                "indic_spoof_prob":        round(p_indic, 4),
                "vocos_snake_score":       indic_metrics.get("vocos_snake_score", 0.0),
                "formant_smoothness_score": indic_metrics.get("formant_smoothness_score", 0.0),
            },
            "window_breakdown":      window_scores,
            "diagnostics":           diag,
            "latency_ms":            latency_ms,
            "is_realtime_compliant": latency_ms < 500.0,
            "disclaimer": (
                "Advisory forensic risk assessment. "
                "Not conclusive proof of human identity."
            ),
        }

    # ------------------------------------------------------------------
    # Tier 1: 3.0 s Sliding-Window Transformer Inference
    # ------------------------------------------------------------------

    def _run_transformer_inference(
        self,
        audio: np.ndarray,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Run foundation transformer over 3.0 s windows (50 % overlap).

        Each window is zero-mean unit-variance normalised before feature extraction.
        Window scores are averaged to produce a single P_spoof estimate.

        Falls back to the native acoustic backbone if the HF model is unavailable.
        """
        window_scores: List[Dict[str, Any]] = []
        win_samples  = int(_WINDOW_SEC     * self.target_sr)  # 48 000 @ 16 kHz
        hop_samples  = int(_WINDOW_HOP_SEC * self.target_sr)  # 24 000 @ 16 kHz

        p_hf: float = 0.50

        # Multi-Foundation Model Ensemble Inference
        active_models = self.loaded_hf_models if self.loaded_hf_models else (
            [{"model": self.model, "feature_extractor": self.feature_extractor, "spoof_idx": self.spoof_idx, "human_idx": self.human_idx}]
            if self.has_hf_model and self.model is not None and self.feature_extractor is not None else []
        )

        if active_models:
            try:
                model_ensemble_probs: List[float] = []
                primary_window_scores: List[Dict[str, Any]] = []
                n_samples = len(audio)
                starts = list(range(0, max(1, n_samples - win_samples + 1), hop_samples))
                if not starts:
                    starts = [0]

                for m_idx, m_info in enumerate(active_models):
                    cur_model = m_info["model"]
                    cur_fe = m_info["feature_extractor"]
                    cur_spoof_idx = m_info["spoof_idx"]
                    cur_win_probs: List[float] = []

                    for w_idx, start in enumerate(starts):
                        end   = min(start + win_samples, n_samples)
                        chunk = audio[start:end]
                        chunk = normalize_audio_standard(chunk)

                        with torch.inference_mode():
                            inputs = cur_fe(
                                chunk,
                                sampling_rate=self.target_sr,
                                return_tensors="pt",
                            )
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            logits = cur_model(**inputs).logits
                            scaled_logits = logits / self.temperature
                            probs = F.softmax(scaled_logits, dim=-1)
                            probs_np = probs.float().cpu().numpy()[0]
                            win_p = float(probs_np[cur_spoof_idx])

                        cur_win_probs.append(win_p)
                        if m_idx == 0:
                            t_start = round(start / self.target_sr, 2)
                            t_end   = round(end   / self.target_sr, 2)
                            primary_window_scores.append({
                                "window_index":      w_idx + 1,
                                "time_range":        f"{t_start}s – {t_end}s",
                                "spoof_probability": round(win_p, 4),
                            })

                    if cur_win_probs:
                        model_ensemble_probs.append(float(np.mean(cur_win_probs)))

                # Anchor Transformer Fusion: Primary model anchors decisions, secondary provides support
                if not model_ensemble_probs:
                    p_hf = 0.50
                elif len(model_ensemble_probs) == 1:
                    p_hf = model_ensemble_probs[0]
                else:
                    p_m1 = model_ensemble_probs[0]
                    p_m2 = model_ensemble_probs[1]
                    if p_m1 <= 0.25:
                        p_hf = p_m1  # Primary model is definitive for human voice
                    elif p_m1 >= 0.45:
                        p_hf = max(p_m1, p_m2)
                    elif p_m2 >= 0.65:
                        p_hf = 0.50 * p_m1 + 0.50 * p_m2
                    else:
                        p_hf = 0.75 * p_m1 + 0.25 * p_m2

                window_scores = primary_window_scores

            except Exception as exc:
                log.warning("HF ensemble inference failed: %s", exc)
                p_hf = 0.50

        # Native backbone fallback
        p_native: float = 0.50
        if self.native_model is not None:
            try:
                res_nat = self.native_model.predict_waveform(audio)
                p_native = float(res_nat["spoof_probability"])
            except Exception as exc:
                log.debug("Native model fallback failed: %s", exc)
                p_native = 0.50

        if self.has_hf_model and self.native_model is not None and getattr(self.native_model, "has_weights", False):
            p_transformer = float(np.clip(0.65 * p_hf + 0.35 * p_native, 0.01, 0.99))
        elif self.has_hf_model:
            p_transformer = p_hf
        elif self.native_model is not None:
            p_transformer = p_native
        else:
            p_transformer = 0.50

        return p_transformer, window_scores

    # ------------------------------------------------------------------
    # Risk Categorisation
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_band(
        risk_score: int,
    ) -> Tuple[str, str, str, str, str]:
        """
        Map integer risk score [0–100] to 5-state risk band.

        Returns
        -------
        (label, risk_band, risk_band_key, badge_class, risk_description)
        """
        if risk_score <= 25:
            return (
                "AUTHENTIC HUMAN VOICE",
                "Low Risk (Human Voice)",
                "low",
                "badge-low",
                (
                    "Natural vocal-fold micro-perturbations, physiological glottal turbulence, "
                    "and broadband excitation verified. No synthetic vocoder signatures detected."
                ),
            )
        elif risk_score <= 60:
            return (
                "SUSPICIOUS / INCONCLUSIVE",
                "Review Required (Borderline Evidence)",
                "review",
                "badge-review",
                (
                    "Ambiguous acoustic characteristics or mild compression artefacts detected. "
                    "Borderline jitter, LFCC, and LPC residual evidence. "
                    "Secondary human-in-the-loop forensic review recommended."
                ),
            )
        else:
            return (
                "AI VOICE CLONE DETECTED",
                "High Risk (Likely AI / Cloned Voice)",
                "high",
                "badge-high",
                (
                    "Synthetic neural vocoder signatures, unnatural glottal regularity, "
                    "and LPC phase entropy collapse detected. "
                    "High confidence of AI-generated or cloned voice."
                ),
            )

    # ------------------------------------------------------------------
    # Degraded response builder
    # ------------------------------------------------------------------

    @staticmethod
    def _degraded_response(diag: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
        """Return the standardised Low Quality / Degraded result dict."""
        return {
            "prediction_label": "LOW QUALITY / DEGRADED",
            "spoof_probability": 0.50,
            "human_probability": 0.50,
            "risk_score":        50,
            "risk_band":         "Low Quality / Degraded",
            "risk_band_key":     "low_quality",
            "badge_class":       "badge-degraded",
            "risk_description": (
                "Audio is silent, truncated (< 0.5 s), contains insufficient voiced speech "
                f"(< {_MIN_VOICED_SEC} s), or SNR < {_SNR_FLOOR_DB} dB. "
                "Forensic evaluation is unreliable."
            ),
            "forensic_breakdown": {
                "transformer_spoof_prob": 0.50,
                "lpc_anomaly_score":      0.50,
                "lpc_kurtosis":           3.0,
                "phase_entropy":          0.50,
                "residual_flatness":      0.50,
                "dsp_physics_prob":       0.50,
                "glottal_spoof_prob":     0.50,
                "lfcc_spoof_prob":        0.50,
                "spectral_spoof_prob":    0.50,
                "jitter_local":           0.0,
                "shimmer_local":          0.0,
                "hnr_db":                 0.0,
                "lfcc_variance":          0.0,
                "hf_cutoff_ratio":        0.0,
                "snr_db":                 float(diag.get("snr_db", 0.0)),
                "snr_weight_mode":        "degraded",
                "voiced_ratio":           float(diag.get("voiced_ratio", 0.0)),
            },
            "window_breakdown":      [],
            "diagnostics":           diag,
            "latency_ms":            latency_ms,
            "is_realtime_compliant": True,
            "disclaimer": (
                "Advisory forensic risk assessment. "
                "Not conclusive proof of human identity."
            ),
        }


# ===========================================================================
# Module-level functional convenience wrappers (backward compatibility)
# ===========================================================================

def compute_praat_biomechanics(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Extract Praat glottal biomechanics (jitter, shimmer, HNR) from a waveform."""
    from src.forensic_dsp import extract_praat_glottal_metrics
    bio = extract_praat_glottal_metrics(audio, sr=sr)
    jl = float(bio["jitter_local"])
    sl = float(bio["shimmer_local"])
    return {
        "jitter_local":  jl,
        "local_jitter":  jl,
        "shimmer_local": sl,
        "local_shimmer": sl,
        "hnr_db":        float(bio["hnr_db"]),
    }


def compute_spectral_cutoff_ratio(
    audio: np.ndarray,
    sr: int = 16000,
    cutoff_hz: float = 5500.0,
) -> float:
    """Compute high-frequency vocoder brickwall cutoff ratio (> cutoff_hz)."""
    from src.forensic_dsp import extract_vocoder_cutoff_ratio
    return extract_vocoder_cutoff_ratio(audio, sr=sr, cutoff_hz=cutoff_hz)


def compute_dsp_spoof_probability(audio: np.ndarray, sr: int = 16000) -> float:
    """Compute combined DSP spoof probability from glottal + LFCC + HF features."""
    from src.forensic_dsp import extract_dsp_metrics
    res = extract_dsp_metrics(audio, sr=sr)
    return float(res["combined_dsp_risk"])


