"""
VoiceShield Phase 1 — Standalone Verification Test Suite.

Tests the complete Phase 1 audio ingestion and preprocessing pipeline:
  Test 1: Ingest valid in-memory generated 16kHz WAV byte array.
  Test 2: Ingest simulated 8kHz telephony mono audio bytes (WAV container).
  Test 3: Ingest pure silence / white noise and assert is_silent: True.
  Test 4: Verify output tensor shapes and zero-mean unit-variance bounds.

Additional coverage:
  Test 5: AcousticChannelNormalizer pipeline (spectral subtraction + CMVN).
  Test 6: Diagnostic dictionary field contract.
  Test 7: SNR estimator smoke-test.
  Test 8: Clipping detection.

Run with:
    pytest tests/test_phase1.py -v

All tests are self-contained and require only in-memory synthetic audio;
no audio files on disk are read or written.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import soundfile as sf

from src.audio_processor import (
    AudioProcessor,
    SAMPLE_RATE,
    compute_snr_db,
    decode_and_sanitize_audio,
    extract_voiced_segments,
    normalize_audio_standard,
)
from src.channel_normalizer import (
    AcousticChannelNormalizer,
    normalize_channel,
)


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(
    duration_sec: float = 1.0,
    sr: int = SAMPLE_RATE,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """
    Return in-memory WAV bytes for a pure sinusoidal tone.

    The generated signal is float32 encoded via soundfile into a WAV container
    so the byte buffer is representative of a real browser-uploaded WAV file.
    """
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _make_8k_wav_bytes(
    duration_sec: float = 1.0,
    freq: float = 300.0,
    amplitude: float = 0.6,
) -> bytes:
    """
    Return in-memory 8 kHz 16-bit PCM WAV bytes (telephony quality).

    The WAV container carries the native 8000 Hz sample rate in its header so
    ``AudioProcessor`` can detect the original SR and upsample to 16 kHz.
    """
    sr = 8000
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio_f32 = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm16, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _make_silence_wav_bytes(duration_sec: float = 1.0, sr: int = SAMPLE_RATE) -> bytes:
    """Return in-memory WAV bytes containing pure digital silence (all zeros)."""
    silence = np.zeros(int(sr * duration_sec), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, sr, format="WAV")
    return buf.getvalue()


def _make_low_noise_wav_bytes(
    duration_sec: float = 1.0,
    sr: int = SAMPLE_RATE,
    rms_level: float = 0.0002,
) -> bytes:
    """
    Return in-memory WAV bytes containing extremely low-level white noise
    (simulates room hiss / microphone self-noise below MIN_AUDIO_RMS_ENERGY).
    """
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, rms_level, int(sr * duration_sec)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, noise, sr, format="WAV")
    return buf.getvalue()


# ===========================================================================
# TEST 1 — Ingest valid in-memory generated 16 kHz WAV byte array
# ===========================================================================

class TestIngest16kWav:
    """Test 1: Validate full ingestion pipeline for a 16 kHz WAV in-memory buffer."""

    def test_returns_three_element_tuple(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.5, sr=16000, freq=440.0)
        result = processor.load_audio_from_bytes(wav_bytes, target_sr=16000)
        assert len(result) == 3, "load_audio_from_bytes must return (full_audio, voiced_audio, diag)"

    def test_output_types(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.5, sr=16000, freq=440.0)
        full_audio, voiced_audio, diag = processor.load_audio_from_bytes(wav_bytes)
        assert isinstance(full_audio, np.ndarray), "full_audio must be np.ndarray"
        assert isinstance(voiced_audio, np.ndarray), "voiced_audio must be np.ndarray"
        assert isinstance(diag, dict), "diag must be a dict"

    def test_correct_sample_count(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.5, sr=16000, freq=440.0)
        full_audio, _, _ = processor.load_audio_from_bytes(wav_bytes)
        expected = int(1.5 * 16000)  # 24000
        assert len(full_audio) == expected, f"Expected {expected} samples, got {len(full_audio)}"

    def test_float32_mono_dtype(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.0, sr=16000)
        full_audio, voiced_audio, _ = processor.load_audio_from_bytes(wav_bytes)
        assert full_audio.ndim == 1, "full_audio must be 1D mono"
        assert full_audio.dtype == np.float32, "full_audio dtype must be float32"
        assert voiced_audio.ndim == 1, "voiced_audio must be 1D mono"

    def test_amplitude_bounded(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.0, sr=16000, amplitude=0.5)
        full_audio, _, _ = processor.load_audio_from_bytes(wav_bytes)
        assert np.max(np.abs(full_audio)) <= 1.0, "Audio amplitude must be bounded in [-1.0, 1.0]"

    def test_diagnostics_keys(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.0, sr=16000)
        _, _, diag = processor.load_audio_from_bytes(wav_bytes)
        required_keys = {
            "original_sr", "duration_sec", "voiced_sec", "snr_db",
            "is_clipped", "is_silent", "sample_rate",
        }
        missing = required_keys - set(diag.keys())
        assert not missing, f"Diagnostics dict missing keys: {missing}"

    def test_diagnostics_values(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=1.5, sr=16000, freq=440.0, amplitude=0.5)
        _, _, diag = processor.load_audio_from_bytes(wav_bytes)
        assert diag["sample_rate"] == 16000
        assert diag["original_sr"] == 16000
        assert diag["duration_sec"] == pytest.approx(1.5, abs=0.05)
        assert diag["is_silent"] is False, "440 Hz tone at 0.5 amplitude must NOT be silent"
        assert diag["is_clipped"] is False
        assert diag["snr_db"] >= 10.0, f"Expected SNR >= 10 dB for clean tone, got {diag['snr_db']}"


# ===========================================================================
# TEST 2 — Ingest simulated 8 kHz telephony mono audio bytes
# ===========================================================================

class TestIngest8kTelephony:
    """Test 2: 8 kHz telephony WAV decoded and upsampled to 16 kHz cleanly."""

    def test_original_sr_detected_as_8000(self):
        processor = AudioProcessor(target_sr=16000)
        pcm8k_bytes = _make_8k_wav_bytes(duration_sec=1.0, freq=300.0)
        _, _, diag = processor.load_audio_from_bytes(pcm8k_bytes, target_sr=16000)
        assert diag["original_sr"] == 8000, (
            f"Expected original_sr=8000 for 8kHz WAV, got {diag['original_sr']}"
        )

    def test_output_resampled_to_16k(self):
        processor = AudioProcessor(target_sr=16000)
        pcm8k_bytes = _make_8k_wav_bytes(duration_sec=1.0, freq=300.0)
        _, _, diag = processor.load_audio_from_bytes(pcm8k_bytes, target_sr=16000)
        assert diag["sample_rate"] == 16000, "Output must be resampled to 16000 Hz"

    def test_sample_count_after_upsample(self):
        """1 second of 8kHz audio resampled to 16kHz should yield 16000 samples."""
        processor = AudioProcessor(target_sr=16000)
        pcm8k_bytes = _make_8k_wav_bytes(duration_sec=1.0, freq=300.0)
        full_audio, _, _ = processor.load_audio_from_bytes(pcm8k_bytes, target_sr=16000)
        assert len(full_audio) == 16000, (
            f"Expected 16000 samples after 8->16kHz upsample, got {len(full_audio)}"
        )

    def test_amplitude_bounded_after_upsample(self):
        processor = AudioProcessor(target_sr=16000)
        pcm8k_bytes = _make_8k_wav_bytes(duration_sec=1.0)
        full_audio, _, _ = processor.load_audio_from_bytes(pcm8k_bytes, target_sr=16000)
        assert np.max(np.abs(full_audio)) <= 1.0, "Upsampled audio must stay within [-1.0, 1.0]"

    def test_telephony_audio_not_silent(self):
        """300 Hz tone at 8kHz is recognisable speech — must NOT trigger is_silent."""
        processor = AudioProcessor(target_sr=16000)
        pcm8k_bytes = _make_8k_wav_bytes(duration_sec=1.5, freq=300.0, amplitude=0.6)
        _, _, diag = processor.load_audio_from_bytes(pcm8k_bytes, target_sr=16000)
        assert diag["is_silent"] is False, "Telephony tone at 0.6 amplitude must NOT be silent"

    def test_float32_dtype(self):
        processor = AudioProcessor(target_sr=16000)
        pcm8k_bytes = _make_8k_wav_bytes(duration_sec=1.0)
        full_audio, _, _ = processor.load_audio_from_bytes(pcm8k_bytes)
        assert full_audio.dtype == np.float32


# ===========================================================================
# TEST 3 — Ingest pure silence / white noise → is_silent: True
# ===========================================================================

class TestSilenceDetection:
    """Test 3: Pure silence and sub-threshold noise must trigger is_silent: True."""

    def test_pure_silence_is_silent(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        silence_bytes = _make_silence_wav_bytes(duration_sec=1.0)
        _, _, diag = processor.load_audio_from_bytes(silence_bytes)
        assert diag["is_silent"] is True, "Pure digital silence must trigger is_silent: True"

    def test_pure_silence_snr_zero(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        silence_bytes = _make_silence_wav_bytes(duration_sec=1.0)
        _, _, diag = processor.load_audio_from_bytes(silence_bytes)
        assert diag["snr_db"] < MIN_AUDIO_SNR_DB_THRESHOLD, (
            "Pure silence SNR must be below the 3 dB threshold"
        )

    def test_low_energy_noise_is_silent(self):
        """Sub-threshold room hiss (RMS ~0.0002) must be classified as silent."""
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        noise_bytes = _make_low_noise_wav_bytes(duration_sec=1.0, rms_level=0.0002)
        _, _, diag = processor.load_audio_from_bytes(noise_bytes)
        assert diag["is_silent"] is True, (
            "Sub-threshold ambient noise (RMS=0.0002) must trigger is_silent: True"
        )

    def test_voiced_sec_near_zero_for_silence(self):
        """
        For pure silence the VAD fallback (librosa.effects.trim) returns the full
        audio to prevent a zero-length output; is_silent is still True because
        rms_energy < MIN_AUDIO_RMS_ENERGY.  We verify that the diagnostics reflect
        a near-zero RMS energy level rather than asserting on voiced_sec directly.
        """
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        silence_bytes = _make_silence_wav_bytes(duration_sec=2.0)
        _, _, diag = processor.load_audio_from_bytes(silence_bytes)
        # The key contract: silence must be flagged even if VAD outputs full audio
        assert diag["is_silent"] is True, "Pure silence must always yield is_silent: True"
        assert diag["rms_energy"] < 0.001, (
            f"Pure silence RMS must be near-zero; got {diag['rms_energy']}"
        )

    def test_active_speech_not_silent(self):
        """Baseline sanity: a healthy speech-like tone must NOT trigger is_silent."""
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        tone_bytes = _make_wav_bytes(duration_sec=1.5, freq=440.0, amplitude=0.5)
        _, _, diag = processor.load_audio_from_bytes(tone_bytes)
        assert diag["is_silent"] is False, "440 Hz at 0.5 amplitude must NOT be silent"


# Helper constant shared by Test 3
MIN_AUDIO_SNR_DB_THRESHOLD: float = 3.0


# ===========================================================================
# TEST 4 — Output tensor shapes and zero-mean unit-variance bounds
# ===========================================================================

class TestTensorShapesAndNormalization:
    """Test 4: Verify shapes, dtype, range, and per-utterance z-score normalization."""

    def test_full_audio_shape_and_dtype(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=2.0, sr=16000, freq=500.0)
        full_audio, _, _ = processor.load_audio_from_bytes(wav_bytes)
        assert full_audio.ndim == 1, "full_audio must be 1D"
        assert full_audio.dtype == np.float32, "full_audio dtype must be float32"
        assert len(full_audio) == 32000, f"Expected 32000 samples, got {len(full_audio)}"

    def test_amplitude_range(self):
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=2.0, freq=500.0, amplitude=0.5)
        full_audio, _, _ = processor.load_audio_from_bytes(wav_bytes)
        assert np.max(np.abs(full_audio)) <= 1.0, "Audio must be bounded in [-1.0, 1.0]"
        assert np.min(full_audio) >= -1.0, "Audio minimum must be >= -1.0"

    def test_normalize_audio_standard_zero_mean(self):
        """normalize_audio_standard must produce zero-mean output."""
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=2.0, freq=500.0)
        full_audio, _, _ = processor.load_audio_from_bytes(wav_bytes)
        normed = normalize_audio_standard(full_audio)
        assert np.mean(normed) == pytest.approx(0.0, abs=1e-5), (
            f"Normalized audio mean must be ~0; got {np.mean(normed):.2e}"
        )

    def test_normalize_audio_standard_unit_variance(self):
        """normalize_audio_standard must produce unit-variance output."""
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=2.0, freq=500.0)
        full_audio, _, _ = processor.load_audio_from_bytes(wav_bytes)
        normed = normalize_audio_standard(full_audio)
        assert np.std(normed) == pytest.approx(1.0, abs=1e-3), (
            f"Normalized audio std must be ~1; got {np.std(normed):.4f}"
        )

    def test_normalization_formula_explicit(self):
        """
        Explicit formula verification:
          x_hat = (x - mu) / (sigma + 1e-7)
        """
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mu = float(np.mean(audio))           # 3.0
        sigma = float(np.std(audio))         # ~1.4142
        expected = (audio - mu) / (sigma + 1e-7)
        result = normalize_audio_standard(audio)
        np.testing.assert_allclose(result, expected.astype(np.float32), rtol=1e-5)

    def test_voiced_audio_is_subset_or_equal(self):
        """voiced_audio must have <= samples than full_audio."""
        processor = AudioProcessor(target_sr=SAMPLE_RATE)
        wav_bytes = _make_wav_bytes(duration_sec=2.0, freq=500.0)
        full_audio, voiced_audio, _ = processor.load_audio_from_bytes(wav_bytes)
        assert len(voiced_audio) <= len(full_audio), (
            "voiced_audio must not be longer than full_audio"
        )
        assert voiced_audio.dtype == np.float32
        assert voiced_audio.ndim == 1

    def test_decode_and_sanitize_audio_functional_api(self):
        """decode_and_sanitize_audio functional API must behave identically to class API."""
        wav_bytes = _make_wav_bytes(duration_sec=1.0, freq=440.0)
        full_audio, voiced_audio, diag = decode_and_sanitize_audio(wav_bytes, target_sr=16000)
        assert isinstance(full_audio, np.ndarray)
        assert isinstance(voiced_audio, np.ndarray)
        assert isinstance(diag, dict)
        assert full_audio.dtype == np.float32
        assert diag["sample_rate"] == 16000


# ===========================================================================
# TEST 5 — Acoustic Channel Normalizer pipeline
# ===========================================================================

class TestAcousticChannelNormalizer:
    """Test 5: Verify AcousticChannelNormalizer spectral subtraction + CMVN pipeline."""

    def test_output_type_and_dtype(self):
        normalizer = AcousticChannelNormalizer(sr=16000)
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        speech = (0.5 * np.sin(2.0 * np.pi * 300 * t)).astype(np.float32)
        result = normalizer.normalize_channel(speech, sr=16000)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_output_length_preserved(self):
        normalizer = AcousticChannelNormalizer(sr=16000)
        audio = np.random.default_rng(0).random(16000).astype(np.float32) * 0.5
        result = normalizer.normalize_channel(audio, sr=16000)
        assert len(result) == len(audio), "Output length must match input"

    def test_amplitude_bounded(self):
        normalizer = AcousticChannelNormalizer(sr=16000)
        rng = np.random.default_rng(1)
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        noisy = (0.5 * np.sin(2 * np.pi * 300 * t) + rng.normal(0, 0.05, 16000)).astype(np.float32)
        result = normalizer.normalize_channel(noisy, sr=16000)
        assert np.max(np.abs(result)) <= 1.0, "Channel-normalized audio must be in [-1.0, 1.0]"

    def test_spectral_subtraction_reduces_background_noise(self):
        """
        After spectral subtraction, energy of a noise-only segment (first 0.25 s)
        must be reduced relative to the noisy input.
        """
        normalizer = AcousticChannelNormalizer(sr=16000)
        rng = np.random.default_rng(2)
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = rng.normal(0.0, 0.08, 16000).astype(np.float32)
        noisy = speech + noise

        denoised = normalizer.spectral_subtraction(noisy)
        noise_segment = noisy[:4000]
        denoised_segment = denoised[:4000]
        energy_noisy = float(np.mean(noise_segment ** 2))
        energy_denoised = float(np.mean(denoised_segment ** 2))
        assert energy_denoised <= energy_noisy, (
            "Spectral subtraction must reduce noise-segment energy"
        )

    def test_cmvn_output_length(self):
        normalizer = AcousticChannelNormalizer(sr=16000)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        result = normalizer.apply_cmvn(audio)
        assert len(result) == len(audio)
        assert result.dtype == np.float32

    def test_functional_wrapper(self):
        """Module-level normalize_channel convenience wrapper must work identically."""
        t = np.linspace(0.0, 1.0, 16000, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        result = normalize_channel(audio, sr=16000)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
        assert result.dtype == np.float32

    def test_empty_input_returns_empty(self):
        normalizer = AcousticChannelNormalizer(sr=16000)
        empty = np.array([], dtype=np.float32)
        result = normalizer.normalize_channel(empty)
        assert len(result) == 0


# ===========================================================================
# TEST 6 — Diagnostics dictionary field contract
# ===========================================================================

class TestDiagnosticsContract:
    """Test 6: Validate the complete diagnostics dictionary field contract."""

    REQUIRED_FIELDS = {
        "original_sr", "duration_sec", "voiced_sec", "snr_db",
        "is_silent", "is_clipped", "rms_energy", "voiced_ratio",
        "sample_rate", "num_samples",
    }

    def test_all_required_fields_present(self):
        wav_bytes = _make_wav_bytes(duration_sec=1.0)
        _, _, diag = decode_and_sanitize_audio(wav_bytes, target_sr=16000)
        missing = self.REQUIRED_FIELDS - set(diag.keys())
        assert not missing, f"Diagnostics missing required fields: {missing}"

    def test_field_types(self):
        wav_bytes = _make_wav_bytes(duration_sec=1.0)
        _, _, diag = decode_and_sanitize_audio(wav_bytes, target_sr=16000)
        assert isinstance(diag["original_sr"], int)
        assert isinstance(diag["duration_sec"], float)
        assert isinstance(diag["voiced_sec"], float)
        assert isinstance(diag["snr_db"], float)
        assert isinstance(diag["is_silent"], bool)
        assert isinstance(diag["is_clipped"], bool)
        assert isinstance(diag["rms_energy"], float)

    def test_snr_non_negative(self):
        wav_bytes = _make_wav_bytes(duration_sec=1.0)
        _, _, diag = decode_and_sanitize_audio(wav_bytes)
        assert diag["snr_db"] >= 0.0, "SNR must always be non-negative"

    def test_voiced_ratio_bounded(self):
        wav_bytes = _make_wav_bytes(duration_sec=1.5, freq=440.0)
        _, _, diag = decode_and_sanitize_audio(wav_bytes)
        assert 0.0 <= diag["voiced_ratio"] <= 1.0, "voiced_ratio must be in [0, 1]"


# ===========================================================================
# TEST 7 — SNR estimator smoke-test
# ===========================================================================

class TestSNREstimator:
    """Test 7: compute_snr_db returns sensible values for edge cases."""

    def test_pure_silence_snr_is_zero(self):
        silence = np.zeros(16000, dtype=np.float32)
        snr = compute_snr_db(silence)
        assert snr == 0.0

    def test_empty_array_snr_is_zero(self):
        snr = compute_snr_db(np.array([], dtype=np.float32))
        assert snr == 0.0

    def test_clean_tone_snr_is_high(self):
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        snr = compute_snr_db(tone)
        assert snr > 10.0, f"Clean 440 Hz tone must have SNR > 10 dB; got {snr:.1f}"

    def test_snr_bounded_in_range(self):
        rng = np.random.default_rng(99)
        audio = rng.random(16000).astype(np.float32) * 0.5
        snr = compute_snr_db(audio)
        assert 0.0 <= snr <= 80.0, f"SNR out of bounds: {snr}"


# ===========================================================================
# TEST 8 — Clipping detection
# ===========================================================================

class TestClippingDetection:
    """Test 8: Verify is_clipped is correctly detected and peak normalization applied."""

    def test_clipped_audio_detected(self):
        """Audio with peak >= 0.999 before normalization should set is_clipped: True."""
        # Build audio that clips at exactly ±1.0
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = np.sin(2 * np.pi * 300 * t).astype(np.float32)  # peak == 1.0
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        _, _, diag = decode_and_sanitize_audio(buf.getvalue())
        assert diag["is_clipped"] is True, "Peak == 1.0 audio must trigger is_clipped: True"

    def test_non_clipped_audio(self):
        """Audio with peak 0.5 must NOT trigger is_clipped."""
        wav_bytes = _make_wav_bytes(duration_sec=1.0, amplitude=0.5)
        _, _, diag = decode_and_sanitize_audio(wav_bytes)
        assert diag["is_clipped"] is False, "0.5 amplitude audio must NOT be clipped"

    def test_post_normalization_amplitude_bounded(self):
        """Even for clipping input, output must be bounded in [-1.0, 1.0]."""
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = (2.0 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # over-driven
        buf = io.BytesIO()
        sf.write(buf, np.clip(audio, -1.0, 1.0), 16000, format="WAV")
        full_audio, _, _ = decode_and_sanitize_audio(buf.getvalue())
        assert np.max(np.abs(full_audio)) <= 1.0
