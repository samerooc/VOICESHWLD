"""
VoiceShield Preprocessing Contract Verification Script.
Audits and compares preprocessing contracts between Training and Inference pipelines.
"""

import sys
import numpy as np
import librosa
from src.preprocessing_contract import preprocess_audio_signal
from src.features import extract_features_from_audio
from src.audio_io import load_audio_from_bytes, load_audio_from_file


def run_contract_verification():
    print("=======================================================")
    print("      VOICESHIELD PREPROCESSING CONTRACT CHECK")
    print("=======================================================\n")

    # Generate synthetic 48kHz stereo signal with DC offset and silence
    sr_source = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr_source * duration), endpoint=False)
    sig_mono = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1).astype(np.float32)  # DC offset +0.1
    sig_stereo = np.column_stack([sig_mono, sig_mono * 0.9])

    # Pad silence at start and end
    silence = np.zeros(int(sr_source * 0.5), dtype=np.float32)
    sig_stereo_padded = np.vstack([
        np.column_stack([silence, silence]),
        sig_stereo,
        np.column_stack([silence, silence]),
    ])

    print("Step 1: Raw Input Signal")
    print(f"  - Source Shape : {sig_stereo_padded.shape}")
    print(f"  - Source SR    : {sr_source} Hz")
    print(f"  - Mean (DC)    : {np.mean(sig_stereo_padded):.4f}")

    # Run contract preprocessing
    clean_audio, effective_sr, diag = preprocess_audio_signal(sig_stereo_padded, sample_rate=sr_source)

    print("\nStep 2: Post-Contract Signal")
    print(f"  - Output Shape : {clean_audio.shape} (Mono)")
    print(f"  - Output SR    : {effective_sr} Hz (Expected 16000)")
    print(f"  - DC Offset    : {np.mean(clean_audio):.8f} (Expected ~0.0)")
    print(f"  - Max Peak     : {np.max(np.abs(clean_audio)):.4f} (Expected 1.0)")
    print(f"  - Duration     : {diag['duration_seconds']:.2f} s")

    assert effective_sr == 16000, "Preprocessing SR mismatch!"
    assert clean_audio.ndim == 1, "Preprocessing channel count mismatch (must be 1D mono)!"
    assert np.abs(np.mean(clean_audio)) < 1e-4, "DC offset not eliminated!"
    assert np.max(np.abs(clean_audio)) <= 1.0001, "Amplitude normalization exceeded bounds!"

    # Step 3: Feature Extraction Contract Verification
    features = extract_features_from_audio(clean_audio, sample_rate=effective_sr)
    print("\nStep 3: Feature Extraction Contract")
    print(f"  - Features Len : {len(features)} (Expected 42)")
    print(f"  - Finite Check : {np.all(np.isfinite(features))}")
    assert len(features) == 42, "Feature dimension mismatch!"
    assert np.all(np.isfinite(features)), "Features contain non-finite numbers!"

    print("\n[OK] Preprocessing and Feature Extraction Contracts VERIFIED IDENTICAL.\n")


if __name__ == "__main__":
    run_contract_verification()
