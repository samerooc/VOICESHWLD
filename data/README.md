# VoiceShield Dataset Documentation & Sources (Phase 2 & Phase 3)

## 1. Overview & Ethical Standards

VoiceShield uses curated, verified research audio partitions designed to test acoustic feature extraction and statistical classification under ethical cybersecurity principles.

### Critical Safety Commitments:
1. **NO Private Audio Harvesting**: No audio is scraped from social media, YouTube, calls, or private files.
2. **NO Voice Cloning**: No real individual's voice is cloned or impersonated.
3. **NO External Ingestion**: Audio is processed entirely locally in memory.
4. **Consent & Licensing**: All audio is approved for research and educational benchmarking.

---

## 2. Dataset Structure

```
data/
├── manifest.csv               # Ground-truth metadata manifest with SHA-256 hashes
├── evaluation_manifest.csv    # Multi-condition evaluation manifest
├── human/                     # Train bona fide speech samples (48kHz)
├── ai_voice/                  # Train synthetic/cloned speech samples (8kHz)
└── test/
    ├── human/                 # Held-out bona fide speech samples (48kHz)
    └── ai_voice/              # Held-out synthetic/cloned speech samples (8kHz)
```

---

## 3. Approved Sources & Protocols

- **Reference Protocol**: ASVspoof (Automatic Speaker Verification and Spoofing Countermeasures) challenge taxonomy.
- **Class 0 (`bona_fide`)**: Natural human speech recordings.
- **Class 1 (`spoof`)**: Synthetic TTS / neural vocoder phase-synthesized audio.
- **Sample Rate Protocol**: Downsampled / upsampled to standard **16,000 Hz** across all feature extraction routines.
- **Licensing**: Permitted for non-commercial research, security prototyping, and algorithmic evaluation.
