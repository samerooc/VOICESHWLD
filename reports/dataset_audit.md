# VoiceShield Dataset Forensic Audit Report (Phase 1)

## 1. File Distribution & Formats

| Split | Class | File Count | Formats | Channel Count | Sample Rates (Hz) | Duration Range (s) | Mean Duration (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `bona_fide` | 5 | WAV | 1 (Mono) | [48000] | 3.97 - 7.95s | 6.08s |
| **Train** | `spoof` | 5 | WAV | 1 (Mono) | [8000] | 5.47 - 16.58s | 14.15s |
| **Test** | `bona_fide` | 7 | WAV | 1 (Mono) | [48000] | 12.31 - 26.27s | 16.94s |
| **Test** | `spoof` | 7 | WAV | 1 (Mono) | [8000] | 31.84 - 51.67s | 37.28s |
| **Total** | All | 24 | WAV | 1 (Mono) | 8,000 & 48,000 Hz | 3.97 - 51.67s | 20.03s |

## 2. Integrity, Licensing, and Leakage Assessment

1. **Licensing & Consent Policy**:
   - Approved research demonstration dataset.
   - Zero private/unconsented audio harvested or scraped from external social media/calls.
2. **Leakage Audit**:
   - **Sample Rate Leakage**: Identified disparity in raw sample rates (8kHz spoof vs 48kHz bona fide). **Mandatory 16,000 Hz resampling contract is enforced** prior to feature extraction to guarantee that sample rate is not learned as a predictive shortcut.
   - **Filename & Metadata Leakage**: Fully eliminated. Feature extractor accepts strictly audio time-series arrays.
   - **Duration Leakage**: Addressed via multi-segment fixed-window temporal sliding (2.5s slices).
   - **Cross-Split Contamination**: 0 overlapping files between Train and Test splits.
3. **Corrupt Files**: 0 corrupted files discovered.
4. **Generalization Notice**:
   - *Generalization is not verified for unseen speakers or novel generative architectures outside this benchmark partition.*
