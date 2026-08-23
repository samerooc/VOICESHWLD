# VoiceShield Reproducibility Guide (Phase 15)

## 1. Overview

VoiceShield machine learning models are 100% reproducible through version-controlled configuration files, deterministic random seeds, and cryptographic dataset manifest hashing.

## 2. Environment Specifications

- **Python Version**: `3.14.x` (or `>= 3.10`)
- **Key Dependencies**:
  - `scikit-learn`
  - `numpy`
  - `librosa`
  - `soundfile`
  - `joblib`
  - `pyyaml`
- **Random Seed**: Fixed at `42` across all data splitters, augmentations, and classifier initializations.

## 3. Reproduction Commands

### Train the Model:
```powershell
python scripts/train.py --config configs/training.yaml
```

### Run Independent Evaluation:
```powershell
python scripts/evaluate_model.py --checkpoint models/voice_detector.pkl
```

### Run Full Reproducibility Suite:
```powershell
python scripts/reproduce_training.py
```
