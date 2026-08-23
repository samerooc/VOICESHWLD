# VoiceShield Model Registry & Storage (Phase 13)

## 1. Registry Requirements

Every trained model stored in `models/` must be accompanied by `model_metadata.json` containing:
- **`model_version`**: Semantic version identifier.
- **`feature_version`**: Fixed schema version (`1.0.0`).
- **`preprocessing_version`**: Preprocessing contract version (`1.0.0`).
- **`model_artifact_sha256`**: Cryptographic SHA-256 hash of the binary file.
- **`class_mapping`**: Strictly `{0: "bona_fide", 1: "spoof"}`.
- **`training_dataset_hash`**: Hash of the training manifest.
- **`validation_metrics`**: Formally computed cross-validation evaluation results.

## 2. Integrity Enforcement

The `src/model_registry.py` loader validates these criteria before deserializing any model artifact into volatile memory, blocking corrupted or unverified checkpoints.
