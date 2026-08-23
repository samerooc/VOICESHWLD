"""
VoiceShield: AI Voice Deepfake & Impersonation Risk Detection Dashboard (Explainable SOC).
Privacy-First, In-Memory Audio Analysis for Security Operations with Signal Diagnostics.
"""

import json
import os
from typing import Any, Dict, Optional
import librosa
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

from src.audio_io import get_audio_metadata, load_audio_from_bytes
from src.config import (
    CONFUSION_MATRIX_PNG,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    N_MFCC,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
)
from src.explainability import (
    EXPLAINABILITY_DISCLAIMER,
    build_explainability_report,
    get_global_feature_importance,
)
from src.model import load_metadata, load_model
from src.privacy import get_privacy_statement
from src.scoring import predict_and_score

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Dark Security Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VoiceShield — AI Voice Deepfake Risk & Explainability SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark Cybersecurity Clean Theme */
    .stApp {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
    }
    .main-header {
        border-left: 5px solid #0284c7;
        padding-left: 16px;
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 4px;
    }
    .metric-card {
        background: #131c2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge-low {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-review {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-high {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-uncertain {
        background-color: #4c1d95;
        color: #c4b5fd;
        border: 1px solid #8b5cf6;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .disclaimer-box {
        background: #0f172a;
        border: 1px solid #38bdf8;
        color: #bae6fd;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-top: 16px;
    }
    .ood-box {
        background: #3f1515;
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.model_registry import verify_and_load_model

# -----------------------------------------------------------------------------
# Cached Loaders for Model, Metadata, and Evaluation Reports
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_and_metadata():
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_METADATA_PATH):
        try:
            return verify_and_load_model(MODEL_PATH, MODEL_METADATA_PATH)
        except Exception:
            return load_model(MODEL_PATH), load_metadata(MODEL_METADATA_PATH)
    return load_model(MODEL_PATH), load_metadata(MODEL_METADATA_PATH)


@st.cache_resource
def get_model():
    m, _ = get_model_and_metadata()
    return m


@st.cache_data
def get_metadata() -> Optional[Dict[str, Any]]:
    _, meta = get_model_and_metadata()
    return meta


@st.cache_data
def get_metrics() -> Optional[Dict[str, Any]]:
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


trained_model, model_metadata = get_model_and_metadata()
eval_metrics = get_metrics()

# -----------------------------------------------------------------------------
# Sidebar: System Metadata & Privacy Guarantees
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("VoiceShield SOC")
    st.caption("Privacy-First Cybersecurity Research Prototype")
    
    st.markdown(
        """
        <div class="disclaimer-box">
            🔒 <b>Privacy-First Protocol</b><br>
            • In-memory stream processing<br>
            • Zero raw audio retention / No history<br>
            • No automatic blocking or external alerts<br>
            • Advisory decision support only
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("📦 Model & Dataset Metadata")
    if model_metadata:
        st.write(f"• **Model Version**: `{model_metadata.get('model_version', '1.0.0')}`")
        st.write(f"• **Feature Version**: `{model_metadata.get('feature_version', '1.0.0')}`")
        st.write(f"• **Class Mapping**: `0 = bona_fide, 1 = spoof`")
        st.write(f"• **Total Features**: `{model_metadata.get('feature_configuration', {}).get('total_features', 42)} Acoustic Markers`")
        st.write(f"• **Dataset Partition**: `24 Samples (10 Train, 14 Test)`")
        st.write(f"• **Tuned Threshold**: `{model_metadata.get('optimal_decision_threshold', 0.50):.3f}`")
        if "training_dataset_hash" in model_metadata:
            hash_short = model_metadata["training_dataset_hash"][:12]
            st.write(f"• **Dataset Hash**: `SHA256:{hash_short}...`")
    else:
        st.info("Train the model with `python scripts/train_model.py` to generate model metadata.")

    st.divider()
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#94a3b8; border:1px solid #334155; padding:10px; border-radius:6px;">
            <b>Visible Disclaimer</b>:<br>
            <i>“Experimental decision-support prototype; not identity proof.”</i><br><br>
            ⚠️ <i>“Prediction reliability depends on audio quality and similarity to evaluation data.”</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Section 1: Project Title and Problem Statement
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">🛡️ VoiceShield: Explainable AI Voice Deepfake & Impersonation Risk Detection Platform</h1>
        <div class="main-subtitle">
            <b>Problem Statement</b>: Generative neural text-to-speech (TTS) and zero-shot voice cloning allow threat actors
            to impersonate executives, employees, and customers during authorization calls.
            VoiceShield inspects acoustic spectral, pitch, timing, and energy markers in volatile memory to provide
            calibrated risk scores and transparent explainability to assist human security analysts.
        </div>
    </div>
    <div style="background:#0c1d33; border:1px solid #1e3a8a; padding:10px 16px; border-radius:8px; margin-bottom:18px; font-size:0.88rem; color:#93c5fd;">
        🔒 <b>Consent & Ethical Notice</b>: Audio samples analyzed are processed strictly in volatile memory. No audio is permanently logged or shared externally. This system never executes automatic blocking.
        <br>⚠️ <i>Prediction reliability depends on audio quality and similarity to evaluation data.</i>
    </div>
    """,
    unsafe_allow_html=True,
)

if trained_model is None:
    st.warning("⚠️ Trained model not found at `models/voice_detector.pkl`. Please run `python scripts/train_model.py` first.")

# -----------------------------------------------------------------------------
# Main Tabs: Real-Time Analyzer, Explainability Diagnostics & Benchmark Reports
# -----------------------------------------------------------------------------
tab_analyzer, tab_explain, tab_reports, tab_stream = st.tabs([
    "🔍 Voice Authenticity Inspector",
    "🔬 Explainability & Signal Diagnostics",
    "📊 Independent Evaluation & Benchmarks",
    "📡 Live Call Streaming Simulator (Sandbox)",
])

with tab_analyzer:
    col_input1, col_input2 = st.columns([1.1, 1])

    with col_input1:
        st.subheader("1. Voice Ingestion")
        input_source = st.radio(
            "Select Audio Ingestion Method",
            ["Pre-loaded Benchmark Samples", "Upload Audio File", "Record Live Microphone"],
            horizontal=True,
        )

        raw_audio_bytes: Optional[bytes] = None
        audio_source_label = "Unknown Source"

        if input_source == "Pre-loaded Benchmark Samples":
            sample_choice = st.selectbox(
                "Select Verified Research Sample",
                [
                    "Human Genuine Voice #1 (data/test/human/01.wav)",
                    "Human Genuine Voice #2 (data/test/human/02.wav)",
                    "AI Cloned Voice #1 (data/test/ai_voice/1.wav)",
                    "AI Cloned Voice #2 (data/test/ai_voice/2.wav)",
                ],
            )
            sample_paths = {
                "Human Genuine Voice #1 (data/test/human/01.wav)": "data/test/human/01.wav",
                "Human Genuine Voice #2 (data/test/human/02.wav)": "data/test/human/02.wav",
                "AI Cloned Voice #1 (data/test/ai_voice/1.wav)": "data/test/ai_voice/1.wav",
                "AI Cloned Voice #2 (data/test/ai_voice/2.wav)": "data/test/ai_voice/2.wav",
            }
            chosen_path = sample_paths[sample_choice]
            if os.path.exists(chosen_path):
                with open(chosen_path, "rb") as f:
                    raw_audio_bytes = f.read()
                audio_source_label = os.path.basename(chosen_path)

        elif input_source == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload Audio (WAV, MP3, MP4, M4A, OGG, OPUS, FLAC)",
                type=["wav", "mp3", "mp4", "m4a", "ogg", "opus", "oga", "flac", "aac"],
            )
            if uploaded_file is not None:
                raw_audio_bytes = uploaded_file.getvalue()
                audio_source_label = uploaded_file.name

        elif input_source == "Record Live Microphone":
            st.caption("🎙️ **Microphone note**: *This analyzes the recording after stop. It is not continuous real-time streaming yet.*")
            mic_capture = st.audio_input("Record Voice Sample")
            if mic_capture is not None:
                raw_audio_bytes = mic_capture.getvalue()
                audio_source_label = "Live_Microphone_Capture.wav"

    with col_input2:
        st.subheader("2. Playback & Action")
        if raw_audio_bytes is not None:
            st.audio(raw_audio_bytes)
            analyze_clicked = st.button("🚀 Analyze Voice Authenticity", type="primary", use_container_width=True)
        else:
            st.info("Awaiting audio input. Please select a sample, upload a file, or record via microphone.")
    # Invalidate stale results if user changes the selected/uploaded audio file
    if raw_audio_bytes is not None:
        if st.session_state.get("active_source_label") != audio_source_label and not analyze_clicked:
            st.session_state.pop("current_audio_bytes", None)
            st.session_state.pop("explain_res", None)
        st.session_state["active_source_label"] = audio_source_label

    # Store in session state when user triggers analysis
    if raw_audio_bytes is not None and analyze_clicked:
        st.session_state["current_audio_bytes"] = raw_audio_bytes
        st.session_state["current_audio_label"] = audio_source_label

    if "current_audio_bytes" in st.session_state:
        current_bytes = st.session_state["current_audio_bytes"]

        with st.spinner("Extracting 42 acoustic features and signal diagnostics..."):
            try:
                # In-memory decoding & validation (Zero Disk Retention)
                curr_label = st.session_state.get("current_audio_label", "audio.wav")
                curr_ext = os.path.splitext(curr_label)[1].lower() or ".wav"
                audio_arr, sr = load_audio_from_bytes(current_bytes, target_sr=SAMPLE_RATE, file_ext=curr_ext)
                metadata = get_audio_metadata(audio_arr, sr)

                # Predict probabilities & calibrated risk score
                if trained_pipeline_obj := get_model():
                    tuned_threshold = (
                        model_metadata.get("optimal_decision_threshold", 0.50)
                        if model_metadata
                        else 0.50
                    )
                    prediction_res = predict_and_score(
                        trained_pipeline_obj,
                        audio_arr,
                        sample_rate=sr,
                        decision_threshold=tuned_threshold,
                    )

                    # Build explainability package
                    train_mean = np.array(model_metadata.get("train_feature_mean")) if model_metadata and "train_feature_mean" in model_metadata else None
                    train_std = np.array(model_metadata.get("train_feature_std")) if model_metadata and "train_feature_std" in model_metadata else None
                    explain_res = build_explainability_report(
                        trained_pipeline_obj,
                        audio_arr,
                        sr,
                        prediction_res,
                        train_mean=train_mean,
                        train_std=train_std,
                    )
                    st.session_state["explain_res"] = explain_res
                    st.session_state["last_audio_arr"] = audio_arr
                    st.session_state["last_sr"] = sr
                else:
                    st.error("Model is not loaded. Please train model with `python scripts/train_model.py` first.")
                    st.stop()

            except ValueError as ve:
                st.error(f"⚠️ **Validation Notice**: {ve}")
                st.stop()
            except Exception as ex:
                st.error(f"❌ **Error analyzing audio**: {ex}")
                st.stop()

        st.divider()

        # Section 3: Detection Results & Risk Assessment
        st.subheader("3. Detection Results & Risk Assessment")

        # Out-Of-Distribution Warning Banner if anomalous
        if explain_res.get("is_out_of_distribution"):
            st.markdown(
                f"""
                <div class="ood-box">
                    ⚠️ <b>Acoustic Anomaly Notice</b>: {explain_res.get('ood_message')}<br>
                    <i>Confidence is not calibrated on audio with extreme background distortion or non-voice signals.</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            st.markdown(f"### Indication: **{prediction_res['prediction_label']}**")
            
            # Risk Bands
            band = prediction_res["risk_band"]
            if band == "Low":
                st.markdown('<span class="badge-low">🟢 Low Risk (0–25)</span>', unsafe_allow_html=True)
            elif band == "Review required":
                st.markdown('<span class="badge-review">🟡 Review Required (26–65)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-high">🔴 High Risk (66–100)</span>', unsafe_allow_html=True)

            # Uncertainty Notice if 0.40 <= Spoof Prob <= 0.60
            if explain_res.get("is_uncertain"):
                st.markdown('<br><span class="badge-uncertain">⚠️ UNCERTAIN — MANUAL REVIEW REQUIRED</span>', unsafe_allow_html=True)

            st.write(f"**Status Guidance**: {prediction_res['risk_description']}")
            st.caption(f"Applied Decision Threshold: `{prediction_res['decision_threshold_used']:.3f}` | Distance from Threshold: `{explain_res['threshold_distance']:+.3f}`")

        with res_col2:
            st.metric(
                "Spoof Risk Score",
                f"{prediction_res['risk_score']} / 100",
                help="0 = Natural Human Signal, 100 = High Risk Synthetic Signal",
            )
            st.progress(prediction_res["risk_score"] / 100.0)

        with res_col3:
            st.metric("Human Voice Probability", f"{prediction_res['human_probability'] * 100:.1f}%")
            st.metric("Spoof / AI Probability", f"{prediction_res['spoof_probability'] * 100:.1f}%")

        # Operational Verification Recommendations
        st.markdown("#### 📋 Analyst Advisory Recommendations")
        for rec in prediction_res["recommendations"]:
            st.write(f"• {rec}")

        # Section 4: Acoustic Feature Summary & Quality Status
        st.subheader("4. Acoustic Feature Summary & Quality Status")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Audio Quality", f"{explain_res['signal_diagnostics'].get('audio_quality', 'Normal')}")
        m_col2.metric("Duration", f"{metadata['duration_seconds']:.2f} sec")
        m_col3.metric("Sample Rate", f"{metadata['sample_rate']} Hz")
        m_col4.metric("RMS Energy", f"{metadata['rms_energy']:.5f}")
        m_col5.metric("Zero Crossing Rate", f"{metadata['zero_crossing_rate']:.4f}")

        # Waveform Visualization
        st.subheader("5. Acoustic Signal Waveform")
        st.line_chart(audio_arr[: min(len(audio_arr), sr * 5)], height=180)

        # Operational Advisory & Limitation Box
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>Visible Disclaimer</b>: <i>“Experimental decision-support prototype; not identity proof.”</i><br><br>
                ⚠️ <b>Limitation Warning</b>: Results are advisory signals based on statistical acoustic features. Unseen generative vocoders, acoustic room reverberation, or telephony compression require human-in-the-loop review.<br><br>
                🔒 <b>Manual Verification Recommendation</b>: Never automatically block transactions or calls. For suspicious or borderline scores, verify caller identity via secondary out-of-band channels (callback to registered phone number or cryptographic passkey).<br><br>
                🛡️ <b>Privacy Guarantee</b>: Zero raw audio history retained. No external alerts or automated enforcement.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# TAB 2: Explainability & Signal Diagnostics
# -----------------------------------------------------------------------------
with tab_explain:
    st.subheader("🔬 Signal Diagnostics & Feature Explainability")
    st.caption("Transparent, non-causal acoustic analysis to help security operators understand model signal cues.")

    st.markdown(
        f"""
        <div class="disclaimer-box">
            ℹ️ <b>Explainability Protocol</b>: {EXPLAINABILITY_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if "explain_res" in st.session_state:
        exp = st.session_state["explain_res"]
        diag = exp.get("signal_diagnostics", {})

        # Per-File Signal Diagnostics Cards
        st.markdown("#### 1. Per-File Prosodic, Spectral & Quality Diagnostics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Audio Quality", f"{diag.get('audio_quality', 'Normal')}")
        d2.metric("Sample Rate", f"{diag.get('sample_rate', 16000)} Hz")
        d3.metric("Silence Ratio", f"{diag.get('silence_ratio', 0.0) * 100:.1f}%")
        d4.metric("Clipping Ratio", f"{diag.get('clipping_ratio', 0.0) * 100:.2f}%")

        d5, d6, d7, d8 = st.columns(4)
        pitch_str = f"{diag.get('pitch_mean_hz')} Hz (±{diag.get('pitch_std_hz')} Hz)" if diag.get('pitch_mean_hz') else "Unvoiced / Aperiodic"
        d5.metric("Pitch (F0)", pitch_str)
        d6.metric("Pitch Variance", f"{diag.get('pitch_variation', 'N/A')}")
        d7.metric("Energy Variance", f"{diag.get('energy_std', 0.0):.5f}")
        d8.metric("Threshold Distance", f"{exp.get('distance_from_threshold', 0.0):+.3f}")

        st.caption(f"**Spectral Summary**: {diag.get('spectral_summary', 'N/A')}")

        st.divider()

        # Top 5 Feature Groups by Importance
        col_grp, col_tbl = st.columns([1, 1.3])

        with col_grp:
            st.markdown("#### 2. Canonical Feature Groups (Random Forest Global Importance)")
            top_groups = exp.get("top_feature_groups", [])
            if top_groups:
                df_groups = pd.DataFrame(top_groups)
                df_groups["importance_percent"] = df_groups["importance_share"] * 100
                st.dataframe(
                    df_groups[["category", "feature_group", "importance_percent"]].rename(
                        columns={"category": "Category", "feature_group": "Feature Group", "importance_percent": "Importance Share (%)"}
                    ),
                    use_container_width=True,
                )
                st.caption("Aggregated from Gini impurity reduction across all decision trees in Baseline v1.")

        with col_tbl:
            st.markdown("#### 3. Acoustic Signal Calibration Status")
            st.write(f"• **Calibration State**: `{exp.get('confidence_status', exp.get('calibration_status'))}`")
            st.write(f"• **Uncertainty Rating**: `{exp.get('uncertainty_banner')}`")
            st.write(f"• **Distributional Fit**: `{exp.get('ood_message')}`")
            st.write(f"• **Model Applied Threshold**: `{exp.get('decision_threshold'):.3f}`")
            st.write(f"• **Calculated Spoof Probability**: `{exp.get('spoof_probability', 0.0) * 100:.1f}%`")

        st.divider()

        # Structured Feature Summary Table
        st.markdown("#### 4. Detailed Feature Summary Table (Categorized)")
        feature_rows = exp.get("feature_summary_table", [])
        if feature_rows:
            df_feat_summary = pd.DataFrame(feature_rows).rename(
                columns={
                    "category": "Category",
                    "feature_group": "Acoustic Feature Group",
                    "value": "Measured Value",
                    "reference_range": "Normal Reference Range",
                    "interpretation": "Physical & Auditory Interpretation",
                }
            )
            st.dataframe(df_feat_summary, use_container_width=True)

    else:
        st.info("💡 Run an audio analysis in the 'Voice Authenticity Inspector' tab first to populate real-time explainability diagnostics.")

        # Show global importance preview even before inference
        if trained_model is not None:
            st.markdown("#### Global Feature Groups (Baseline v1)")
            _, global_groups_df = get_global_feature_importance(trained_model)
            st.dataframe(global_groups_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Evaluation Summary Loaded from reports/metrics.json
# -----------------------------------------------------------------------------
with tab_reports:
    st.subheader("📊 Independent Test Set Benchmark Evaluation")
    st.caption("Results loaded dynamically from `reports/metrics.json` (Evaluated on untouched out-of-sample test files).")

    if eval_metrics is None:
        st.warning("⚠️ No evaluation report found at `reports/metrics.json`. Please run `python scripts/evaluate_model.py` first.")
    else:
        ov = eval_metrics.get("overall_metrics", {})
        pcm = eval_metrics.get("per_class_metrics", {})

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{ov.get('accuracy', 0.0) * 100:.2f}%")
        c2.metric("Macro F1-Score", f"{ov.get('macro_f1', 0.0):.4f}")
        c3.metric("ROC-AUC Score", f"{ov.get('roc_auc', 0.0):.4f}")
        c4.metric("Untouched Test Samples", f"{eval_metrics.get('total_test_samples', 0)}")

        st.divider()

        col_cm, col_pc = st.columns([1, 1])

        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm_dict = eval_metrics.get("confusion_matrix", {})
            cm_matrix = cm_dict.get("matrix_2x2", [[0, 0], [0, 0]])

            cm_df = pd.DataFrame(
                cm_matrix,
                index=["Actual Bona Fide (0)", "Actual Spoof (1)"],
                columns=["Predicted Bona Fide (0)", "Predicted Spoof (1)"],
            )
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

            if os.path.exists(CONFUSION_MATRIX_PNG):
                st.image(CONFUSION_MATRIX_PNG, caption="Confusion Matrix Visual Plot", width=400)

        with col_pc:
            st.markdown("#### Per-Class Metrics")
            class_data = [
                {
                    "Class": "Bona Fide Human Voice (0)",
                    "Precision": f"{pcm.get('bona_fide', {}).get('precision', 0.0) * 100:.1f}%",
                    "Recall": f"{pcm.get('bona_fide', {}).get('recall', 0.0) * 100:.1f}%",
                    "F1-Score": f"{pcm.get('bona_fide', {}).get('f1_score', 0.0):.4f}",
                },
                {
                    "Class": "Spoof Synthetic Voice (1)",
                    "Precision": f"{pcm.get('spoof', {}).get('precision', 0.0) * 100:.1f}%",
                    "Recall": f"{pcm.get('spoof', {}).get('recall', 0.0) * 100:.1f}%",
                    "F1-Score": f"{pcm.get('spoof', {}).get('f1_score', 0.0):.4f}",
                },
            ]
            st.dataframe(pd.DataFrame(class_data), use_container_width=True)

            st.info(
                f"**Applied Decision Threshold**: `{eval_metrics.get('decision_threshold_used', 0.50):.3f}`\n"
                f"**False Positive Rate (FPR)**: `{ov.get('false_positive_rate', 0.0) * 100:.1f}%`\n"
                f"**False Negative Rate (FNR)**: `{ov.get('false_negative_rate', 0.0) * 100:.1f}%`"
            )

        with st.expander("📋 View Per-File Test Set Predictions Audit"):
            if "per_file_results" in eval_metrics:
                st.dataframe(pd.DataFrame(eval_metrics["per_file_results"]), use_container_width=True)

        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>Research Notice</b>: {eval_metrics.get('production_reliability_disclaimer', STATUTORY_DISCLAIMER)}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# TAB 4: Sandbox Live Call Streaming Simulator (160ms Window / 40ms Stride)
# -----------------------------------------------------------------------------
with tab_stream:
    st.subheader("📡 Sandbox Audio Streaming Simulator (Local Prerecorded WAV)")
    st.caption("Simulates real-time chunked audio processing over 160 ms windows with 40 ms stride.")

    st.markdown(
        """
        <div style="background:#1e1b4b; border:1px solid #6366f1; padding:14px 18px; border-radius:8px; margin-bottom:16px; color:#e0e7ff;">
            ⚠️ <b>SANDBOX SIMULATION — NOT A LIVE CALL</b><br>
            • <i>“This simulation demonstrates the processing flow only. It is not a telecom integration or production latency benchmark.”</i><br>
            • Operates exclusively on local prerecorded WAV files in memory.<br>
            • Zero live telecom interception, zero SIP/RTP capture hooks.<br>
            • Demonstrates rolling Exponential Moving Average (EMA) risk aggregation for call center SOC integration.<br>
            • 🔒 <b>Privacy Guarantee</b>: Zero raw audio history saved.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st_col1, st_col2 = st.columns([1.1, 1])

    with st_col1:
        stream_sample = st.selectbox(
            "Select Prerecorded Call Sample to Stream",
            [
                "Synthetic AI Impersonation Call (data/test/ai_voice/1.wav)",
                "Synthetic AI Spoof Attack (data/test/ai_voice/2.wav)",
                "Genuine Human Customer Call (data/test/human/01.wav)",
                "Genuine Human Executive Call (data/test/human/02.wav)",
            ],
        )
        stream_map = {
            "Synthetic AI Impersonation Call (data/test/ai_voice/1.wav)": "data/test/ai_voice/1.wav",
            "Synthetic AI Spoof Attack (data/test/ai_voice/2.wav)": "data/test/ai_voice/2.wav",
            "Genuine Human Customer Call (data/test/human/01.wav)": "data/test/human/01.wav",
            "Genuine Human Executive Call (data/test/human/02.wav)": "data/test/human/02.wav",
        }
        chosen_sim_file = stream_map[stream_sample]

    with st_col2:
        max_chunks_to_run = st.slider("Number of 160ms Windows to Process", min_value=5, max_value=50, value=25)
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            start_sim_btn = st.button("▶️ Start Simulation", type="primary", use_container_width=True)
        with c_btn2:
            stop_sim_btn = st.button("⏹️ Stop Simulation", use_container_width=True)

    if start_sim_btn:
        from scripts.simulate_stream import run_stream_simulation
        
        st.markdown("#### Live Streaming Analysis Feed")
        progress_bar = st.progress(0.0)

        sim_results = run_stream_simulation(
            audio_path=chosen_sim_file,
            window_ms=160,
            stride_ms=40,
            max_windows=max_chunks_to_run,
            simulated_delay_sec=0.01,
        )

        df_sim = pd.DataFrame(sim_results)
        if not df_sim.empty:
            progress_bar.progress(1.0)
            
            final_roll = df_sim["rolling_risk_score"].iloc[-1]
            final_band = df_sim["risk_band"].iloc[-1]
            avg_proc = df_sim["processing_ms"].mean()
            skipped_cnt = int((~df_sim["is_valid"]).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Rolling Risk Score", f"{final_roll:.1f} / 100")
            m2.metric("Operational Risk Band", final_band)
            m3.metric("Avg Window Latency", f"{avg_proc:.2f} ms")
            m4.metric("Windows / Skipped", f"{len(df_sim)} / {skipped_cnt}")

            # Line chart of rolling risk over time
            st.markdown("##### Real-Time Rolling Risk Timeline")
            chart_df = df_sim[["timestamp_sec", "rolling_risk_score", "instantaneous_spoof_prob"]].copy()
            chart_df["instant_score_pct"] = chart_df["instantaneous_spoof_prob"] * 100.0
            chart_df = chart_df.rename(columns={"rolling_risk_score": "Rolling Risk Score (EMA)", "instant_score_pct": "Instant Spoof %"}).set_index("timestamp_sec")
            st.line_chart(chart_df[["Rolling Risk Score (EMA)", "Instant Spoof %"]])

            # Manual verification recommendation
            st.markdown(
                """
                <div class="disclaimer-box">
                    🔒 <b>Analyst Advisory Recommendation</b>: If rolling score enters <i>Review required</i> or <i>High risk</i>, request secondary out-of-band verification (callback or passkey). Never automatically block calls or transactions based on automated stream signals.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("📋 View Real-Time Window Analysis Log"):
                st.dataframe(df_sim, use_container_width=True)