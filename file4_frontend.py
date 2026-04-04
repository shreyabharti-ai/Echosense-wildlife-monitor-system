import io
import time
import tempfile
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from file3_ui_design import (
    inject_css, hero_header, section_divider,
    confidence_bar, top3_bars, result_card,
    fact_box, stat_grid, echo_footer,
    CATEGORY_META, PALETTE,
)
from file5_backend import (
    load_echosense_model, run_prediction,
    get_waveform_data, is_model_ready,
)

st.set_page_config(
    page_title="EchoSense · Wildlife Sound ID",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

if "history" not in st.session_state:
    st.session_state.history = []   # list of prediction results
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 0.5rem;">
        <div style="font-size:2.4rem">🌿</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.72rem;
                    letter-spacing:0.15em; color:#6b9e74; text-transform:uppercase;">
            EchoSense
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    
        # Model loader
    st.markdown("**⚙️ Model**")

    model_dir = st.text_input(
        "Model directory",
        value="echosense_model",
        label_visibility="collapsed",
        placeholder="echosense_model/",
        help="Path to the folder saved by file2_model_training.py"
    )

    if st.button("🔄 Load Model", use_container_width=True):
        with st.spinner("Loading EchoSense..."):
            ok = load_echosense_model(model_dir)
            st.session_state.model_loaded = ok
            if ok:
                st.success("✅ Model loaded!")
            else:
                st.error("❌ Model not found. Run file2 first.")

    if st.session_state.model_loaded:
        st.markdown(
            '<span class="chip chip-green">✅ Model Ready</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="chip chip-red">⚠️ Not Loaded</span>',
            unsafe_allow_html=True
        )

    st.divider()

     # Settings
    st.markdown("**🎛️ Settings**")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.20, 0.90, 0.40, 0.05,
        help="Predictions below this score are marked as uncertain"
    )
    show_spectrogram = st.toggle("Show mel-spectrogram", value=True)
    show_waveform    = st.toggle("Show waveform",        value=True)
    show_top3        = st.toggle("Show top-3 predictions", value=True)

    st.divider()
    
     # History
    if st.session_state.history:
        st.markdown("**📋 Session History**")
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            meta = CATEGORY_META.get(h.get("category","").lower(),
                                      CATEGORY_META["unknown"])
            conf = int(h.get("confidence", 0) * 100)
            sp   = h.get("species", "Unknown")
            st.markdown(
                f'<div style="font-size:0.78rem; color:#6b9e74; '
                f'padding:4px 0; border-bottom:1px solid #2d5c35;">'
                f'{meta["emoji"]} {sp} <span style="color:#4ade80">{conf}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()

    # Supported species summary
    st.markdown("**🐾 Supported Categories**")
    for cat, meta in CATEGORY_META.items():
        if cat == "unknown":
            continue
        st.markdown(
            f'<span class="chip chip-green">{meta["emoji"]} {meta["label"]}</span>',
            unsafe_allow_html=True
        )

hero_header()

tab_predict, tab_about, tab_history = st.tabs(
    ["🎙️ Identify", "📖 About", "📋 History"]
)

with tab_predict:

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        section_divider("AUDIO INPUT")

        st.markdown("""
        <div class="upload-zone">
            <span class="upload-icon">🎵</span>
            <p class="upload-label">Drop your wildlife recording here</p>
            <p class="upload-hint">MP3 · WAV · OGG · FLAC · M4A</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload wildlife audio",
            type=["mp3", "wav", "ogg", "flac", "m4a"],
            label_visibility="collapsed",
            help="Upload a recording of any animal sound"
        )

        if uploaded_file is not None:
    
            section_divider("PLAYBACK")
            st.audio(uploaded_file, format=uploaded_file.type)

            file_bytes = uploaded_file.getvalue()
            file_kb    = len(file_bytes) / 1024

            try:
                audio_array, sr = librosa.load(
                    io.BytesIO(file_bytes),
                    sr=22050, mono=True, duration=30.0
                )
                duration = len(audio_array) / sr
                stat_grid(duration, sr, file_kb)
            except Exception as e:
                st.markdown(
                    f'<div class="echo-error">Could not read audio: {e}</div>',
                    unsafe_allow_html=True
                )
                audio_array = None
                duration    = 0.0
    # Waveform
            if show_waveform and audio_array is not None:
                section_divider("WAVEFORM")
                fig, ax = plt.subplots(figsize=(7, 2))
                fig.patch.set_facecolor("#122614")
                ax.set_facecolor("#122614")

                times = np.linspace(0, duration, len(audio_array))
                ax.fill_between(times, audio_array, alpha=0.7,
                                color=PALETTE["accent"])
                ax.plot(times, audio_array, color=PALETTE["accent"],
                        linewidth=0.6, alpha=0.9)
                ax.set_xlim(0, duration)
                ax.set_xlabel("Time (s)", color=PALETTE["text_muted"],
                              fontsize=8)
                ax.tick_params(colors=PALETTE["text_muted"], labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor(PALETTE["border"])
                ax.axhline(0, color=PALETTE["border"], linewidth=0.5,
                           linestyle="--")

                st.markdown('<div class="wave-container">', unsafe_allow_html=True)
                st.pyplot(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig)

            if show_spectrogram and audio_array is not None:
                section_divider("MEL-SPECTROGRAM")
                mel = librosa.feature.melspectrogram(
                    y=audio_array[:int(22050 * min(duration, 10))],
                    sr=22050, n_mels=128, n_fft=2048, hop_length=512
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                fig2, ax2 = plt.subplots(figsize=(7, 3))
                fig2.patch.set_facecolor("#122614")
                ax2.set_facecolor("#122614")

                img = librosa.display.specshow(
                    mel_db, sr=22050, hop_length=512,
                    x_axis="time", y_axis="mel",
                    ax=ax2, cmap="magma"
                )
                plt.colorbar(img, ax=ax2, format="%+2.0f dB",
                             label="dB").ax.yaxis.label.set_color(
                                 PALETTE["text_muted"])
                ax2.set_title("Mel Frequency Spectrogram",
                              color=PALETTE["text_muted"], fontsize=8, pad=6)
                ax2.tick_params(colors=PALETTE["text_muted"], labelsize=7)
                for spine in ax2.spines.values():
                    spine.set_edgecolor(PALETTE["border"])

                st.markdown('<div class="wave-container">', unsafe_allow_html=True)
                st.pyplot(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig2)

    with col_result:
        section_divider("IDENTIFICATION RESULT")

        if uploaded_file is None:
        
            st.markdown("""
            <div style="
                background: #122614;
                border: 1px dashed #2d5c35;
                border-radius: 22px;
                padding: 3.5rem 2rem;
                text-align: center;
                color: #6b9e74;
            ">
                <div style="font-size:3rem; margin-bottom:1rem">🌿</div>
                <div style="font-size:1rem; font-weight:600; color:#e2f5e4">
                    Awaiting audio input
                </div>
                <div style="font-size:0.82rem; margin-top:0.4rem">
                    Upload a recording to identify the species
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
         
            if not st.session_state.model_loaded:
                st.markdown("""
                <div class="echo-warning">
                    ⚠️ Model not loaded. Click <b>Load Model</b> in the sidebar first.
                </div>
                """, unsafe_allow_html=True)

            else:
                predict_btn = st.button(
                    "🔍 Identify Species",
                    use_container_width=True,
                    type="primary"
                )

                if predict_btn:
                    with st.spinner("🎙️ Analysing sound..."):
                        # Save to temp file for backend
                        suffix = "." + uploaded_file.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        result = run_prediction(
                            tmp_path,
                            confidence_threshold=confidence_threshold
                        )

                    if "error" in result:
                        st.markdown(
                            f'<div class="echo-error">❌ {result["error"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                    
                        result_card(result)

                        confidence_bar(result["confidence"])
\
                        fact_box(result.get("species",""))
\
                        if show_top3 and result.get("top3"):
                            section_divider("TOP 3 CANDIDATES")
                            top3_bars(result["top3"])
\
                        action = result.get("action","")
                        chip_cls = "chip-green" if "Logged" in action else "chip-amber"
                        st.markdown(
                            f'<span class="chip {chip_cls}">⚡ {action}</span>',
                            unsafe_allow_html=True
                        )

                        st.session_state.history.append(result)
                        if result["confidence"] >= 0.80:
                            st.balloons()

                            with tab_about:
    st.markdown("""
    <div style="max-width:700px; margin:1.5rem auto;">

    <div style="
        background:#122614; border:1px solid #2d5c35;
        border-radius:18px; padding:2rem; margin-bottom:1.5rem;
    ">
        <h3 style="color:#4ade80; margin:0 0 0.8rem">🌿 What is EchoSense?</h3>
        <p style="color:#e2f5e4; line-height:1.7; font-size:0.95rem">
        EchoSense is a biodiversity monitoring system that uses audio sensing
        on IoT field devices to detect, classify, and track wildlife —
        building a longitudinal picture of species presence, population
        patterns, and ecosystem health without disturbing the habitat.
        </p>
    </div>

    <div style="
        background:#122614; border:1px solid #2d5c35;
        border-radius:18px; padding:2rem; margin-bottom:1.5rem;
    ">
        <h3 style="color:#4ade80; margin:0 0 0.8rem">🔬 How It Works</h3>
        <div style="color:#e2f5e4; font-size:0.9rem; line-height:1.8;">
        <b style="color:#86efac">1. Audio Capture</b> — Microphones on solar IoT devices record 24/7<br>
        <b style="color:#86efac">2. Cleaning</b> — Spectral subtraction removes wind, rain, traffic noise<br>
        <b style="color:#86efac">3. Feature Extraction</b> — MFCCs, Chroma, Spectral Contrast, Tonnetz<br>
        <b style="color:#86efac">4. Classification</b> — Soft-voting ensemble (RF + SVM + GBM)<br>
        <b style="color:#86efac">5. Dashboard</b> — Species logged with timestamp + GPS location
        </div>
    </div>

    <div style="
        background:#122614; border:1px solid #2d5c35;
        border-radius:18px; padding:2rem; margin-bottom:1.5rem;
    ">
        <h3 style="color:#4ade80; margin:0 0 0.8rem">🐾 Supported Animal Groups</h3>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.88rem; color:#e2f5e4;">
            <div>🐦 <b>Birds</b> — 20 species</div>
            <div>🐸 <b>Frogs</b> — 15 species</div>
            <div>🦗 <b>Insects</b> — 12 species</div>
            <div>🦊 <b>Mammals</b> — 13 species</div>
        </div>
    </div>

    <div style="
        background:#122614; border:1px solid #2d5c35;
        border-radius:18px; padding:2rem;
    ">
        <h3 style="color:#4ade80; margin:0 0 0.8rem">📦 File Structure</h3>
        <pre style="
            background:#0d1f0e; border:1px solid #2d5c35;
            border-radius:10px; padding:1rem;
            color:#86efac; font-size:0.78rem; overflow-x:auto;
        ">file1_data_collection.py   # Download audio datasets
file2_model_training.py    # Clean, extract features, train
file3_ui_design.py         # CSS, theme tokens, HTML components
file4_frontend.py          # Streamlit app (this file)
file5_backend.py           # Model loading & prediction API
echosense_model/           # Saved model artifacts
raw_audio/                 # Downloaded audio files
processed/                 # Feature arrays (X.npy, y.npy)</pre>
    </div>

    </div>
    """, unsafe_allow_html=True)

with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#6b9e74;">
            <div style="font-size:2.5rem">📋</div>
            <p>No predictions yet this session.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="color:#6b9e74; font-family:'Space Mono',monospace;
                    font-size:0.75rem; margin-bottom:1rem;">
            {len(st.session_state.history)} IDENTIFICATION(S) THIS SESSION
        </div>
        """, unsafe_allow_html=True)

        for i, h in enumerate(reversed(st.session_state.history)):
            meta  = CATEGORY_META.get(h.get("category","").lower(),
                                       CATEGORY_META["unknown"])
            sp    = h.get("species", "Unknown")
            conf  = int(h.get("confidence", 0) * 100)
            color = meta["color"]

            st.markdown(f"""
            <div style="
                background:#122614; border:1px solid #2d5c35;
                border-radius:14px; padding:1rem 1.4rem;
                margin-bottom:10px; display:flex;
                align-items:center; gap:1rem;
            ">
                <span style="font-size:2rem">{meta['emoji']}</span>
                <div style="flex:1">
                    <div style="color:{color}; font-weight:700; font-size:1rem">
                        {sp}
                    </div>
                    <div style="color:#6b9e74; font-size:0.75rem;
                                font-family:'Space Mono',monospace;">
                        {meta['label']} · {conf}% confidence
                    </div>
                </div>
                <div style="
                    background:rgba(74,222,128,0.1);
                    border:1px solid rgba(74,222,128,0.2);
                    color:#4ade80; font-size:0.72rem;
                    font-family:'Space Mono',monospace;
                    padding:3px 10px; border-radius:20px;
                ">#{len(st.session_state.history) - i}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All", use_container_width=False):
            st.session_state.history = []
            st.rerun()

echo_footer()
