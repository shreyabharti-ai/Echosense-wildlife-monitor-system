import io
import pickle
import warnings
import numpy as np
from pathlib import Path

import librosa

warnings.filterwarnings("ignore")

_MODELS = {}
_SCALER = None
_LE = None
_READY = False


SAMPLE_RATE = 22050
CLIP_DURATION = 5.0
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 40


def load_echosense_model(model_dir="echosense_model"):
    global _MODELS, _SCALER, _LE, _READY

    
    model_path = Path(model_dir)
    required = [
        "RandomForest.pkl",
        "SVM_RBF.pkl",
        "GradientBoosting.pkl",
        "scaler.pkl",
        "label_encoder.pkl"
    ]

    missing = [f for f in required if not (model_path / f).exists()]
    if missing:
        _READY = False
        return False

    try:
        _MODELS = {}
        for name in ["RandomForest", "SVM_RBF", "GradientBoosting"]:
            with open(model_path / f"{name}.pkl", "rb") as f:
                _MODELS[name] = pickle.load(f)

        with open(model_path / "scaler.pkl", "rb") as f:
            _SCALER = pickle.load(f)

        with open(model_path / "label_encoder.pkl", "rb") as f:
            _LE = pickle.load(f)

        _READY = True
        return True

    except:
        _READY = False
        return False


def is_model_ready():
    return _READY


def _load_audio(source):
    try:
        if isinstance(source, (str, Path)):
            audio, _ = librosa.load(str(source), sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
        else:
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            audio, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    except:
        return None

    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio[:target].astype(np.float32)


def _spectral_subtract(audio):
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(stft), np.angle(stft)
    noise = np.mean(mag[:, :10], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - noise, 0.0)
    stft_clean = mag_clean * np.exp(1j * phase)
    out = librosa.istft(stft_clean, hop_length=HOP_LENGTH)

    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(out) < target:
        out = np.pad(out, (0, target - len(out)))
    return out[:target].astype(np.float32)

def _remove_silence(audio, top_db=25):
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(trimmed) < target:
        trimmed = np.pad(trimmed, (0, target - len(trimmed)))
    return trimmed[:target].astype(np.float32)


def _has_signal(audio, threshold=0.001):
    return float(np.sqrt(np.mean(audio ** 2))) > threshold


def preprocess_audio(source):
    audio = _load_audio(source)
    if audio is None:
        return None

    audio = _spectral_subtract(audio)
    audio = _remove_silence(audio)

    if not _has_signal(audio):
        return None

    return audio
def _mfcc(audio):
    m = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([m.mean(1), m.std(1), m.max(1), m.min(1)])


def _chroma(audio):
    c = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([c.mean(1), c.std(1)])


def _spectral(audio):
    cen = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    ro = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    zc = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    ct = librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    return np.concatenate([
        [cen.mean(), cen.std()],
        [bw.mean(), bw.std()],
        [ro.mean(), ro.std()],
        [zc.mean(), zc.std()],
        ct.mean(1), ct.std(1)
    ])


def _tonnetz(audio):
    harm = librosa.effects.harmonic(audio)
    tn = librosa.feature.tonnetz(y=harm, sr=SAMPLE_RATE)
    return np.concatenate([tn.mean(1), tn.std(1)])


def extract_features(audio):
    return np.concatenate([
        _mfcc(audio),
        _chroma(audio),
        _spectral(audio),
        _tonnetz(audio)
    ]).astype(np.float32)

def _soft_vote(features_scaled):
    proba = None
    for model in _MODELS.values():
        p = model.predict_proba(features_scaled)
        proba = p if proba is None else proba + p
    return (proba / len(_MODELS))[0]


def run_prediction(source, confidence_threshold=0.40):
    if not _READY:
        return {"error": "model not loaded"}

    audio = preprocess_audio(source)
    if audio is None:
        return {"error": "invalid audio"}

    try:
        feats = extract_features(audio)
    except:
        return {"error": "feature error"}

    if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
        return {"error": "bad features"}

    feats_scaled = _SCALER.transform([feats])
    avg_proba = _soft_vote(feats_scaled)

    best_idx = int(np.argmax(avg_proba))
    best_conf = float(avg_proba[best_idx])
    label = _LE.classes_[best_idx]

    parts = label.split("::")
    category = parts[0] if len(parts) == 2 else "unknown"
    species = parts[1] if len(parts) == 2 else label

    top3_idx = np.argsort(avg_proba)[-3:][::-1]
    top3 = []
    for idx in top3_idx:
        lbl = _LE.classes_[idx]
        sp = lbl.split("::")[-1] if "::" in lbl else lbl
        top3.append({
            "species": sp,
            "confidence": round(float(avg_proba[idx]), 4)
        })

    if best_conf < confidence_threshold:
        return {
            "prediction": "uncertain",
            "species": "unknown",
            "category": "unknown",
            "confidence": round(best_conf, 4),
            "top3": top3
        }

    return {
        "prediction": label,
        "species": species,
        "category": category,
        "confidence": round(best_conf, 4),
        "top3": top3
    }

def get_waveform_data(source, max_points=1000):
    audio = _load_audio(source)
    if audio is None:
        return np.array([]), np.array([])

    if len(audio) > max_points:
        step = len(audio) // max_points
        audio = audio[::step]

    duration = len(audio) / SAMPLE_RATE
    times = np.linspace(0, duration, len(audio))
    return times, audio


def batch_predict(file_paths, confidence_threshold=0.40):
    results = []
    for path in file_paths:
        result = run_prediction(path, confidence_threshold)
        result["file"] = str(path)
        results.append(result)
    return results
