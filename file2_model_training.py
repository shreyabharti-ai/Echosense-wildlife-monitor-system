import os
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,accuracy_score)

warnings.filterwarnings("ignore")

SAMPLE_RATE    = 22050
CLIP_DURATION  = 5.0
HOP_LENGTH     = 512
N_FFT          = 2048
N_MELS         = 128
N_MFCC         = 40

RAW_AUDIO_DIR  = "raw_audio"
PROCESSED_DIR  = "processed"
MODEL_DIR      = "echosense_model"
PLOTS_DIR      = "training_plots"

for d in [PROCESSED_DIR, MODEL_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


def load_audio(filepath):
    """Load, mono-convert, resample, and pad/trim to fixed duration."""
    try:
        audio, _ = librosa.load(str(filepath), sr=SAMPLE_RATE,
                                mono=True, duration=CLIP_DURATION)
    except Exception as e:
        return None

    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]
    return audio


def spectral_subtract(audio):
    """Hand-crafted spectral subtraction noise reduction."""
    stft      = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_est  = np.mean(mag[:, :10], axis=1, keepdims=True)
    mag_clean  = np.maximum(mag - noise_est, 0.0)
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_out  = librosa.istft(stft_clean, hop_length=HOP_LENGTH)
    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(audio_out) < target_len:
        audio_out = np.pad(audio_out, (0, target_len - len(audio_out)))
    return audio_out[:target_len]


def remove_silence(audio, top_db=25):
    """Trim leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(trimmed) < target_len:
        trimmed = np.pad(trimmed, (0, target_len - len(trimmed)))
    return trimmed[:target_len]


def has_signal(audio, threshold=0.001):
    """Reject near-silent clips."""
    return float(np.sqrt(np.mean(audio ** 2))) > threshold


def clean_audio(filepath):
    """Full cleaning pipeline. Returns numpy array or None."""
    audio = load_audio(filepath)
    if audio is None:
        return None
    audio = spectral_subtract(audio)
    audio = remove_silence(audio)
    if not has_signal(audio):
        return None
    return audio

def feat_mfcc(audio):
    """40 MFCCs × 4 stats = 160 features."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([np.mean(mfcc, 1), np.std(mfcc, 1),
                            np.max(mfcc, 1),  np.min(mfcc, 1)])  


def feat_chroma(audio):
    """12 chroma × 2 stats = 24 features."""
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([np.mean(chroma, 1), np.std(chroma, 1)])


def feat_spectral(audio):
    """Spectral shape descriptors — ~22 features."""
    c  = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    ro = librosa.feature.spectral_rolloff(y=audio,  sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    zc = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    ct = librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return np.concatenate([
        [np.mean(c),  np.std(c)],
        [np.mean(bw), np.std(bw)],
        [np.mean(ro), np.std(ro)],
        [np.mean(zc), np.std(zc)],
        np.mean(ct, 1), np.std(ct, 1),
    ])  


def feat_tonnetz(audio):
    """Tonal centroid (tonnetz) — 6 × 2 = 12 features."""
    harm  = librosa.effects.harmonic(audio)
    tonn  = librosa.feature.tonnetz(y=harm, sr=SAMPLE_RATE)
    return np.concatenate([np.mean(tonn, 1), np.std(tonn, 1)])  


def extract_features(audio):
    """
    Full feature vector combining all descriptors.
    Total: 160 + 24 + 22 + 12 = ~218 features
    """
    return np.concatenate([
        feat_mfcc(audio),
        feat_chroma(audio),
        feat_spectral(audio),
        feat_tonnetz(audio),
    ])


def mel_spectrogram(audio):
    """Return log-mel spectrogram (128×T) for visualization."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=SAMPLE_RATE // 2
    )
    return librosa.power_to_db(mel, ref=np.max)

def build_dataset():
    """
    Walk raw_audio/<category>/<species>/ folders,
    clean audio, extract features, encode labels.
    Saves X.npy, y.npy, label_encoder.pkl, scaler.pkl
    """
    print("\n🔧 Building dataset...")
    X, y = [], []
    category_folders = [d for d in Path(RAW_AUDIO_DIR).iterdir() if d.is_dir()]

    for cat_folder in sorted(category_folders):
        category = cat_folder.name
        species_folders = [d for d in cat_folder.iterdir() if d.is_dir()]

        for sp_folder in sorted(species_folders):
            species    = sp_folder.name.replace("_", " ")
            audio_files = (list(sp_folder.glob("*.mp3")) +
                           list(sp_folder.glob("*.wav")) +
                           list(sp_folder.glob("*.ogg")))

            kept = 0
            for fp in audio_files:
                audio = clean_audio(fp)
                if audio is None:
                    continue
                feats = extract_features(audio)
                if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
                    continue
                X.append(feats)
                y.append(f"{category}::{species}")
                kept += 1

            print(f"  [{category}] {species}: {kept}/{len(audio_files)} kept")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    np.save(f"{PROCESSED_DIR}/X.npy", X_scaled)
    np.save(f"{PROCESSED_DIR}/y.npy", y_enc)

    with open(f"{PROCESSED_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(f"{PROCESSED_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Dataset: {X.shape[0]} samples | {len(le.classes_)} classes")
    print(f"   Feature vector size: {X.shape[1]}")
    return X_scaled, y_enc, le, scaler


def train_model(X, y, le):
    """
    Train 3-model soft-voting ensemble:
    RandomForest + SVM(RBF) + GradientBoosting
    """
    print("\n🏋️  Training EchoSense Ensemble...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_split=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Cross-validation (5-fold):")
    print("  " + "─" * 45)

    for name, model in models.items():
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(accuracy_score(y_train[val_idx],
                                          model.predict(X_train[val_idx])))
        mu, sigma = np.mean(scores), np.std(scores)
        print(f"  {name:<25}: {mu:.4f} ± {sigma:.4f}")

    print("\n  Training on full train split...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✅ {name}")
        ─
    def ensemble_predict_proba(X_in):
        proba = None
        for m in models.values():
            p = m.predict_proba(X_in)
            proba = p if proba is None else proba + p
        return proba / len(models)

    proba      = ensemble_predict_proba(X_test)
    y_pred_ens = np.argmax(proba, axis=1)
    acc        = accuracy_score(y_test, y_pred_ens)

    print(f"\n  🎯 Ensemble Test Accuracy: {acc:.4f} ({acc*100:.1f}%)\n")
    print(classification_report(y_test, y_pred_ens,
                                  target_names=le.classes_, zero_division=0))

    _plot_confusion_matrix(y_test, y_pred_ens, le.classes_)

    _plot_feature_importance(models["RandomForest"])

    return models, X_test, y_test, le

def _plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(10, len(class_names) // 2)
    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="YlOrRd", linewidths=0.4)
    plt.title("EchoSense — Confusion Matrix", fontsize=14, pad=15)
    plt.ylabel("True Species")
    plt.xlabel("Predicted Species")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved → {path}")


def _plot_feature_importance(rf_model, top_k=25):
    imp  = rf_model.feature_importances_
    top  = np.argsort(imp)[-top_k:][::-1]
    plt.figure(figsize=(12, 5))
    plt.bar(range(top_k), imp[top], color="#2e7d32", alpha=0.85)
    plt.title(f"Top {top_k} Feature Importances (RandomForest)", fontsize=13)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Feature importance saved → {path}")


def save_model(models, scaler, le):
    """Persist all model components to echosense_model/"""
    for name, model in models.items():
        with open(f"{MODEL_DIR}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open(f"{MODEL_DIR}/classes.txt", "w") as f:
        for cls in le.classes_:
            f.write(cls + "\n")

    print(f"\n💾 Model saved → {MODEL_DIR}/")
    print(f"   Files: {', '.join(os.listdir(MODEL_DIR))}")


def load_model(model_dir=MODEL_DIR):
    """Load all model artifacts from disk. Returns (models, scaler, le)."""
    models = {}
    for name in ["RandomForest", "SVM_RBF", "GradientBoosting"]:
        path = Path(model_dir) / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return models, scaler, le


def predict(audio_input, models, scaler, le, confidence_threshold=0.40):
    """
    Predict species from audio.
    audio_input: filepath (str/Path) OR numpy array (already at SAMPLE_RATE)
    Returns dict with prediction, confidence, top3, mel_spec.
    """
    if isinstance(audio_input, (str, Path)):
        audio = clean_audio(audio_input)
    else:
        audio = audio_input 

    if audio is None:
        return {"error": "Could not process audio — file may be silent or corrupt."}

    feats = extract_features(audio)
    if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
        return {"error": "Feature extraction failed — invalid audio signal."}

    feats_scaled = scaler.transform([feats])

    proba = None
    for m in models.values():
        p = m.predict_proba(feats_scaled)
        proba = p if proba is None else proba + p
    avg_proba = (proba / len(models))[0]

    best_idx  = int(np.argmax(avg_proba))
    best_conf = float(avg_proba[best_idx])
    label     = le.classes_[best_idx]
    parts     = label.split("::")
    category  = parts[0] if len(parts) == 2 else "unknown"
    species   = parts[1] if len(parts) == 2 else label

    top3_idx  = np.argsort(avg_proba)[-3:][::-1]
    top3 = []
    for idx in top3_idx:
        lbl = le.classes_[idx]
        sp  = lbl.split("::")[-1] if "::" in lbl else lbl
        top3.append({"species": sp, "confidence": round(float(avg_proba[idx]), 4)})

    mel = mel_spectrogram(audio)

    if best_conf < confidence_threshold:
        return {
            "prediction":  "Uncertain",
            "species":     "Unknown",
            "category":    "Unknown",
            "confidence":  round(best_conf, 4),
            "top3":        top3,
            "mel_spec":    mel,
            "action":      "Low confidence — saved for review",
        }

    return {
        "prediction":  label,
        "species":     species,
        "category":    category,
        "confidence":  round(best_conf, 4),
        "top3":        top3,
        "mel_spec":    mel,
        "action":      "Logged to dashboard",
    }


if __name__ == "__main__":
  
    X, y, le, scaler = build_dataset()

   
    models, X_test, y_test, le = train_model(X, y, le)

    save_model(models, scaler, le)

    print("\n✅ Training complete! Run file5_backend.py next.\n")