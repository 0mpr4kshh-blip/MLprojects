"""Shared code: preprocessing, feature extraction and an SVD-based PCA."""
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

SR = 16000
N_FFT, HOP = 512, 160            # 32 ms window, 10 ms hop
MIN_LEN = int(0.25 * SR)


# ------------------------------------------------------------------ preprocessing
def bandpass(y, lo=100, hi=6000):
    """4th-order Butterworth band-pass: removes rumble/hum and hiss outside speech range."""
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, y)


def spectral_subtract(y, alpha=1.5, beta=0.1, n_fft=512, hop=128):
    """Background-noise removal by spectral subtraction.
    Noise profile = average spectrum of the quietest 15% of frames."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    k = max(2, int(0.15 * mag.shape[1]))
    quiet = np.argsort(mag.mean(axis=0))[:k]
    noise = mag[:, quiet].mean(axis=1, keepdims=True)
    clean = np.maximum(mag - alpha * noise, beta * mag)      # beta = spectral floor
    return librosa.istft(clean * np.exp(1j * phase), hop_length=hop, length=len(y))


def preprocess(y, denoise=True):
    """crop padding -> band-pass -> noise removal -> pre-emphasis -> peak normalise."""
    y = np.asarray(y, dtype=np.float32)
    nz = np.flatnonzero(y)                                   # chopper pads with exact zeros
    if nz.size == 0:
        return None
    y = y[nz[0]:nz[-1] + 1]
    if len(y) < MIN_LEN:
        y = np.pad(y, (0, MIN_LEN - len(y)))
    y = bandpass(y)
    if denoise:
        y = spectral_subtract(y)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])               # pre-emphasis
    peak = np.max(np.abs(y))
    return (y / peak if peak > 0 else y).astype(np.float32)


# ------------------------------------------------------------------ features
def extract_features(y):
    """MFCC (+delta) statistics, MFCC means over 3 time slices, and basic spectral stats."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=N_FFT, hop_length=HOP, n_mels=40)
    delta = librosa.feature.delta(mfcc, width=5)
    thirds = [t.mean(axis=1) for t in np.array_split(mfcc, 3, axis=1)]   # coarse temporal shape
    extra = [
        librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP),
        librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP),
        librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP),
        librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP),
    ]
    parts = [mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1), *thirds]
    parts += [np.array([e.mean(), e.std()]) for e in extra]
    return np.concatenate(parts).astype(np.float32)


def load_dataset(root, denoise=True):
    """dataset/<label>/*.wav  ->  X, y, filenames"""
    X, y, files = [], [], []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for w in sorted(cls_dir.glob("*.wav")):
            sig, _ = librosa.load(w, sr=SR, mono=True)
            p = preprocess(sig, denoise)
            if p is None:
                continue
            X.append(extract_features(p)); y.append(cls_dir.name); files.append(w.name)
    return np.array(X), np.array(y), files


# ------------------------------------------------------------------ PCA via SVD
class SVDPCA(BaseEstimator, TransformerMixin):
    """PCA implemented directly with the singular value decomposition.
    X_centered = U S Vt  ->  principal axes = rows of Vt, variance = S^2/(n-1).
    n_components: int = number of axes, float in (0,1) = fraction of variance to keep."""

    def __init__(self, n_components=0.95):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        var = S ** 2 / (len(X) - 1)
        self.all_ratio_ = var / var.sum()
        if isinstance(self.n_components, float) and self.n_components < 1:
            k = int(np.searchsorted(np.cumsum(self.all_ratio_), self.n_components) + 1)
        else:
            k = int(self.n_components)
        self.n_components_ = max(1, min(k, len(S)))
        self.components_ = Vt[:self.n_components_]
        self.explained_variance_ratio_ = self.all_ratio_[:self.n_components_]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T
