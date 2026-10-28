import numpy as np
import librosa
from config import SR

def _stats(m):
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])

def extract_features(y, sr=SR):
    """Fixed-length feature vector (mean+std of each frame-level feature)."""
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    S = np.abs(librosa.stft(y))
    parts = [
        mfcc, delta,
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr),
        librosa.feature.spectral_contrast(S=S, sr=sr),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(S=S),
    ]
    return np.concatenate([_stats(p) for p in parts]).astype(np.float32)

def load_audio(path, seconds=None):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=seconds)
    return y
