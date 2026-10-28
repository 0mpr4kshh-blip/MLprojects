"""Live microphone classification. Ctrl+C to stop."""
import numpy as np, joblib, sounddevice as sd
from config import MODEL_PATH, SR
from features import extract_features

WINDOW_S, HOP_S, SILENCE_RMS = 3.0, 1.0, 0.005
model = joblib.load(MODEL_PATH)
buf = np.zeros(int(WINDOW_S * SR), dtype=np.float32)

print("Listening... (Ctrl+C to quit)")
try:
    while True:
        chunk = sd.rec(int(HOP_S * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        buf = np.concatenate([buf[len(chunk):], chunk[:, 0]])
        if np.sqrt(np.mean(buf ** 2)) < SILENCE_RMS:
            print("[silence]"); continue
        feats = extract_features(buf).reshape(1, -1)
        label = model.predict(feats)[0]
        scores = model.decision_function(feats)[0]
        conf = np.max(scores)
        print(f"{label:10s} (margin {conf:+.2f})")
except KeyboardInterrupt:
    print("\nStopped.")
