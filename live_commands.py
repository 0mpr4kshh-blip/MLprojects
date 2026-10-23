"""
Use the trained model on new audio.
    python live_commands.py --file test.wav      # classify every command found in a file
    python live_commands.py                       # live microphone (needs sounddevice)
Keep this file next to commands_ml.py and chop_commands.py.
"""
import argparse
import numpy as np, joblib, librosa
from commands_ml import preprocess, extract_features, SR
from chop_commands import find_segments

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="results/model.joblib")
ap.add_argument("--file")
ap.add_argument("--block", type=float, default=3.0, help="mic block length (s)")
a = ap.parse_args()
model = joblib.load(a.model)

def classify(seg):
    f = extract_features(preprocess(seg)).reshape(1, -1)
    return model.predict(f)[0], float(np.max(model.decision_function(f)))

if a.file:
    y, _ = librosa.load(a.file, sr=SR, mono=True)
    segs, _, _ = find_segments(y, 4.0, 0.35, 0.15, 0.10)
    for s, e in segs:
        label, margin = classify(y[s:e])
        print(f"{s / SR:6.2f}s  {label:12s} (margin {margin:+.2f})")
    print(f"{len(segs)} commands found")
else:
    import sounddevice as sd
    print("Listening... say a command (Ctrl+C to stop)")
    try:
        while True:
            blk = sd.rec(int(a.block * SR), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            y = blk[:, 0]
            if np.sqrt(np.mean(y ** 2)) < 0.003:
                continue
            segs, _, _ = find_segments(y, 4.0, 0.35, 0.15, 0.10)
            for s, e in segs:
                label, margin = classify(y[s:e])
                print(f"{label:12s} (margin {margin:+.2f})")
    except KeyboardInterrupt:
        print("\nStopped.")
