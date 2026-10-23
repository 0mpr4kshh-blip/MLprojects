"""
Chop a long recording into fixed-length (3 s) WAV clips, one per spoken command.

Usage:
    python chop_commands.py recordings/left.wav
    python chop_commands.py recordings/*.wav --out dataset --plot

Each input file's name (without .wav) becomes the label, so record ONE file per
command (e.g. left.wav = you saying "left" 50 times with pauses in between).
Output: dataset/<label>/<label>_001.wav, _002.wav, ...
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

SR = 16000
FRAME, HOP = 400, 160            # 25 ms window, 10 ms hop


def find_segments(y, sensitivity, min_gap, min_dur, pad):
    """Return list of (start_sample, end_sample) for each detected utterance."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    noise_floor = np.percentile(rms, 10)
    thr = max(noise_floor * sensitivity, 0.02 * rms.max())
    active = rms > thr

    # frame runs -> intervals
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]

    # merge intervals separated by short pauses (keeps multi-syllable words whole)
    gap_frames = int(min_gap * SR / HOP)
    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    segs = []
    for s, e in merged:
        if (e - s) * HOP / SR < min_dur:      # too short: click / bump / cough
            continue
        s_samp = max(0, s * HOP - int(pad * SR))
        e_samp = min(len(y), e * HOP + int(pad * SR))
        segs.append((s_samp, e_samp))
    return segs, rms, thr


def to_fixed(seg, length):
    """Centre the segment in a zero-padded buffer of `length` samples."""
    if len(seg) >= length:                    # too long: keep the loudest window
        env = np.convolve(seg ** 2, np.ones(SR // 10), mode="same")
        c = int(np.argmax(env))
        s = int(np.clip(c - length // 2, 0, len(seg) - length))
        return seg[s:s + length], True
    out = np.zeros(length, dtype=np.float32)
    off = (length - len(seg)) // 2
    out[off:off + len(seg)] = seg
    return out, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="input recording(s)")
    ap.add_argument("--out", default="dataset", help="output root folder")
    ap.add_argument("--duration", type=float, default=3.0, help="clip length (s)")
    ap.add_argument("--sensitivity", type=float, default=4.0,
                    help="speech threshold = noise floor x this (lower = more sensitive)")
    ap.add_argument("--min-gap", type=float, default=0.35,
                    help="pauses shorter than this (s) don't split a command")
    ap.add_argument("--min-dur", type=float, default=0.15,
                    help="ignore sounds shorter than this (s)")
    ap.add_argument("--pad", type=float, default=0.10,
                    help="extra audio kept before/after each command (s)")
    ap.add_argument("--plot", action="store_true",
                    help="save a PNG per file showing detections (for tuning)")
    a = ap.parse_args()

    length = int(a.duration * SR)
    for f in a.files:
        f = Path(f)
        label = f.stem
        y, _ = librosa.load(f, sr=SR, mono=True)
        segs, rms, thr = find_segments(y, a.sensitivity, a.min_gap, a.min_dur, a.pad)

        out_dir = Path(a.out) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        too_long = 0
        for i, (s, e) in enumerate(segs, 1):
            clip, trimmed = to_fixed(y[s:e], length)
            too_long += trimmed
            sf.write(out_dir / f"{label}_{i:03d}.wav", clip, SR, subtype="PCM_16")
        print(f"{f.name}: {len(segs)} clips -> {out_dir}"
              + (f"  ({too_long} longer than {a.duration}s were trimmed)" if too_long else ""))

        if a.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.arange(len(y)) / SR
            fig, ax = plt.subplots(figsize=(14, 3))
            ax.plot(t, y, lw=0.4)
            for s, e in segs:
                ax.axvspan(s / SR, e / SR, color="green", alpha=0.3)
            ax.set_title(f"{f.name}: {len(segs)} detections")
            fig.tight_layout()
            fig.savefig(out_dir / f"{label}_detections.png", dpi=110)
            plt.close(fig)


if __name__ == "__main__":
    main()
