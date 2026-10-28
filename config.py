from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ESC_DIR = DATA_DIR / "ESC-50-master"
EXTRA_DIR = DATA_DIR / "extra"          # optional: extra/<class_name>/*.wav (e.g. speech)
MODEL_PATH = ROOT / "model.joblib"
FEATURE_CACHE = ROOT / "features.npz"

SR = 22050
CLIP_SECONDS = 5

# Our target classes -> ESC-50 category names that feed them.
# (ESC-50 has no "traffic"/"speech" class, so vehicles are approximated and
#  speech comes from the optional EXTRA_DIR.)
CLASS_MAP = {
    "rain":    ["rain"],
    "birds":   ["chirping_birds", "crow", "rooster"],
    "vehicle": ["engine", "car_horn", "siren", "train", "helicopter"],
    "human":   ["laughing", "coughing", "crying_baby", "footsteps", "clapping"],
}
