import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from config import *
from features import extract_features, load_audio

def build_dataset():
    if FEATURE_CACHE.exists():
        d = np.load(FEATURE_CACHE, allow_pickle=True)
        return d["X"], d["y"], d["g"]
    cat_to_cls = {c: k for k, cats in CLASS_MAP.items() for c in cats}
    meta = pd.read_csv(ESC_DIR / "meta" / "esc50.csv")
    meta = meta[meta.category.isin(cat_to_cls)]
    X, y, g = [], [], []
    for i, row in enumerate(meta.itertuples()):
        wav = load_audio(ESC_DIR / "audio" / row.filename, CLIP_SECONDS)
        X.append(extract_features(wav)); y.append(cat_to_cls[row.category]); g.append(row.fold)
        if i % 100 == 0: print(f"  ESC-50 {i}/{len(meta)}")
    # optional extra data, e.g. data/extra/speech/*.wav (fold assigned randomly)
    if EXTRA_DIR.exists():
        rng = np.random.default_rng(0)
        for cls_dir in EXTRA_DIR.iterdir():
            for f in cls_dir.glob("*.wav"):
                X.append(extract_features(load_audio(f, CLIP_SECONDS)))
                y.append(cls_dir.name); g.append(int(rng.integers(1, 6)))
    X, y, g = np.array(X), np.array(y), np.array(g)
    np.savez(FEATURE_CACHE, X=X, y=y, g=g)
    return X, y, g

X, y, g = build_dataset()
print("Dataset:", X.shape, dict(zip(*np.unique(y, return_counts=True))))

train, test = g != 5, g == 5            # fold 5 held out as the final test set
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(random_state=0)),        # PCA is computed via SVD internally
    ("svm", SVC(kernel="rbf", class_weight="balanced")),
])
grid = {
    "pca__n_components": [0.90, 0.95, 0.99],
    "svm__C": [1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001],
}
search = GridSearchCV(pipe, grid, cv=GroupKFold(4), n_jobs=-1, verbose=1)
search.fit(X[train], y[train], groups=g[train])
print("Best params:", search.best_params_, "CV acc: %.3f" % search.best_score_)

best = search.best_estimator_
pred = best.predict(X[test])
print(classification_report(y[test], pred))
ConfusionMatrixDisplay.from_predictions(y[test], pred, xticks_rotation=45)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120)

best.fit(X, y)                           # refit on everything for deployment
joblib.dump(best, MODEL_PATH)
print("Saved", MODEL_PATH)
