"""
Train + evaluate the spoken-command classifier.
    preprocessing -> MFCC features -> StandardScaler -> PCA (via SVD) -> SVM

    python train_commands.py --data dataset --out results
"""
import argparse, time
from pathlib import Path
import numpy as np, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from commands_ml import SVDPCA, load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", default="results")
ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--no-denoise", action="store_true", help="skip spectral subtraction")
a = ap.parse_args()

out = Path(a.out); out.mkdir(exist_ok=True)
lines = []
def log(msg=""):
    print(msg); lines.append(str(msg))

# 1. data -------------------------------------------------------------------
t0 = time.time()
X, y, _ = load_dataset(a.data, denoise=not a.no_denoise)
classes, counts = np.unique(y, return_counts=True)
log(f"Loaded {len(X)} clips, {X.shape[1]} features each ({time.time()-t0:.0f}s)")
log("Classes: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
log(f"Noise removal: {'OFF' if a.no_denoise else 'ON (band-pass + spectral subtraction)'}")
if len(classes) < 2 or counts.min() < 5:
    raise SystemExit("Need at least 2 classes with >=5 clips each.")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, stratify=y, random_state=42)
cv = StratifiedKFold(min(5, int(np.unique(ytr, return_counts=True)[1].min())), shuffle=True, random_state=0)
log(f"Train: {len(Xtr)}  Test: {len(Xte)}")

# 2. model + hyper-parameter search ------------------------------------------
pipe = Pipeline([("scale", StandardScaler()), ("pca", SVDPCA()), ("svm", SVC(class_weight="balanced"))])
grid = [
    {"pca__n_components": [0.90, 0.95, 0.99], "svm__kernel": ["rbf"],
     "svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.01, 0.001]},
    {"pca__n_components": [0.90, 0.95, 0.99], "svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
]
search = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1)
search.fit(Xtr, ytr)
best = search.best_estimator_
log(f"\nBest params: {search.best_params_}")
log(f"Cross-validated accuracy (train set): {search.best_score_:.3f}")
log(f"PCA kept {best['pca'].n_components_} of {X.shape[1]} dimensions")

# 3. evaluation on held-out test set -----------------------------------------
pred = best.predict(Xte)
acc = (pred == yte).mean()
log(f"\nTEST ACCURACY: {acc:.3f}\n")
log(classification_report(yte, pred, digits=3))
ConfusionMatrixDisplay.from_predictions(yte, pred, normalize="true", xticks_rotation=45, cmap="Blues")
plt.title(f"Confusion matrix (test acc {acc:.1%})"); plt.tight_layout()
plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()
cm = confusion_matrix(yte, pred, labels=classes)
for i in range(len(classes)):                                 # most common mistakes
    for j in range(len(classes)):
        if i != j and cm[i, j]:
            log(f"  '{classes[i]}' misheard as '{classes[j]}': {cm[i, j]}x")

# 4. performance analysis -----------------------------------------------------
pca = best["pca"]
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(pca.all_ratio_) + 1), np.cumsum(pca.all_ratio_), marker=".")
plt.axvline(pca.n_components_, color="r", ls="--", label=f"chosen = {pca.n_components_}")
plt.xlabel("principal components"); plt.ylabel("cumulative variance explained")
plt.title("PCA (SVD) variance"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(out / "pca_variance.png", dpi=120); plt.close()

Z = pca.transform(best["scale"].transform(Xtr))[:, :2]        # first two principal axes
plt.figure(figsize=(6, 5))
for c in classes:
    plt.scatter(*Z[ytr == c].T, s=14, label=c, alpha=.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Training data in PCA space")
plt.legend(); plt.tight_layout(); plt.savefig(out / "pca_scatter.png", dpi=120); plt.close()

ks = [k for k in (2, 3, 5, 8, 12, 20, 30, 50, 80) if k < X.shape[1]] + [X.shape[1]]
accs = [cross_val_score(clone(best).set_params(pca__n_components=k), Xtr, ytr, cv=cv).mean() for k in ks]
plt.figure(figsize=(6, 4)); plt.plot(ks, accs, marker="o")
plt.xlabel("PCA components"); plt.ylabel("CV accuracy"); plt.grid(alpha=.3)
plt.title("Accuracy vs. number of components"); plt.tight_layout()
plt.savefig(out / "accuracy_vs_components.png", dpi=120); plt.close()
log("\nCV accuracy vs PCA components: " + ", ".join(f"{k}:{s:.3f}" for k, s in zip(ks, accs)))

log("\nWith vs without dimensionality reduction (5-fold CV, train set):")
for name, model in [("with PCA   ", clone(best)), ("without PCA", clone(best).set_params(pca="passthrough"))]:
    t = time.perf_counter(); s = cross_val_score(model, Xtr, ytr, cv=cv)
    log(f"  {name}: acc {s.mean():.3f} (+/-{s.std():.3f}), time {time.perf_counter()-t:.2f}s")

# 5. save final model (refit on all data) -------------------------------------
final = clone(best).fit(X, y)
joblib.dump(final, out / "model.joblib")
(out / "report.txt").write_text("\n".join(lines))
log(f"\nSaved model + report + plots to {out}/")
