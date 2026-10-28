# Environmental Sound Classification (MFCC + PCA + SVM)

    pip install -r requirements.txt
    python download_data.py     # ESC-50
    python train.py             # features -> PCA -> SVM, saves model.joblib
    python realtime.py          # live mic demo

Optional: add speech by putting WAVs in data/extra/speech/ (e.g. from Mozilla
Common Voice or Google Speech Commands), delete features.npz, re-run train.py.
