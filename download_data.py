"""Downloads and unpacks ESC-50 (2000 clips, 50 classes, ~600 MB)."""
import io, zipfile, requests
from config import DATA_DIR, ESC_DIR

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

if ESC_DIR.exists():
    print("ESC-50 already present.")
else:
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading ESC-50 ...")
    r = requests.get(URL, timeout=600)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(DATA_DIR)
    print("Done ->", ESC_DIR)
