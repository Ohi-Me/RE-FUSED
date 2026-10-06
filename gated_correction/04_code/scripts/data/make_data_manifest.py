"""Write 03_data/docs/DATA_MANIFEST.csv: every raw and processed data file with size, SHA-256 of the bytes, row count
and a content hash that does not depend on the parquet writer version (pandas.util.hash_pandas_object over rows).

NYISO raw files are monthly parquet conversions of the MIS zip archives (fetch_nyiso.py); NYISO may revise archives,
so a re-download can legitimately differ -- compare the content hash per month to locate any revision.
"""
import glob
import hashlib
import os
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "foreign_data_study", "03_data", "docs", "DATA_MANIFEST.csv")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(path):
    if not path.endswith(".parquet"):
        return "", ""
    df = pd.read_parquet(path)
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()
    return len(df), h


def main():
    rows = []
    patterns = ["03_data/raw/nyiso/*/*.parquet", "03_data/raw/public/*.csv", "03_data/raw/public/*.zip",
                "03_data/processed/*.parquet"]
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(ROOT, "foreign_data_study", pat))):
            n, ch = content_hash(f)
            rel = os.path.relpath(f, os.path.join(ROOT, "foreign_data_study")).replace("\\", "/")
            rows.append(dict(file=rel, bytes=os.path.getsize(f), sha256=sha(f), rows=n, content_sha256=ch,
                             retrieved_or_built_utc=datetime.fromtimestamp(os.path.getmtime(f), timezone.utc).strftime("%Y-%m-%d")))
            print(rel, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows), "files")


if __name__ == "__main__":
    sys.exit(main())
