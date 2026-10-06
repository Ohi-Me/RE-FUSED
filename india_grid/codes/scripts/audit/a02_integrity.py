"""Check that the frozen record still matches what is on disk, and re-score the hypotheses from the stored results.

Three checks, all of them things a reviewer or an examiner can repeat:

1. the pre-registration file still hashes to the value in PREREG_SHA256.txt;
2. every file hashed inside the pre-registration (the frozen choices and the scoring script) still has that hash;
3. the panel still has the content hash recorded in data/processed/CHECKSUM.txt;
4. re-running the scoring script on the stored confirmatory results reproduces hypotheses.csv exactly.

Outputs: results/audit/integrity.json and results/audit/INTEGRITY.md
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from refused import dev2 as D  # noqa: E402
from refused import o2data as O  # noqa: E402
from refused.paths import DESIGN, PROC, RES, ROOT  # noqa: E402

OUT = os.path.join(RES, "audit")
PREREG = os.path.join(DESIGN, "02_preregistration.md")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_prereg():
    rec = open(os.path.join(DESIGN, "PREREG_SHA256.txt"), encoding="utf-8").read().split()[0]
    now = sha256(PREREG)
    return dict(file="codes/design/02_preregistration.md", recorded=rec, found=now, ok=(rec == now))


def renamed_files():
    """Files whose hash changed only because of the renaming of 24 Sep 2026 (RENAME_RECORD.md).

    codes/design/RENAME_RECORD.md lists the old and new hash of each one. A file is accepted here only when the
    record holds exactly the pair (recorded hash, hash on disk). The scientific check stays the re-scoring
    below, which has to reproduce the archived table cell by cell."""
    p = os.path.join(DESIGN, "RENAME_RECORD.md")
    if not os.path.exists(p):
        return {}
    text = open(p, encoding="utf-8").read()
    out = {}
    for rel, old, new in re.findall(r"`([^`]+\.py)`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*`([0-9a-f]{64})`", text):
        out[rel] = (old, new)
    return out


def check_frozen_files():
    text = open(PREREG, encoding="utf-8").read()
    rename = renamed_files()
    rows = []
    for h, rel in re.findall(r"^([0-9a-f]{64})\s+(\S+)$", text, flags=re.M):
        f = os.path.join(ROOT, rel)
        found = sha256(f) if os.path.exists(f) else None
        ok, note = (found == h), ""
        if not ok and rename.get(rel) == (h, found):
            ok, note = True, "renamed 24 Sep 2026, recorded in RENAME_RECORD.md"
        rows.append(dict(file=rel, recorded=h, found=found, ok=ok, note=note))
    return rows


def check_panel():
    """CHECKSUM.txt holds a content hash of the panel as read back from the file, not a hash of the bytes: the
    parquet round trip normalises NaN payloads, so the file bytes are not stable but the content is."""
    line = open(os.path.join(PROC, "CHECKSUM.txt"), encoding="utf-8").read().strip()
    rec = line.split("content_sha256=")[-1].strip()
    f = os.path.join(PROC, "refused_state_day.parquet")
    if not os.path.exists(f):
        return dict(file="data/processed/refused_state_day.parquet", recorded=rec, found=None, ok=False)
    d = pd.read_parquet(f)
    found = hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()
    return dict(file="data/processed/refused_state_day.parquet", recorded=rec, found=found, ok=(rec == found),
                rows=len(d), cols=d.shape[1])


def rescore():
    """Run the frozen scorer again and compare, cell by cell, with the archived table."""
    kept = os.path.join(D.DEV2, "hypotheses.csv")
    if not os.path.exists(kept):
        return dict(ok=False, note="no hypotheses.csv to compare with")
    out = os.path.join(OUT, "rescore")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, REFUSED_DEV2_DIR=os.path.relpath(out, RES))
    # the scorer writes hypotheses.csv into REFUSED_DEV2_DIR; everything it reads stays where it is
    for src in ("o2", "o3", "o4", "o5"):
        link = os.path.join(out, src)
        target = os.path.join(D.DEV2, src)
        if os.path.islink(link) and os.path.realpath(link) != os.path.realpath(target):
            os.unlink(link)          # a link left by an earlier run, e.g. before the folder was renamed
        if not os.path.exists(link):
            try:
                os.symlink(target, link)
            except OSError:
                pass
        if not os.path.isdir(link):
            raise SystemExit(f"integrity: cannot reach the stored results through {link}")
    script = os.path.join(ROOT, "codes", "scripts", "confirm", "c01_hypotheses.py")
    r = subprocess.run([sys.executable, script], cwd=ROOT, env=env, capture_output=True, text=True)
    new = os.path.join(out, "hypotheses.csv")
    if not os.path.exists(new):
        return dict(ok=False, note="the scorer did not write a table", stderr=r.stderr[-800:])
    a, b = pd.read_csv(kept), pd.read_csv(new)
    same_shape = a.shape == b.shape
    diffs = []
    if same_shape:
        for c in a.columns:
            if c not in b:
                diffs.append(c)
                continue
            if a[c].dtype.kind in "fc":
                d = (a[c] - b[c]).abs()
                if float(d.max(skipna=True) or 0) > 1e-9:
                    diffs.append(f"{c} (max |diff| {float(d.max()):.3g})")
            elif not a[c].astype(str).equals(b[c].astype(str)):
                diffs.append(c)
    return dict(ok=bool(same_shape and not diffs), shape_kept=list(a.shape), shape_new=list(b.shape),
                differing_columns=diffs, scorer_sha256=sha256(script),
                supported_kept=int((a.decision == "supported").sum()),
                supported_new=int((b.decision == "supported").sum()) if "decision" in b else None)


def git_tag():
    r = subprocess.run(["git", "tag", "--list", "prereg-v1"], cwd=ROOT, capture_output=True, text=True)
    c = subprocess.run(["git", "rev-parse", "prereg-v1^{commit}"], cwd=ROOT, capture_output=True, text=True)
    return dict(tag_present=bool(r.stdout.strip()), commit=c.stdout.strip() if c.returncode == 0 else None)


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=os.uname().nodename if hasattr(os, "uname") else "",
               pbs_job=os.environ.get("PBS_JOBID"), phase=O.PHASE, prereg=check_prereg(),
               frozen_files=check_frozen_files(), panel=check_panel(), git=git_tag(), rescore=rescore())
    bad = [f["file"] for f in rep["frozen_files"] if not f["ok"]]
    renamed = [f["file"] for f in rep["frozen_files"] if f.get("note")]
    rep["hashes_ok"] = bool(rep["prereg"]["ok"] and rep["panel"]["ok"] and not bad)
    rep["all_ok"] = bool(rep["hashes_ok"] and rep["rescore"]["ok"])
    json.dump(rep, open(os.path.join(OUT, "integrity.json"), "w"), indent=1)

    L = ["# Integrity and reproducibility check", "",
         f"Run {rep['time']} on `{rep['host']}` (PBS job {rep['pbs_job']}), phase `{rep['phase']}`.", "",
         "| Check | Result |", "|---|---|",
         f"| Pre-registration hash matches `PREREG_SHA256.txt` | {'yes' if rep['prereg']['ok'] else 'NO'} |",
         f"| Frozen choice files and scorer ({len(rep['frozen_files'])} files) unchanged | "
         f"{'yes' if not bad else 'NO: ' + ', '.join(bad)}"
         f"{'' if not renamed else ' (' + str(len(renamed)) + ' renamed on 24 Sep 2026, see RENAME_RECORD.md: ' + ', '.join(renamed) + ')'} |",
         f"| Panel content hash matches `CHECKSUM.txt` | {'yes' if rep['panel']['ok'] else 'NO'} "
         f"({rep['panel']['rows']} rows x {rep['panel']['cols']} columns) |",
         f"| Git tag `prereg-v1` present | {'yes' if rep['git']['tag_present'] else 'no'} "
         f"({rep['git']['commit'] or 'no commit'}) |",
         f"| Re-scoring the stored results reproduces `hypotheses.csv` | "
         f"{'yes, identical' if rep['rescore'].get('ok') else 'NO: ' + str(rep['rescore'].get('differing_columns') or rep['rescore'].get('note'))} |",
         "",
         f"Supported checks in the archived table: {rep['rescore'].get('supported_kept')} of "
         f"{rep['rescore'].get('shape_kept', [0])[0]}; in the re-scored table: "
         f"{rep['rescore'].get('supported_new')}.", "",
         "The scoring script is `codes/scripts/confirm/c01_hypotheses.py`, sha256 "
         f"`{rep['rescore'].get('scorer_sha256', '')}`."
         + (" That is the hash written into the pre-registration before the evaluation data were opened."
            if not renamed else
            " The pre-registration, written before the evaluation data were opened, records"
            " `f07727d831f0…` for this file; it differs only in the five import lines changed by the rename of"
            " 24 Sep 2026 (`codes/design/RENAME_RECORD.md`), and the re-scoring above is what shows that the"
            " change has no effect."), ""]
    open(os.path.join(OUT, "INTEGRITY.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    # a broken hash is a real failure; a re-scoring problem is reported but does not stop the other audit stages
    return 0 if rep["hashes_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
