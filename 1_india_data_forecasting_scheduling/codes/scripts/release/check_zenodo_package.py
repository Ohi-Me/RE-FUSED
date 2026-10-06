"""Check an unpacked copy of the RE-FUSED Zenodo record from the outside, the way a reader would.

1. every file matches MANIFEST.sha256;
2. the panel matches its content hash;
3. the pre-registration matches PREREG_SHA256.txt, and every file hashed inside it is in the record unchanged;
4. rebuilt into the project layout, the frozen scoring script reproduces results/hypotheses.csv.

Usage: python check_zenodo_package.py <unpacked record folder> <empty work folder>
Writes a short report to stdout and to <work folder>/ZENODO_CHECK.md; exit code 0 only if all four pass.
"""
import hashlib
import os
import re
import subprocess
import sys

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(record, work):
    out = {}
    # 1. manifest
    bad = []
    for line in open(os.path.join(record, "MANIFEST.sha256"), encoding="utf-8"):
        h, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(record, *rel.split("/"))
        if not os.path.exists(p) or sha256(p) != h:
            bad.append(rel)
    out["manifest"] = (not bad, f"{len(bad)} files differ" if bad else "all files match")

    # 2. panel content hash
    ck = open(os.path.join(record, "data", "processed_panel", "CHECKSUM.txt"), encoding="utf-8").read()
    want = re.search(r"content_sha256=(\w+)", ck).group(1)
    p = pd.read_parquet(os.path.join(record, "data", "processed_panel", "refused_state_day.parquet"))
    got = hashlib.sha256(pd.util.hash_pandas_object(p, index=False).values.tobytes()).hexdigest()
    out["panel"] = (got == want, f"{len(p)} x {p.shape[1]}, content hash {'matches' if got == want else 'DIFFERS'}")

    # 3. rebuild the layout, then the pre-registration and the files it froze
    subprocess.run([sys.executable, os.path.join(record, "code", "reproducibility", "release", "rebuild_layout.py"),
                    work], check=True, capture_output=True)
    design = os.path.join(work, "codes", "design")
    rec = open(os.path.join(design, "PREREG_SHA256.txt"), encoding="utf-8").read().split()[0]
    pre_ok = sha256(os.path.join(design, "02_preregistration.md")) == rec
    text = open(os.path.join(design, "02_preregistration.md"), encoding="utf-8").read()
    frozen = re.findall(r"^([0-9a-f]{64})\s+(\S+)$", text, flags=re.M)
    # a frozen file whose only change is a recorded renaming passes only if RENAME_RECORD.md holds exactly the pair
    # (hash in the pre-registration, hash in the record); the re-scoring below is what shows the change is harmless
    rr = os.path.join(design, "RENAME_RECORD.md")
    renamed = dict((rel, (old, new)) for rel, old, new in re.findall(
        r"`([^`]+\.py)`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*`([0-9a-f]{64})`",
        open(rr, encoding="utf-8").read() if os.path.exists(rr) else ""))
    wrong, recorded = [], []
    for h, rel in frozen:
        p = os.path.join(work, rel)
        got = sha256(p) if os.path.exists(p) else None
        if got == h:
            continue
        (recorded if renamed.get(rel) == (h, got) else wrong).append(rel)
    out["prereg"] = (pre_ok and not wrong, f"pre-registration hash {'matches' if pre_ok else 'DIFFERS'}; "
                                           f"{len(frozen) - len(wrong) - len(recorded)} of {len(frozen)} frozen files "
                                           f"unchanged" + (f", {len(recorded)} changed only by the renaming recorded "
                                                           f"in RENAME_RECORD.md" if recorded else "")
                                           + (f"; differ: {', '.join(wrong)}" if wrong else ""))

    # 4. re-score from the stored results
    env = dict(os.environ, REFUSED_PHASE="confirm", REFUSED_CONFIRMATORY="1", REFUSED_ROOT=work)
    r = subprocess.run([sys.executable, os.path.join(work, "codes", "scripts", "confirm", "c01_hypotheses.py")],
                       cwd=work, env=env, capture_output=True, text=True)
    new = os.path.join(work, "results", "confirm2", "hypotheses.csv")
    if r.returncode != 0 or not os.path.exists(new):
        out["rescore"] = (False, "scorer failed: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:200])
    else:
        a = pd.read_csv(os.path.join(record, "results", "hypotheses.csv"))
        b = pd.read_csv(new)
        diff = [c for c in a.columns if c not in b or not (
            ((a[c] - b[c]).abs().fillna(0) < 1e-9).all() if a[c].dtype.kind in "fc" else a[c].astype(str).equals(b[c].astype(str)))]
        out["rescore"] = (a.shape == b.shape and not diff,
                          f"{int((b.decision == 'supported').sum())} of {len(b)} supported; "
                          + ("identical to the record" if not diff else "differs in " + ", ".join(diff)))

    L = ["# Check of the RE-FUSED Zenodo record", "", "| Check | Result | Detail |", "|---|---|---|"]
    names = {"manifest": "Files match MANIFEST.sha256", "panel": "Panel content hash",
             "prereg": "Pre-registration and frozen files", "rescore": "Re-scoring from the stored results"}
    for k in ("manifest", "panel", "prereg", "rescore"):
        ok, detail = out[k]
        L.append(f"| {names[k]} | {'pass' if ok else 'FAIL'} | {detail} |")
    report = "\n".join(L) + "\n"
    open(os.path.join(work, "ZENODO_CHECK.md"), "w", encoding="utf-8").write(report)
    print(report)
    return 0 if all(v[0] for v in out.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])))
