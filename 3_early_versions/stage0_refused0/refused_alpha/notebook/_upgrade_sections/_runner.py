"""
_runner.py — development driver for the Part II section files.

Each `sNN_*.py` file is written as a sequence of notebook cells delimited by
`# %% [code]` / `# %% [markdown]` markers. This runner executes the code cells of
one or more section files IN ORDER inside a single shared namespace, exactly as
the notebook kernel will, so a section can be developed and verified before it is
transplanted into the notebook.

Usage
-----
    python _runner.py s01 s02          # run sections 1 and 2
    python _runner.py --all            # run every section in order
    python _runner.py s03 --resume     # reuse the cached namespace from a prior run
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                       # REFUSED_Ready
CACHE = HERE / "_ns_cache.pkl"


def parse_cells(path: Path) -> List[Tuple[str, str]]:
    """Split a section file into (kind, source) cells on `# %%` markers."""
    text = path.read_text(encoding="utf-8")
    cells: List[Tuple[str, str]] = []
    kind, buf = None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "markdown" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def strip_markdown(src: str) -> str:
    """Markdown cells are stored as `# ` comments in the .py files."""
    out = []
    for line in src.splitlines():
        out.append(line[2:] if line.startswith("# ") else
                   ("" if line.strip() == "#" else line))
    return "\n".join(out)


def main() -> int:
    # The section text uses typographic characters (arrows, box-drawing, Greek).
    # A Jupyter kernel sends stdout as UTF-8, but this dev runner inherits the
    # Windows console codepage (cp1252), which cannot encode them — so reconfigure
    # rather than let a print statement abort a multi-hour run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", help="section prefixes, e.g. s01 s02")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="restore the namespace cached by a previous run")
    ap.add_argument("--bootstrap", action="store_true",
                    help="run _bootstrap.py first (cheap shared state + checkpointed preds)")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale end-to-end run into a separate output dir")
    args = ap.parse_args()

    files = sorted(HERE.glob("s[0-9][0-9]_*.py"))
    if not args.all:
        want = tuple(args.sections)
        files = [f for f in files if f.name.startswith(want)] if want else files
    if not files:
        print("no section files matched")
        return 1

    ns: Dict[str, object] = {"__name__": "__main__"}
    if args.resume and CACHE.exists():
        with CACHE.open("rb") as fh:
            ns.update(pickle.load(fh))
        print(f"[runner] restored {len(ns)} names from cache")

    sys.path.insert(0, str(ROOT))
    import os
    os.chdir(ROOT)

    def _run_helper(path: Path, label: str) -> None:
        print(f"[runner] {label} via {path.name}")
        ns["__file__"] = str(path)          # helpers resolve siblings from this
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)

    if args.smoke:
        _run_helper(HERE / "_smoke.py", "SMOKE MODE (reduced scale, separate output dir)")
    elif args.bootstrap:
        _run_helper(HERE / "_bootstrap.py", "bootstrapping")

    for f in files:
        cells = parse_cells(f)
        code_cells = [(k, s) for k, s in cells if k == "code"]
        print(f"\n{'#'*78}\n[runner] {f.name} — {len(code_cells)} code cell(s)\n{'#'*78}")
        for i, (kind, src) in enumerate(cells):
            if kind != "code":
                continue
            t0 = time.time()
            try:
                exec(compile(src, f"{f.name}:cell{i}", "exec"), ns)
            except Exception:
                print(f"\n[runner] FAILED in {f.name} cell {i}:\n", flush=True)
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                return 2
            print(f"[runner]   cell {i} ok ({time.time()-t0:.1f}s)")

    picklable = {}
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        try:
            pickle.dumps(v)
            picklable[k] = v
        except Exception:
            pass
    with CACHE.open("wb") as fh:
        pickle.dump(picklable, fh)
    print(f"\n[runner] cached {len(picklable)} names -> {CACHE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
