"""
_build_notebook.py — assemble Part II and append it to REFUSED_Alpha_Final.ipynb.

Safety contract
---------------
The existing notebook is the reproducibility record of the published results, so
this script is built around one rule: **the first N cells must come out
byte-identical to the way they went in.**

  1. a timestamped backup is written before anything is touched;
  2. every existing cell is hashed (source + outputs + metadata) BEFORE the edit;
  3. new cells are only ever APPENDED — never inserted, reordered or replaced;
  4. the hashes are recomputed AFTER writing and compared one by one;
  5. any mismatch, or any shrink in cell count, restores the backup and aborts.

Two stages, so a failed execution can never corrupt the target notebook:

    python _build_notebook.py stage      # sections -> _part2_staging.ipynb
    python _build_notebook.py execute    # run the staging notebook (captures outputs)
    python _build_notebook.py append     # verified append into the real notebook
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
NB_DIR = HERE.parent
ROOT = NB_DIR.parent
TARGET = NB_DIR / "REFUSED_Alpha_Final.ipynb"
STAGING = HERE / "_part2_staging.ipynb"
EXECUTED = HERE / "_part2_executed.ipynb"

sys.path.insert(0, str(HERE))
from _runner import parse_cells, strip_markdown  # noqa: E402


# ------------------------------------------------------------------ hashing
def cell_fingerprint(cell: dict) -> str:
    """Hash everything that makes a cell what it is."""
    blob = json.dumps({
        "cell_type": cell.get("cell_type"),
        "source": cell.get("source"),
        "outputs": cell.get("outputs", []),
        "execution_count": cell.get("execution_count"),
        "metadata": cell.get("metadata", {}),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_nb(nb: dict, path: Path) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------- stage
def build_cells() -> List[dict]:
    """Turn the section .py files into notebook cells, in order."""
    cells: List[dict] = []
    for f in sorted(HERE.glob("s[0-9][0-9]_*.py")):
        for kind, src in parse_cells(f):
            if kind == "markdown":
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": strip_markdown(src).splitlines(keepends=True)})
            else:
                cells.append({"cell_type": "code", "metadata": {},
                              "execution_count": None, "outputs": [],
                              "source": src.splitlines(keepends=True)})
    return cells


def stage() -> int:
    base = load_nb(TARGET)
    cells = build_cells()
    nb = {"cells": cells,
          "metadata": base.get("metadata", {}),
          "nbformat": base.get("nbformat", 4),
          "nbformat_minor": base.get("nbformat_minor", 5)}
    save_nb(nb, STAGING)
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"[stage] {len(cells)} cells ({n_code} code) -> {STAGING.name}")
    return 0


# ----------------------------------------------------------------- execute
def resolve_kernel(preferred: str = "") -> str:
    """Pick a kernelspec backed by the interpreter that is running this script.

    The default `python3` spec points at whatever `python` resolves to on PATH,
    which on this machine is the Windows Store stub — a kernel that cannot even
    start, let alone import torch. Matching on `sys.executable` guarantees the
    notebook runs in the same environment the sections were developed against.
    """
    from jupyter_client.kernelspec import KernelSpecManager

    if preferred:
        return preferred
    mgr = KernelSpecManager()
    me = Path(sys.executable).resolve()
    for name in mgr.find_kernel_specs():
        try:
            argv0 = mgr.get_kernel_spec(name).argv[0]
        except Exception:
            continue
        try:
            if Path(argv0).resolve() == me:
                return name
        except OSError:
            continue
    print("[execute] WARNING: no kernelspec matches this interpreter; "
          "falling back to 'python3'")
    return "python3"


def execute(timeout: int = 36000, kernel: str = "") -> int:
    import nbformat
    from nbclient import NotebookClient

    kernel_name = resolve_kernel(kernel)
    nb = nbformat.read(STAGING, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel_name,
                            resources={"metadata": {"path": str(ROOT)}},
                            allow_errors=False)
    print(f"[execute] kernel={kernel_name}")
    print(f"[execute] running {len(nb.cells)} cells with cwd={ROOT} ...")
    started = datetime.now()
    client.execute()
    nbformat.write(nb, EXECUTED)
    print(f"[execute] finished in {datetime.now()-started} -> {EXECUTED.name}")
    return 0


# ------------------------------------------------------------------ append
def append() -> int:
    src = EXECUTED if EXECUTED.exists() else STAGING
    if src is STAGING:
        print("[append] WARNING: appending UNEXECUTED cells (no _part2_executed.ipynb)")

    target = load_nb(TARGET)
    before = [cell_fingerprint(c) for c in target["cells"]]
    n_before = len(before)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = NB_DIR / f"REFUSED_Alpha_Final.BACKUP_{stamp}_pre_part2.ipynb"
    shutil.copy2(TARGET, backup)
    print(f"[append] backup -> {backup.name}")
    print(f"[append] existing cells: {n_before}")

    new_cells = load_nb(src)["cells"]
    target["cells"] = target["cells"] + new_cells
    save_nb(target, TARGET)

    # ---- verify -----------------------------------------------------------
    check = load_nb(TARGET)
    after = [cell_fingerprint(c) for c in check["cells"]]
    problems = []
    if len(after) < n_before:
        problems.append(f"cell count shrank: {n_before} -> {len(after)}")
    for i, (a, b) in enumerate(zip(before, after[:n_before])):
        if a != b:
            problems.append(f"existing cell {i} was modified")

    if problems:
        shutil.copy2(backup, TARGET)
        print("[append] VERIFICATION FAILED — original restored:")
        for p in problems:
            print("   ", p)
        return 2

    print(f"[append] VERIFIED: all {n_before} existing cells byte-identical")
    print(f"[append] appended {len(new_cells)} new cells "
          f"-> {len(after)} total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_name", choices=["stage", "execute", "append", "verify"])
    ap.add_argument("--timeout", type=int, default=36000)
    ap.add_argument("--kernel", default="",
                    help="kernelspec name; default is the one matching this interpreter")
    a = ap.parse_args()
    if a.stage_name == "stage":
        return stage()
    if a.stage_name == "execute":
        return execute(a.timeout, a.kernel)
    if a.stage_name == "append":
        return append()
    nb = load_nb(TARGET)
    print(f"{TARGET.name}: {len(nb['cells'])} cells")
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"]).split("\n")[0][:80]
        print(f"  {i:>3} {c['cell_type']:<9} {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
