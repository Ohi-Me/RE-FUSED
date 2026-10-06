"""Fill the frozen-choices table of the pre-registration draft from the development result files.

Reads 06_results/dev/o2/eval/selection.json, 06_results/dev/o4/weights.json and 06_results/dev/o5/frozen_params.json
(and records their SHA-256), replaces the "[from development]" rows of 02_design/02_preregistration_DRAFT.md and
writes 02_design/02_preregistration.md. It does NOT hash or tag: freezing is a separate, deliberate step
(`freeze_prereg.py`) after the authors have read the filled document.
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from refused8.paths import DESIGN, RES, ROOT  # noqa: E402


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    sel_f = os.path.join(RES, "dev", "o2", "eval", "selection.json")
    w_f = os.path.join(RES, "dev", "o4", "weights.json")
    p_f = os.path.join(RES, "dev", "o5", "frozen_params.json")
    missing = [f for f in (sel_f, w_f, p_f) if not os.path.exists(f)]
    if missing:
        sys.exit("development results missing: " + ", ".join(os.path.relpath(m, ROOT) for m in missing))
    sel = json.load(open(sel_f))
    w = json.load(open(w_f))
    p = json.load(open(p_f))
    o2_models = "; ".join(f"{s['tid']}: {s['selected']}" for s in sel)
    o2_levels = "; ".join(f"{s['tid']}: " + ", ".join(f"{m}={v['chosen']}" for m, v in s["levels"].items()
                                                     if m == s["selected"]) for s in sel)
    o4 = (f"R={w['weights']['R']:.1f}, S={w['weights']['S']:.1f}, C={w['weights']['C']:.1f}, U={w['weights']['U']:.1f}, "
          f"κ={w['kappa']}; without U: R={w['weights_noU']['R']:.1f}, S={w['weights_noU']['S']:.1f}, "
          f"C={w['weights_noU']['C']:.1f}, κ={w['kappa_noU']} (U source: {w['model_for_U']})")
    o5 = (f"A2 β={p['A2']['beta']}, ρ={p['A2']['rho']}; A3 ρ₀={p['A3']['rho0']}; A4 β₀={p['A4']['beta0']}, "
          f"β_U={p['A4']['beta_U']}, β_stress={p['A4']['beta_stress']}; A2r β=({p['A2r']['beta_low']}, "
          f"{p['A2r']['beta_high']}) (forecast source: {p['model']})")
    rows = {
        "O2 selected model per target (T1–T5)": (o2_models, sel_f),
        "O2 calibration level per target and model": (o2_levels, sel_f),
        "O4 weights (R, S, C, U) and κ; weights without U": (o4, w_f),
        "O5 A2 β, ρ; A3 ρ₀; A4 β₀, β_U, β_stress; A2r β": (o5, p_f),
    }
    text = open(os.path.join(DESIGN, "02_preregistration_DRAFT.md"), encoding="utf-8").read()
    for item, (value, src) in rows.items():
        pat = re.compile(r"^\| " + re.escape(item) + r" \| … \| (?:`[^`]+`|same) \|$", re.M)
        new = f"| {item} | {value} | `{os.path.relpath(src, ROOT)}` (sha256 {sha(src)[:16]}) |"
        text, n = pat.subn(lambda m: new, text)
        if n != 1:
            sys.exit(f"row not found exactly once: {item}")
    text = text.replace("# RE-FUSED-8 pre-registration (DRAFT — not frozen)", "# RE-FUSED-8 pre-registration")
    text = text.replace("Status: draft written 15 Sep 2026 while development training runs.",
                        "Status: filled from development results; frozen when PREREG_SHA256.txt and tag prereg-v1 exist.")
    open(os.path.join(DESIGN, "02_preregistration.md"), "w", encoding="utf-8").write(text)
    print("written 02_design/02_preregistration.md; review before freezing")


if __name__ == "__main__":
    main()
