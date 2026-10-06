"""Blinding guard (ported from RE-FUSED): reserved-period rows can only be loaded after the pre-registration is frozen.

Conditions (all required):
  * environment variable REFUSED_CONFIRMATORY == "1"
  * codes/design/02_preregistration.md exists
  * codes/design/PREREG_SHA256.txt holds the SHA-256 of that file's current bytes
  * git tag "prereg-v1" exists in the repository
"""
import hashlib
import os
import subprocess

from .paths import DESIGN, ROOT

PREREG = os.path.join(DESIGN, "02_preregistration.md")
SHAFILE = os.path.join(DESIGN, "PREREG_SHA256.txt")


class BlindingError(RuntimeError):
    pass


def confirmatory_unlocked(verbose=False):
    reasons = []
    if os.environ.get("REFUSED_CONFIRMATORY") != "1":
        reasons.append("REFUSED_CONFIRMATORY is not 1")
    if not os.path.exists(PREREG):
        reasons.append("pre-registration file missing")
    elif not os.path.exists(SHAFILE):
        reasons.append("pre-registration hash missing")
    else:
        h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
        if open(SHAFILE).read().split()[0] != h:
            reasons.append("pre-registration changed after hashing")
    try:
        tags = subprocess.run(["git", "tag", "--list", "prereg-v1"], cwd=ROOT, capture_output=True, text=True).stdout
        if "prereg-v1" not in tags:
            reasons.append("git tag prereg-v1 missing")
    except Exception as e:  # pragma: no cover
        reasons.append("git unavailable: %s" % e)
    return (not reasons, reasons) if verbose else (not reasons)


def require_confirmatory():
    ok, reasons = confirmatory_unlocked(verbose=True)
    if not ok:
        raise BlindingError("confirmatory data are blinded: " + "; ".join(reasons))
