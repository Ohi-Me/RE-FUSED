"""Blinding guard: confirmatory rows can only be loaded after the pre-registration is frozen.

Conditions (all required):
  * environment variable REFUSED_CONFIRMATORY == "1"
  * comparison_other_countries/02_design/02_preregistration.md exists
  * comparison_other_countries/02_design/PREREG_SHA256.txt holds the SHA-256 of that file's current bytes
  * git tag "prereg-v1" exists in the repository
"""
import hashlib
import os
import subprocess

from .paths import FOREIGN, ROOT

PREREG = os.path.join(FOREIGN, "02_design", "02_preregistration.md")      # the earlier study on non-Indian data
SHAFILE = os.path.join(FOREIGN, "02_design", "PREREG_SHA256.txt")


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


# ---- the Indian hourly study has its own pre-registration and its own lock ----
PREREG_INDIA = os.path.join(ROOT, "02_design", "05_preregistration_india.md")
SHAFILE_INDIA = os.path.join(ROOT, "02_design", "PREREG_INDIA_SHA256.txt")
TAG_INDIA = "prereg-india-v1"


def india_unlocked(verbose=False):
    """The test months of the all-India hourly study (from 1 March 2026) can be read only when
    REFUSED_INDIA_CONFIRMATORY == "1", 02_design/05_preregistration_india.md exists, PREREG_INDIA_SHA256.txt holds
    its SHA-256, and the git tag prereg-india-v1 exists."""
    reasons = []
    if os.environ.get("REFUSED_INDIA_CONFIRMATORY") != "1":
        reasons.append("REFUSED_INDIA_CONFIRMATORY is not 1")
    if not os.path.exists(PREREG_INDIA):
        reasons.append("India pre-registration file missing")
    elif not os.path.exists(SHAFILE_INDIA):
        reasons.append("India pre-registration hash missing")
    elif open(SHAFILE_INDIA).read().split()[0] != hashlib.sha256(open(PREREG_INDIA, "rb").read()).hexdigest():
        reasons.append("India pre-registration changed after hashing")
    try:
        tags = subprocess.run(["git", "tag", "--list", TAG_INDIA], cwd=ROOT, capture_output=True, text=True).stdout
        if TAG_INDIA not in tags:
            reasons.append(f"git tag {TAG_INDIA} missing")
    except Exception as e:  # pragma: no cover
        reasons.append("git unavailable: %s" % e)
    return (not reasons, reasons) if verbose else (not reasons)


def require_india_confirmatory():
    ok, reasons = india_unlocked(verbose=True)
    if not ok:
        raise BlindingError("Indian hourly test months are blinded: " + "; ".join(reasons))
