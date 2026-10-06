"""Export the frozen experiment configurations actually used into 05_configs/ (machine-readable, from run records).

Every ladder run stores its full LadderConfig in its JSON diagnostics; this script collects them per block so the
configuration of any reported number can be read without running code.
Outputs:
  05_configs/confirm_ladders.json      block -> list of {config, runs: [key_seed, ...]}   (C1..C7, pre-registered)
  05_configs/sensitivity_dc1.json      same for the DC1 re-runs (C3, C4, C5)
  05_configs/dev_ladders.json          same for development ladders (every folder under 06_results/dev with run JSONs)
  05_configs/ladder_defaults.json      LadderConfig defaults of the frozen code
"""
import dataclasses
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refused_gate.ladder import LadderConfig  # noqa: E402
from refused_gate.paths import RES, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "comparison_other_countries", "05_configs")


def collect(folders):
    blocks = {}
    for folder in folders:
        name = os.path.relpath(folder, RES).replace("\\", "/")
        groups = []
        for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
            try:
                rec = json.load(open(f))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            cfg = rec.get("config") if isinstance(rec, dict) else None
            if cfg is None:
                continue
            key = os.path.basename(f)[:-5]
            for g in groups:
                if g["config"] == cfg:
                    g["runs"].append(key)
                    break
            else:
                groups.append(dict(config=cfg, runs=[key]))
        if groups:
            blocks[name] = groups
    return blocks


def dump(obj, name):
    path = os.path.join(OUT, name)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1)
    n = sum(len(g["runs"]) for gs in obj.values() for g in gs) if isinstance(obj, dict) and obj and isinstance(next(iter(obj.values())), list) else ""
    print("wrote", path, n)


def main():
    os.makedirs(OUT, exist_ok=True)
    dump(collect(sorted(glob.glob(os.path.join(RES, "confirm", "C*")))), "confirm_ladders.json")
    dump(collect(sorted(glob.glob(os.path.join(RES, "confirm_sensitivity", "C*")))), "sensitivity_dc1.json")
    dev = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(RES, "dev", "**", "*.json"), recursive=True)})
    dump(collect(dev), "dev_ladders.json")
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(LadderConfig()).items()}
    json.dump(d, open(os.path.join(OUT, "ladder_defaults.json"), "w"), indent=1)
    print("wrote ladder_defaults.json")


if __name__ == "__main__":
    main()
