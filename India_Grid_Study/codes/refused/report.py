"""Small reporting helpers (no optional dependencies)."""
import numpy as np
import pandas as pd


def md_table(df, digits=4):
    """GitHub-flavoured Markdown table from a DataFrame (numbers rounded to `digits`)."""
    df = df.copy()
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{digits}g}" if abs(v) >= 1e4 or (0 < abs(v) < 1e-3) else f"{round(v, digits)}"
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r.tolist()) + " |")
    return "\n".join(lines)
