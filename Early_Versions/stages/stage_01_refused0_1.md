# RE-FUSED-0.1 — the repackage (12 August 2026)

**Folder then:** `Refused0.1` (25 files, 141 MB)
**Kept now:** `Early_Versions/stage1_4_releases/refused0.1/`

## What it was

A portable bundle of RE-FUSED-Alpha, made for sharing. The notebook is **byte-identical** to RE-FUSED-0's
(`MD5 20fe64…`), and so is the data. Nothing in the code, the data or the model changed.

## What it added

Nothing scientific. It is the reference "before" state for the change that came next.

## What it removed

The audits, the baseline CSVs and the `results/` folder. After the repackage, the outputs survived only inside
the notebook. That is worth remembering: repackaging for portability quietly dropped the evidence that made the
results checkable, and it took the evolution study to notice.

## Status

Superseded. Kept for two reasons: it fixes the "before" point of the RE-FUSED-1 comparison, and it is the example
of a packaging step that lost evidence.
