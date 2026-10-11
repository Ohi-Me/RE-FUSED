# Errata for the published data documentation (Zenodo RE-FUSED 1.0.0, 10.5281/zenodo.23194781)

The data values are correct; only documentation text is affected. Corrections apply from the next Zenodo version.

| Date found | File | Field | Published text | Correct text |
|---|---|---|---|---|
| 10 Oct 2026 | `FIELD_REGISTRY.csv` (data dictionary) | `dev_drawal_schedule_gwh` | note "OD(+)/UD(−) = actual − scheduled drawal, daily net" (copied from `dev_od_ud_gwh`) | "scheduled drawal, daily total" (Grid-India PSP section C, "Drawal Schedule"); actual drawal = this column + `dev_od_ud_gwh` |

The cause was one registry call in `codes/scripts/build/b07_panel.py` that gave both columns the same note; it is fixed
in the code (11 Oct 2026). The published `FIELD_REGISTRY.csv` is left unchanged here so that the local archive keeps
matching the published record and its checksums.
