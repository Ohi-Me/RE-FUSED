# Errata for the data documentation

The data values were correct in every version; only documentation text was affected.

| Found | Fixed in | File | Field | Text in version 1.0.0 | Correct text |
|---|---|---|---|---|---|
| 10 Oct 2026 | 1.1.0 | `FIELD_REGISTRY.csv` (data dictionary) | `dev_drawal_schedule_gwh` | note "OD(+)/UD(−) = actual − scheduled drawal, daily net" (copied from `dev_od_ud_gwh`) | "scheduled drawal, daily total" (Grid-India PSP section C, "Drawal Schedule"); actual drawal = this column + `dev_od_ud_gwh` |

The cause was one registry call in `codes/scripts/build/b07_panel.py` that gave both columns the same note; the code was
fixed on 11 Oct 2026.
