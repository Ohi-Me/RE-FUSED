# Deviations after the freeze (pre-registration v2, tag `prereg-v1`)

The pre-registration file is frozen by its hash, so deviations are listed here instead of in its section 6.

| Date | What | Why | Result read before the change? |
|---|---|---|---|
| 18 Sep 2026 10:10 | The first freeze attempt (job 31822) recorded the hash but made no git commit, because `git add` was given a file that was not on the cluster; the blinding guard then refused every stage, so no reserved data were read. The freeze step now checks every git call; job 31823 froze correctly (commit 4d3807e, tag `prereg-v1`). The pre-registration text and its hash did not change. | Bug in the freeze step. | No confirmatory result existed. |
| 18 Sep 2026 10:15 | Leakage test in the confirmatory phase: its issue days are moved forward by one year (the test now also asserts that every issue day is after the training period). | The development-phase dates (e.g. 15 Jul 2023) fall inside the confirmatory training period, where train-period scaling statistics legitimately include later days, so the test failed for a reason that is not leakage. The test logic is unchanged. | No. Only the failing test output was seen; no forecast or evaluation result. |
