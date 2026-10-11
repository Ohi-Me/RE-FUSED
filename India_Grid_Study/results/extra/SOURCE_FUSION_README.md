# Exploratory: source fusion for scheduling (protocol codes/design/10_source_fusion_protocol.md)

| Step | Job | Output |
|---|---|---|
| Development run (fit FY2023-24, score FY2024-25), CPU queue | 35030 (11 Oct 2026) | `source_fusion_dev.json`, `logs/pbs/refused_source_fusion_dev.out` |
| Protocol frozen and run on the evaluation period | not done | stopped after development, as section 4 of the protocol says |

Blends S = w A2 + (1 - w) H_pub of the paper's frozen rule A2 and the State's recorded schedule of day t-2.

## Development result (mean system regret, Rs crore per day, FY2024-25)

| | stylised, projected | stylised, not projected | DSM-closer, projected | DSM-closer, not projected |
|---|---|---|---|---|
| A2 | 15.37 | 11.58 | 68.23 | 49.46 |
| H_pub (recorded schedule, t-2) | 17.98 | 14.55 | 78.91 | 62.22 |
| G0 (better single source) | A2 | A2 | A2 | A2 |
| G1 (one weight) | 15.36 (w = 0.9) | 11.58 (w = 0.9) | 68.23 (w = 1) | 49.46 (w = 1) |
| G2 (weight per State) | 15.42 | 11.66 | 68.62 | 49.83 |
| G3 (online Fixed Share) | 15.39 | 11.62 | 68.43 | 49.67 |
| G4 (learned weight) | 15.56 | 11.82 | 69.39 | 50.94 |

No variant beats G0 (Holm p = 1 throughout); G2, G3 and G4 are worse than A2 alone, G4 significantly.

## Reading

* The premise did not hold. Compared like for like (both projected, or both not projected), the forecast rule A2 has
  lower regret than the recorded schedule in the development year too, by 2.6 to 12.8 Rs crore per day. The earlier
  note that "unprojected persistence was cheaper than every rule" compared the unprojected schedule with projected
  rules; it reflects the cost of the fixed schedule limits, not information in the recorded schedule.
* With one source dominating, the best blend is the dominant source itself (G1 chooses w = 0.9 to 1; at 0.9 the gain is 0.002 to 0.006 Rs crore per day, interval covering zero), and every adaptive
* By-product: projection onto the fixed limits costs A2 about 3.8 Rs crore per day (stylised) and 18.8 (DSM-closer)
  in FY2024-25. The paper already reports that limits following demand lower A2's regret (15.80 to 13.49 in the
  evaluation period); this development figure is consistent with it.
