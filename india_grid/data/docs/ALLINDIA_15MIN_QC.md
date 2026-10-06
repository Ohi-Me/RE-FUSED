# All-India 15-minute series: coverage and quality control

Built by `codes/scripts/build/b09_psp_timeseries.py` from 674 Daily PSP reports (1348 XLS files read). Data dates 2024-11-04 to 2026-09-13: 679 calendar days, 674 with data.

| Column | Missing blocks | Out of range | Frozen | Mean | Min | Max |
|---|---|---|---|---|---|---|
| freq_hz | 613 | 12 | 1394 | 50 | 0 | 50 |
| demand_met_mw | 613 | 0 | 0 | 201,027 | 126,804 | 270,820 |
| nuclear_mw | 613 | 0 | 0 | 6,014 | 0 | 7,356 |
| wind_mw | 613 | 0 | 0 | 11,215 | 0 | 36,584 |
| solar_mw | 613 | 514 | 166 | 19,863 | -1,066 | 85,444 |
| hydro_mw | 613 | 1 | 0 | 17,826 | -4,141 | 40,046 |
| gas_mw | 613 | 0 | 0 | 2,679 | 0 | 10,947 |
| thermal_mw | 613 | 6 | 0 | 143,482 | 0 | 189,002 |
| others_mw | 613 | 1 | 95 | 2,375 | -933 | 11,055 |
| net_demand_met_mw | 613 | 0 | 0 | 169,949 | 107,555 | 234,041 |
| total_generation_mw | 613 | 6 | 0 | 203,668 | 0 | 273,654 |
| net_exchange_mw | 613 | 0 | 0 | -187 | -2,800 | 2,703 |

Sheet layouts (days): original 569, with_storage 105. From 1 June 2026 the sheet adds storage demand (pumped storage and battery charging, included in demand met) and storage generation (pumped storage and batteries, no longer counted in hydro). Net demand met is defined the same way throughout (demand met minus wind and solar).

Parse outcomes: ok 675, no_timeseries_sheet 671, no_rows 2.
Choices for dates with more than one report: dropped_duplicate_content 1.

Source: Grid Controller of India, Daily Power Supply Position report, sheet TimeSeries (SCADA, instantaneous values at the start of each block). Grid-India notes that the data are operational SCADA values that may contain telemetry errors, and that demand met and RE generation are those incident on the transmission system (distribution-connected and behind-the-meter generation excluded).
