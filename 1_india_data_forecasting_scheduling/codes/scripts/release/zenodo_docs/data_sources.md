# Data sources

All data in this record come from reports published by agencies of the Government of India. We did not use any
commercial data, data from power exchanges' paid services, or data from aggregator websites.

## Sources

| Publisher | Report | What we take from it | Typical delay |
|---|---|---|---|
| Grid Controller of India (NLDC) | Daily Power Supply Position report | State energy met, demand, drawal schedule and actual drawal, deviations, regional generation, frequency statistics | about 1 day |
| Grid Controller of India (NLDC) | Deviation settlement rate files | normal deviation rate, day-ahead and real-time market prices by bid area | about 2 weeks |
| Central Electricity Authority, National Power Portal | Daily Generation Report | station-wise programme and actual generation, capacity, outages, coal stock days | about 2 days |
| Central Electricity Authority | Daily Renewable Generation Report | State wind, solar and other renewable generation | about 2 days |
| Central Electricity Authority | CO2 Baseline Database | emission factors by State and fuel, used for the carbon intensity columns | yearly |
| India Meteorological Department, Pune | Gridded daily rainfall (0.25 degree) and temperature (1 degree) | State-average rainfall and temperature | yearly files |

The delay column is the gap we measured between a data day and the day its value first became public. The exact
delay of every column is in `data/data_dictionary/FIELD_REGISTRY.csv`, and it is what the forecasting code uses to
decide which values a forecast is allowed to see.

## How the files were collected

The download scripts are in `code/preprocessing/acquire/`. For each source they fetch the public files and write a
manifest with the URL, the local file name, the HTTP status, the size, the SHA-256 hash and the retrieval time.
These manifests are in `data/source_manifests/`, one per source, including the requests that returned nothing.
We keep the manifests instead of the raw files because the raw reports belong to the publishing agencies and are
still available from them. Anyone can download them again and check each file against its hash.

## Things to know before using the data

* The renewable generation report stopped on {re_last}. After that date the renewable columns are empty.
* The gridded weather files end with the last published year, so weather columns end on 31 December 2025.
* Daily generation reports are missing for some days, most of them in the 2020 lockdown months.
* Scheduled and actual values are kept in separate columns and are never mixed.
* Energies are in GWh, power in MW, prices in rupees per MWh and emissions in tonnes of CO2.

The quality-control counts and the checks against independent official figures are in
`data/data_dictionary/QC_REPORT.md` and `data/data_dictionary/validation.json`.

## A note on older paths

The source registry and the datasheet were written when our project folders were numbered (`03_data`,
`04_code`). In this record those folders are `data/` and `code/`.
