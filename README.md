# README

This repository contains four primary modules used for managing signals, fetching signal data, running machine-learning pipelines, and refreshing daily calculations. The key scripts to run are **TablePipeline.py** (for end-to-end model training) and **DataRefresh.py** (for incremental metric calculations).

---

## Files

1. **Database.py**  
   - Manages connections to BigQuery and handles reading/writing data for:
     - A signal lookup table (`wmg.signal_lookup`)
     - A signal repository table (`wmg.signal_repository`)
     - A daily IC metrics table (`wmg.daily_ic2`)
     - Data in `core_raw` for baseline signals

2. **GetSignal.py**  
   - Fetches signal data from the repository, optionally checking baseline tables to ensure coverage is up-to-date.  
   - Can rebuild (refresh) a signal’s coverage if more recent data is found in the underlying baseline tables.

3. **TablePipeline.py**  
   - Provides an end-to-end machine-learning pipeline that:
     1. Fetches raw data from BigQuery (IC, F-values, MI data, etc.).
     2. Engineers features (rolling stats, transformations).
     3. Trains models (LightGBM or CatBoost) on a rolling window.
     4. Outputs predictions and optionally writes them back to BigQuery.

4. **DataRefresh.py**  
   - Contains classes to perform incremental calculations (IC, monotonicity checks, mutual information, F-values) for signals.
   - Each class:
     - Checks existing coverage in BigQuery.
     - Fetches any new signal data (via **GetSignal**).
     - Merges with returns data.
     - Computes daily or rolling metrics.
     - Appends new results to the appropriate BigQuery tables.


