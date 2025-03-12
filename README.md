# README

This repository contains four Jupyter notebooks that, used in tandem, fully encapsulate the pipeline for updating signals, computing metrics, and producing a final table for machine-learning workflows. The notebooks and their respective functions are:

1. **Update Signal Location Table**  
   - Reconciles each signal’s location via a lookup table in BigQuery.  
   - Distinguishes baseline signals from interaction signals and, for interaction signals, documents the source tables and associated calculations.  
   - Ensures dependencies are tracked so that, when source tables are refreshed (e.g., daily), signals can be rebuilt or updated accordingly.

2. **Update Signal Data**  
   - Uses the signal-location table from Step 1 to rebuild signals.  
   - Saves the results into a master table named `signal_repository`, which stores all historical signal data.  
   - Downstream metric/feature calculations and pipelines pull from this master table, rather than multiple disparate tables.

3. **Update Signal Metrics**  
   - Updates metrics (e.g., F-values, monotonicity) for each signal in the ML pipeline.  
   - Identifies any missing dates or “delta” periods and computes the relevant metrics.  
   - Appends these updates to the respective metric tables in BigQuery.

4. **Table Pipeline**  
   - Produces the final table used by Quint’s `ListFold` pipeline (utilizing LightGBM or CatBoost).  
   - Retrieves signal metrics, merges them with returns data or other features, and prepares training data.  
   - Trains models on a rolling basis and can write predictions or other outputs back to BigQuery.

---

## Underlying Python Modules

While the four notebooks above provide the main workflow, they rely on several Python modules for core functionality:

1. **Database.py**  
   - Manages connections to BigQuery.  
   - Handles reading/writing data for:  
     - The signal lookup table (`wmg.signal_lookup`)  
     - The signal repository table (`wmg.signal_repository`)  
     - The daily IC metrics table (`wmg.daily_ic2`)  
     - Data in `core_raw` (for baseline signals)

2. **GetSignal.py**  
   - Fetches signal data from the repository.  
   - Optionally checks baseline tables to ensure the latest coverage is included.  
   - Can rebuild (refresh) a signal’s coverage if more recent data is found in underlying tables.

3. **TablePipeline.py**  
   - Provides an end-to-end machine-learning pipeline that:  
     1. Fetches raw data from BigQuery (IC, F-values, MI data, etc.).  
     2. Engineers features (rolling stats, transformations).  
     3. Trains models (LightGBM or CatBoost) on a rolling window.  
     4. Outputs predictions and optionally writes them back to BigQuery.  
   - Underlies the **Table Pipeline** notebook.

4. **DataRefresh.py**  
   - Contains classes for incremental calculations (IC, monotonicity checks, mutual information, F-values) on signals.  
   - Checks existing coverage in BigQuery, fetches new signal data if needed, merges with returns data, and computes daily/rolling metrics.  
   - Appends new results to the appropriate BigQuery tables.  
   - Underlies the **Update Signal Metrics** notebook.

---

## Usage

1. **Update Signal Location Table**  
   - Run this notebook if new signals or interactions have been introduced, or if the underlying source tables have changed.

2. **Update Signal Data**  
   - Run after the location table is updated, to rebuild signals and populate the master `signal_repository`.

3. **Update Signal Metrics**  
   - Run to ensure all signal metrics (e.g., F-values, monotonicity) are current.  
   - This aligns with daily/weekly refresh cycles.

4. **Table Pipeline**  
   - Generates the final model input table and can train models using the latest metrics and signals.

For automated or headless workflows, you can also call the underlying modules (**TablePipeline.py**, **DataRefresh.py**, etc.) in your scheduling environment as needed.

---

## Additional Notes

- **Signal Lookup Table**: The `signal_lookup` table is populated with signals that are available as columns in any table of the `core_raw` feature set. For signals (e.g., original clusters) to be included in the `signal_lookup` table, they must first exist as column names within at least one table in the `core_raw` dataset. This ensures that any new signal being tracked can be verified against actual data availability in `core_raw` before being processed in the pipeline.

By following these four notebooks in sequence, you ensure that the signal definitions are always up-to-date, metrics are refreshed, and the final data table is ready for model training and inference.
