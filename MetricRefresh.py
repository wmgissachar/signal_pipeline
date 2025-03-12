import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from pandas_gbq import to_gbq
from datetime import timedelta

from scipy.optimize import curve_fit
import warnings

# Need imports
from Database import Database
from GetSignal import GetSignal
from sklearn.feature_selection import mutual_info_regression




class IncrementalICCalculator:
    """
    A class that checks how much data for a given signal is already computed
    in daily_ic2. If there's new data in the baseline/feature_set coverage, 
    it pulls that (plus one year prior for context), computes daily IC metrics,
    and appends them to wmg.daily_ic2.
    """

    def __init__(
        self,
        db,               # your Database instance
        get_signal,       # an instance of your updated GetSignal class
        project_id: str = "issachar-feature-library",
        wmg_dataset: str = "wmg",
        returns_table: str = "t1_returns",
        daily_ic2_table: str = "daily_ic2"
    ):
        """
        Args:
            db: An instance of your Database class
            get_signal: An instance of your updated GetSignal class
            project_id (str): GCP project ID
            wmg_dataset (str): The BQ dataset name containing daily_ic2
            returns_table (str): The table name containing t1_returns
            daily_ic2_table (str): The table name where we store IC metrics
        """
        self.db = db
        self.get_signal = get_signal
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.returns_table = returns_table
        self.daily_ic2_table = daily_ic2_table

        self.bq_client = bigquery.Client(project=self.project_id)

    def run(self, signal_name: str, rank_signal: bool = True, refresh_data: bool = False):
        """
        1) Check coverage in daily_ic2 for this signal.
        2) If there's new coverage, we want to compute daily metrics from
           [new_start_date - 1 year .. newest coverage].
        3) We use get_signal.run(...) with refresh_data to ensure we have up-to-date coverage.
        4) Then we pull returns & do IC calculations, finally appending to daily_ic2.
        """
        # ---------------------------------------------------------------------
        # 1) Check if the signal exists in your 'signal_lookup' table
        # ---------------------------------------------------------------------
        sig_info = self.db.get_signal(signal_name)
        if not sig_info:
            raise ValueError(f"Signal '{signal_name}' not found in lookup table or has no metadata.")

        # ---------------------------------------------------------------------
        # 2) Find existing coverage in daily_ic2 for this signal => (min_date, max_date)
        # ---------------------------------------------------------------------
        min_date_ic2, max_date_ic2 = self._get_daily_ic2_minmax_date(signal_name)

        # If there's no coverage at all, we'll process from earliest coverage
        if max_date_ic2 is None:
            new_start_date = None
        else:
            new_start_date = (max_date_ic2 + timedelta(days=1)).strftime("%Y-%m-%d")

        # ---------------------------------------------------------------------
        # 3) Retrieve coverage from get_signal (which can rebuild if refresh_data=True)
        # ---------------------------------------------------------------------
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,
            end_date=None,
            refresh_data=refresh_data
        )
        if df_signal.empty:
            print(f"No signal data retrieved for '{signal_name}'. Nothing to compute.")
            return

        # ---------------------------------------------------------------------
        # 4) Fetch returns from (new_start_date - 1 year) if partial coverage
        # ---------------------------------------------------------------------
        if new_start_date is not None:
            dt_for_returns = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns_str = dt_for_returns.strftime("%Y-%m-%d")
        else:
            start_for_returns_str = None

        df_returns = self._fetch_returns_in_date_range(start_for_returns_str, None)
        if df_returns.empty:
            print("No returns data found for the requested period. Stopping.")
            return

        # ---------------------------------------------------------------------
        # 5) Merge signal + returns on [date, requestId]
        # ---------------------------------------------------------------------
        df_signal.set_index(["date","requestId"], inplace=True)
        df_returns.set_index(["date","requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner")

        if df_merged.empty:
            print("No overlapping signal & returns data. Nothing to compute.")
            return

        # Rank the signal if requested
        if rank_signal:
            df_merged[signal_name] = df_merged.groupby("date")[signal_name].rank(pct=True)

        # ---------------------------------------------------------------------
        # 6) Compute daily metrics (all_IC, topq_IC, etc.)
        # ---------------------------------------------------------------------
        daily_metrics_df = self._compute_daily_metrics(
            df_merged, signal_col=signal_name, returns_col="t1_returns"
        )

        # If it's empty, no rows => no uploading
        if daily_metrics_df.empty:
            print(f"No daily metrics computed for '{signal_name}'. Nothing to upload.")
            return

        # Force the index to be a DatetimeIndex in case it's not
        try:
            daily_metrics_df.index = pd.to_datetime(daily_metrics_df.index, errors="coerce")
        except Exception as e:
            print(f"Could not convert daily_metrics_df.index to datetime: {e}")
            print("Skipping date filtering.")
        else:
            # ---------------------------------------------------------------------
            # 7) Filter out rows prior to new_start_date, if we have that
            # ---------------------------------------------------------------------
            if new_start_date is not None:
                dt_new_start = pd.to_datetime(new_start_date)
                daily_metrics_df = daily_metrics_df.loc[daily_metrics_df.index >= dt_new_start]

        if daily_metrics_df.empty:
            print(f"Daily metrics DataFrame is empty after date filtering. Nothing to upload.")
            return

        # ---------------------------------------------------------------------
        # 8) Append new rows to daily_ic2
        # ---------------------------------------------------------------------
        self._append_to_daily_ic2(daily_metrics_df, signal_name)
        print(f"Appended {len(daily_metrics_df)} new daily IC rows for '{signal_name}' into daily_ic2.")


    ###########################################################################
    # Helpers: _get_daily_ic2_minmax_date, _fetch_returns_in_date_range, 
    #          _compute_daily_metrics, _append_to_daily_ic2
    ###########################################################################
    def _get_daily_ic2_minmax_date(self, signal_name: str):
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.daily_ic2_table}"
        query = f"""
        SELECT 
          CAST(MIN(date) AS DATETIME) as min_date,
          CAST(MAX(date) AS DATETIME) as max_date
        FROM `{full_table}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)

        row = df.iloc[0]
        return (row["min_date"], row["max_date"])


    def _fetch_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        query_params = []

        # We store date column as TIMESTAMP => pass TIMESTAMP parameters
        if start_date:
            start_dt = pd.to_datetime(start_date)
            where_clauses.append("date >= @start_ts")
            query_params.append(bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start_dt))

        if end_date:
            end_dt = pd.to_datetime(end_date)
            where_clauses.append("date <= @end_ts")
            query_params.append(bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end_dt))

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT date, requestId, t1_returns
        FROM `{full_table}`
        {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return pd.DataFrame(columns=["date", "requestId", "t1_returns"])

        # Convert the date column to naive timestamps
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df


    def _compute_daily_metrics(self, df_merged: pd.DataFrame, signal_col: str, returns_col: str) -> pd.DataFrame:
        """
        For each date, compute:
         - all_IC (OLS slope), all_tstat, all_pvalue, all_fstat, all_IC_spearmanr, all_pvalue_spearmanr
         - topq_IC, topq_tstat, topq_pvalue, topq_fstat, topq_IC_spearmanr, topq_pvalue_spearmanr
        Return a DataFrame with index=date, plus those columns.
        """
        # Grab unique dates from the multi-index
        dates = df_merged.index.get_level_values("date").unique()
        if len(dates) == 0:
            return pd.DataFrame()

        results = []
        for dt in dates:
            day_data = df_merged.xs(dt, level="date")  # sub-DataFrame for a single date
            all_stats = self._compute_all_ic(day_data, signal_col, returns_col)
            topq_stats = self._compute_topq_ic(day_data, signal_col, returns_col)

            row = {
                "date": dt,
                **all_stats,
                **topq_stats
            }
            results.append(row)

        if not results:
            # No data to form a DataFrame
            return pd.DataFrame()

        # Construct final DataFrame => index=date
        out_df = pd.DataFrame(results)
        # Convert "date" column to datetime, then set as index
        out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce")
        out_df.set_index("date", inplace=True)

        return out_df


    def _compute_all_ic(self, df_day: pd.DataFrame, sc: str, rc: str) -> dict:
        X = df_day[sc]
        Y = df_day[rc]
        valid = X.notna() & Y.notna()
        X = X[valid]
        Y = Y[valid]

        if len(X) < 5 or X.nunique() < 2:
            return {
                "all_IC": np.nan,
                "all_tstat": np.nan,
                "all_pvalue": np.nan,
                "all_fstat": np.nan,
                "all_IC_spearmanr": np.nan,
                "all_pvalue_spearmanr": np.nan
            }

        Xc = sm.add_constant(X)
        model = sm.OLS(Y, Xc).fit()
        slope = model.params.iloc[1]
        tstat = model.tvalues.iloc[1]
        pval = model.pvalues.iloc[1]
        fstat = model.fvalue

        corr, p_spear = spearmanr(X, Y)
        return {
            "all_IC": slope,
            "all_tstat": tstat,
            "all_pvalue": pval,
            "all_fstat": fstat,
            "all_IC_spearmanr": corr,
            "all_pvalue_spearmanr": p_spear
        }

    def _compute_topq_ic(self, df_day: pd.DataFrame, sc: str, rc: str) -> dict:
        threshold = df_day[sc].quantile(0.75)
        df_top = df_day[df_day[sc] >= threshold]
        X = df_top[sc]
        Y = df_top[rc]
        valid = X.notna() & Y.notna()
        X = X[valid]
        Y = Y[valid]

        if len(X) < 5 or X.nunique() < 2:
            return {
                "topq_IC": np.nan,
                "topq_tstat": np.nan,
                "topq_pvalue": np.nan,
                "topq_fstat": np.nan,
                "topq_IC_spearmanr": np.nan,
                "topq_pvalue_spearmanr": np.nan
            }

        Xc = sm.add_constant(X)
        model = sm.OLS(Y, Xc).fit()
        slope = model.params.iloc[1]
        tstat = model.tvalues.iloc[1]
        pval = model.pvalues.iloc[1]
        fstat = model.fvalue

        corr, p_spear = spearmanr(X, Y)
        return {
            "topq_IC": slope,
            "topq_tstat": tstat,
            "topq_pvalue": pval,
            "topq_fstat": fstat,
            "topq_IC_spearmanr": corr,
            "topq_pvalue_spearmanr": p_spear
        }


    def _append_to_daily_ic2(self, df_metrics: pd.DataFrame, signal_name: str):
        """
        Prepares the final columns and appends rows to daily_ic2.
        """
        df_metrics = df_metrics.copy()
        df_metrics.reset_index(inplace=True)  # make 'date' a column
        df_metrics["signal"] = signal_name

        final_cols = [
            "date",
            "all_IC", "all_tstat", "all_pvalue", "all_fstat",
            "all_IC_spearmanr", "all_pvalue_spearmanr",
            "topq_IC", "topq_tstat", "topq_pvalue", "topq_fstat",
            "topq_IC_spearmanr", "topq_pvalue_spearmanr",
            "signal"
        ]
        for c in final_cols:
            if c not in df_metrics.columns:
                df_metrics[c] = np.nan
        df_metrics = df_metrics[final_cols]

        table_id = f"{self.project_id}.{self.wmg_dataset}.{self.daily_ic2_table}"
        to_gbq(df_metrics, table_id, project_id=self.project_id, if_exists="append")
        print(f"Appended {len(df_metrics)} rows into {table_id}.")

        


class IncrementalMonotonic:
    """
    Similar to IncrementalICCalculator, but computes quartile binning,
    daily monotonic checks, rolling window proportions, etc.
    Then appends results to 'rolling_monotonic_tables_pct_st'.
    """

    def __init__(
        self,
        db,
        get_signal,
        project_id: str = "issachar-feature-library",
        wmg_dataset: str = "wmg",
        returns_table: str = "t1_returns",
        monotonic_table: str = "rolling_monotonic_tables_pct_st"
    ):
        """
        Args:
            db: An instance of your Database class.
            get_signal: An instance of your GetSignal class (to fetch signal coverage).
            project_id, wmg_dataset, returns_table: Where returns & your table live.
            monotonic_table: The table to store final results, e.g. 'rolling_monotonic_tables_pct_st'.
        """
        self.db = db
        self.get_signal = get_signal  # We'll call get_signal.run(...) to fetch coverage
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.returns_table = returns_table
        self.monotonic_table = monotonic_table

        self.bq_client = bigquery.Client(project=self.project_id)
        # 1) Create the table if it doesn't exist
        self._create_monotonic_table_if_not_exists()

    def _create_monotonic_table_if_not_exists(self):
        """
        Ensure that `rolling_monotonic_tables_pct_st` exists in BQ.
        We define a minimal schema: [date TIMESTAMP, signal STRING, ...].
        You can adjust the columns as you see fit.
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.monotonic_table}"
        table_ref = bigquery.TableReference.from_string(full_table)

        try:
            self.bq_client.get_table(table_ref)
            # Table exists
        except NotFound:
            # Need to create it with an approximate schema
            schema = [
                bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("signal", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("monotonic_increasing_pct_21", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_decreasing_pct_21", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_neither_pct_21", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_increasing_pct_42", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_decreasing_pct_42", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_neither_pct_42", "FLOAT", mode="NULLABLE"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            self.bq_client.create_table(table)
            print(f"Created table {full_table} with columns: {','.join(f.name for f in schema)}")

    def safe_qcut(self, group):
        """
        Attempt to qcut a group into 4 bins (quartiles).
        If we can't (e.g. not enough unique values), return NaN.
        """
        if len(group) >= 4:
            try:
                return pd.qcut(group, 4, labels=False, duplicates='drop')
            except ValueError:
                return pd.Series(np.nan, index=group.index)
        else:
            return pd.Series(np.nan, index=group.index)

    def run(self, signal_name: str, refresh_data: bool = False):
        """
        Main flow:
          1) Check coverage in rolling_monotonic_tables_pct_st (min/max date).
          2) Pull new coverage from get_signal (plus returns).
          3) For new coverage, compute quartile bins, daily monotonic check,
             rolling window proportions, store results.
          4) Append new rows to monotonic_table.
        """
        # 1) Check coverage in monotonic table
        min_mono, max_mono = self._get_minmax_date_in_monotonic_table(signal_name)

        # We'll do from day after max_mono if coverage exists
        if max_mono is None:
            new_start_date = None
        else:
            new_start_date = (max_mono + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Use get_signal to fetch coverage
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,
            end_date=None,
            refresh_data=refresh_data
        )
        if df_signal.empty:
            print(f"No data retrieved for '{signal_name}'. Stopping.")
            return

        # We'll pull returns from new_start_date - 1 year if partial coverage,
        # or from earliest coverage if new_start_date is None
        if new_start_date is not None:
            dt_for_returns = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns_str = dt_for_returns.strftime("%Y-%m-%d")
        else:
            start_for_returns_str = None

        df_returns = self._fetch_returns_in_date_range(start_for_returns_str, None)
        if df_returns.empty:
            print("No returns data found for the requested period. Stopping.")
            return

        # 3) Merge on [date, requestId]
        df_signal.set_index(["date", "requestId"], inplace=True)
        df_returns.set_index(["date", "requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner")

        if df_merged.empty:
            print("No overlapping signal & returns data. Stopping.")
            return

        df_merged.reset_index(inplace=True)
        # Next steps: quartile bins in descending order
        df_merged["rank"] = df_merged.groupby("date")[signal_name].rank(pct=True, ascending=False)
        df_merged["eng_bin"] = df_merged.groupby("date")["rank"].transform(self.safe_qcut)

        # group by [date, eng_bin] => compute mean t1_returns
        df_quartile = df_merged.groupby(["date","eng_bin"])["t1_returns"].mean().unstack()

        # if df_quartile is empty or missing columns [0,1,2,3], short-circuit
        if df_quartile.empty:
            print("df_quartile is empty, no monotonic checks performed.")
            return

        # daily monotonic check
        for col_needed in [0,1,2,3]:
            if col_needed not in df_quartile.columns:
                # if any bin is missing, we can't do a full chain check
                # we can fill with np.nan or skip monotonic check
                df_quartile[col_needed] = np.nan

        df_quartile["monotonic_decreasing"] = df_quartile.apply(
            lambda x: x[0] > x[1] > x[2] > x[3] if x[[0,1,2,3]].notna().all() else False,
            axis=1
        )
        df_quartile["monotonic_increasing"] = df_quartile.apply(
            lambda x: x[0] < x[1] < x[2] < x[3] if x[[0,1,2,3]].notna().all() else False,
            axis=1
        )
        df_quartile["monotonic_neither"] = ~(
            df_quartile["monotonic_decreasing"] | df_quartile["monotonic_increasing"]
        )

        # We'll do rolling over windows, e.g. [21, 42]
        df_mono = pd.DataFrame(index=df_quartile.index)
        windows = [21, 42]

        # ensure index is sorted by date for rolling
        df_mono.sort_index(inplace=True)
        df_quartile.sort_index(inplace=True)

        for w in windows:
            df_mono[f"monotonic_increasing_pct_{w}"] = (
                df_quartile["monotonic_increasing"].rolling(window=w).mean()
            )
            df_mono[f"monotonic_decreasing_pct_{w}"] = (
                df_quartile["monotonic_decreasing"].rolling(window=w).mean()
            )
            df_mono[f"monotonic_neither_pct_{w}"] = (
                df_quartile["monotonic_neither"].rolling(window=w).mean()
            )

        df_mono["signal"] = signal_name

        # Convert the index (which is [date, eng_bin]) => we only want 'date' as the main index
        # We'll pick the 'date' level from the MultiIndex
        if isinstance(df_mono.index, pd.MultiIndex):
            # if the first level is 'date'
            # or we do .reset_index(level='eng_bin', drop=True)
            df_mono.reset_index(inplace=True)
            # Now we have columns => ['date','eng_bin', ...]
            # We'll group by 'date' or set index to 'date' ignoring 'eng_bin'?
            # Typically we do set_index('date')
            df_mono.set_index("date", inplace=True)
        else:
            # If it's already single-level by date, do nothing, or ensure it's datetime
            pass

        # handle if new_start_date is not None => filter out old coverage
        if new_start_date is not None:
            dt_new_start = pd.to_datetime(new_start_date)
            # ensure index is datetime
            if not isinstance(df_mono.index, pd.DatetimeIndex):
                # attempt conversion
                df_mono.index = pd.to_datetime(df_mono.index, errors="coerce")
            df_mono = df_mono.loc[df_mono.index >= dt_new_start]

        if df_mono.empty:
            print("No new rolling monotonic data to append after date filtering.")
            return

        # 4) Append new rows to monotonic_table
        self._append_to_monotonic_table(df_mono, signal_name)
        print(f"Appended {len(df_mono)} new rows for monotonic coverage of '{signal_name}'.")

    ###########################################################################
    # HELPER: CREATE TABLE if not exist
    ###########################################################################
    def _create_monotonic_table_if_not_exists(self):
        """
        Ensure that the rolling_monotonic_tables_pct_st table is created,
        with minimal columns. Adjust as needed for your final schema.
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.monotonic_table}"
        table_ref = bigquery.TableReference.from_string(full_table)

        try:
            self.bq_client.get_table(table_ref)
            # it exists
        except NotFound:
            schema = [
                bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("signal", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("monotonic_increasing_pct_21", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_decreasing_pct_21", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_neither_pct_21", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_increasing_pct_42", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_decreasing_pct_42", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("monotonic_neither_pct_42", "FLOAT", mode="NULLABLE"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            self.bq_client.create_table(table)
            print(f"Created table {full_table} with columns: {[f.name for f in schema]}")

    ###########################################################################
    # HELPER: GET MIN/MAX DATE FROM monotonic table
    ###########################################################################
    def _get_minmax_date_in_monotonic_table(self, signal_name: str):
        """
        Query rolling_monotonic_tables_pct_st to find coverage for 'signal'.
        Return (min_date, max_date).
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.monotonic_table}"
        query = f"""
        SELECT
          CAST(MIN(date) AS DATETIME) AS min_date,
          CAST(MAX(date) AS DATETIME) AS max_date
        FROM `{full_table}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("signal", "STRING", signal_name)]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)

        row = df.iloc[0]
        return (row["min_date"], row["max_date"])


    ###########################################################################
    # HELPER: FETCH RETURNS DATA
    ###########################################################################
    def _fetch_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        query_params = []

        if start_date:
            start_dt = pd.to_datetime(start_date)
            where_clauses.append("date >= @start_ts")
            query_params.append(bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start_dt))
        if end_date:
            end_dt = pd.to_datetime(end_date)
            where_clauses.append("date <= @end_ts")
            query_params.append(bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end_dt))

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT date, requestId, t1_returns
        FROM `{full_table}`
        {where_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return pd.DataFrame(columns=["date","requestId","t1_returns"])

        # parse date
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df

    ###########################################################################
    # HELPER: APPEND NEW ROWS
    ###########################################################################
    def _append_to_monotonic_table(self, df_mono: pd.DataFrame, signal_name: str):
        """
        Save rows to rolling_monotonic_tables_pct_st. 
        We expect df_mono to have an index of 'date' or at least a date column.
        We'll reset_index() to ensure 'date' is a column for BigQuery.
        """
        df_mono = df_mono.copy()
        # if 'date' is the index, we reset
        if df_mono.index.name == "date":
            df_mono.reset_index(inplace=True)
        # else, date might already be a column

        # define final columns
        final_cols = [
            "date",
            "monotonic_increasing_pct_21",
            "monotonic_decreasing_pct_21",
            "monotonic_neither_pct_21",
            "monotonic_increasing_pct_42",
            "monotonic_decreasing_pct_42",
            "monotonic_neither_pct_42",
            "signal"
        ]
        for c in final_cols:
            if c not in df_mono.columns:
                df_mono[c] = np.nan

        df_mono = df_mono[final_cols]

        # attempt to convert date col
        df_mono["date"] = pd.to_datetime(df_mono["date"], errors="coerce")

        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.monotonic_table}"
        to_gbq(
            df_mono,
            destination_table=f"{self.wmg_dataset}.{self.monotonic_table}",
            project_id=self.project_id,
            if_exists="append"
        )
        print(f"Appended {len(df_mono)} rows to {full_table} for signal '{signal_name}'.")
        


class MutualInformationCalculator:
    """
    Class to compute daily and rolling mutual information. 
    Expects a DataFrame with a MultiIndex [date, requestId] (or at least
    'date' in the index), plus columns x_col, y_col.
    """

    def __init__(self, df, x_col, y_col, window_size=42):
        """
        df: DataFrame with index including 'date'. 
        x_col: name of the signal column
        y_col: name of the returns column
        window_size: integer for rolling-window computations
        """
        # Ensure DataFrame is sorted by 'date' for rolling windows
        self.df = df.sort_index(level="date")
        self.x_col = x_col
        self.y_col = y_col
        self.window_size = window_size
        # unique dates in ascending order
        self.unique_dates = self.df.index.get_level_values("date").unique().sort_values()

    def compute_daily_mi(self) -> pd.DataFrame:
        """
        For each date, compute mutual information. 
        Return a DataFrame: index=date, column='mi'.
        """
        def compute_mi_for_group(g):
            X = g[self.x_col].values.reshape(-1,1)
            Y = g[self.y_col].values
            if len(X) < 5:
                return np.nan
            mi_val = mutual_info_regression(X, Y, discrete_features=False, random_state=0)
            return mi_val[0]

        if self.df.empty:
            return pd.DataFrame(columns=["mi"])

        # group by date level => compute MI
        daily_mi = self.df.groupby(level="date").apply(compute_mi_for_group)
        out_df = pd.DataFrame(daily_mi, columns=["mi"])
        return out_df

    def compute_rolling_mi_entire_window(self) -> pd.DataFrame:
        """
        For each consecutive rolling window of length self.window_size over unique dates,
        combine all data in that window and compute mutual information (X_col vs Y_col).
        Return DataFrame with index=end_date_of_window, column=f"mi_{self.window_size}".
        """
        def compute_mi_for_window(df_window):
            X = df_window[self.x_col].values.reshape(-1,1)
            Y = df_window[self.y_col].values
            if len(X) < 5:
                return np.nan
            mi_val = mutual_info_regression(X, Y, discrete_features=False, random_state=0)
            return mi_val[0]

        results = []
        total_windows = len(self.unique_dates) - self.window_size + 1
        if total_windows < 1:
            return pd.DataFrame(columns=[f"mi_{self.window_size}"])

        for start_idx in range(total_windows):
            # gather the dates in this rolling window
            window_dates = self.unique_dates[start_idx : start_idx + self.window_size]
            # subset df by those dates
            df_window = self.df.loc[(slice(window_dates.min(), window_dates.max())), :]
            if df_window.empty:
                continue

            window_mi = compute_mi_for_window(df_window)
            end_date = window_dates[-1]  # the last date in that window
            results.append({"date": end_date, f"mi_{self.window_size}": window_mi})

        if not results:
            return pd.DataFrame(columns=[f"mi_{self.window_size}"])

        out_df = pd.DataFrame(results).set_index("date")
        return out_df


class IncrementalMI:
    """
    A class that computes daily MI and rolling-window MI for a single signal,
    storing results incrementally in two tables: mi_daily, mi_rolling.
    """

    def __init__(
        self,
        db,
        get_signal,
        project_id="issachar-feature-library",
        wmg_dataset="wmg",
        returns_table="t1_returns",
        table_daily_mi="mi_daily",
        table_rolling_mi="mi_rolling",
        window_size=42
    ):
        """
        Args:
            db: The Database instance for BigQuery table management
            get_signal: The GetSignal instance for pulling the signal coverage
            project_id, wmg_dataset, returns_table: location of your returns
            table_daily_mi, table_rolling_mi: tables to store results
            window_size: Rolling window size for the rolling MI
        """
        self.db = db
        self.get_signal = get_signal
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.returns_table = returns_table
        self.table_daily_mi = table_daily_mi
        self.table_rolling_mi = table_rolling_mi
        self.window_size = window_size

        self.bq_client = bigquery.Client(project=self.project_id)
        # Attempt to create these tables if they don't exist
        self._create_table_if_not_exists(self.table_daily_mi, daily=True)
        self._create_table_if_not_exists(self.table_rolling_mi, daily=False)

    def run(self, signal_name: str, refresh_data: bool = False):
        """
        Main incremental logic:
          1) Check coverage in mi_daily for 'signal_name' => (min_date, max_date).
          2) If there's new coverage needed, fetch from day after max_date - 1 year in get_signal,
             or from earliest coverage if none found.
          3) Merge with returns, compute daily MI, compute rolling MI (entire window).
          4) Filter out rows prior to new_start_date
          5) Append new daily rows to mi_daily, new rolling rows to mi_rolling.
        """
        # 1) Check coverage in mi_daily
        min_mi, max_mi = self._get_minmax_date_in_table(self.table_daily_mi, signal_name)

        if max_mi is None:
            new_start_date = None
        else:
            new_start_date = (max_mi + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Pull coverage from get_signal
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,
            end_date=None,
            refresh_data=refresh_data
        )
        if df_signal.empty:
            print(f"No signal coverage for '{signal_name}' after date={new_start_date}. Stopping.")
            return

        # If partial coverage, do from new_start_date - 1 year for returns
        if new_start_date:
            start_for_returns_dt = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns = start_for_returns_dt.strftime("%Y-%m-%d")
        else:
            start_for_returns = None

        df_returns = self._fetch_returns_in_date_range(start_for_returns, None)
        if df_returns.empty:
            print("No returns data found. Stopping.")
            return

        # Merge
        df_signal.set_index(["date","requestId"], inplace=True)
        df_returns.set_index(["date","requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner")

        # Possibly drop rows missing either column
        df_merged.dropna(subset=[signal_name, "t1_returns"], how="any", inplace=True)
        if df_merged.empty:
            print("No overlapping data after merges. Stopping.")
            return

        # 3) Build the MI calculator
        micalc = MutualInformationCalculator(
            df_merged,
            x_col=signal_name,
            y_col="t1_returns",
            window_size=self.window_size
        )

        # compute daily MI
        daily_mi_df = micalc.compute_daily_mi()  # index=date, col=[mi]
        if not daily_mi_df.empty:
            daily_mi_df["signal"] = signal_name

        # compute entire-window rolling MI
        rolling_mi_df = micalc.compute_rolling_mi_entire_window()  # index=date, col=[mi_XX]
        if not rolling_mi_df.empty:
            rolling_mi_df["signal"] = signal_name

        # 4) If new_start_date, filter out older rows
        if new_start_date:
            dt_ns = pd.to_datetime(new_start_date)
            # Convert index to datetime if not already
            if not daily_mi_df.empty:
                daily_mi_df.index = pd.to_datetime(daily_mi_df.index, errors="coerce")
                daily_mi_df = daily_mi_df.loc[daily_mi_df.index >= dt_ns]
            if not rolling_mi_df.empty:
                rolling_mi_df.index = pd.to_datetime(rolling_mi_df.index, errors="coerce")
                rolling_mi_df = rolling_mi_df.loc[rolling_mi_df.index >= dt_ns]

        # check empties
        if daily_mi_df.empty and rolling_mi_df.empty:
            print(f"No new MI rows to upload for '{signal_name}'.")
            return

        # 5) Append daily results => mi_daily
        if not daily_mi_df.empty:
            self._append_to_table(daily_mi_df, self.table_daily_mi, daily=True)
        # 6) Append rolling => mi_rolling
        if not rolling_mi_df.empty:
            self._append_to_table(rolling_mi_df, self.table_rolling_mi, daily=False)

        print(f"Incremental MI update complete for '{signal_name}'.")

    ###########################################################################
    # Internal helpers
    ###########################################################################
    def _create_table_if_not_exists(self, table_name: str, daily=True):
        """
        Create table if doesn't exist. We'll define minimal columns.
        For daily => columns: date TIMESTAMP, mi FLOAT, signal STRING
        For rolling => columns: date TIMESTAMP, mi_XX FLOAT, signal STRING
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        table_ref = bigquery.TableReference.from_string(full_table)

        try:
            self.bq_client.get_table(table_ref)
            # table exists
        except NotFound:
            # need to create
            schema = [
                bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("signal", "STRING", mode="REQUIRED"),
            ]
            if daily:
                # daily => 'mi'
                schema.append(bigquery.SchemaField("mi", "FLOAT", mode="NULLABLE"))
            else:
                # rolling => e.g. 'mi_42'
                # we'll call it 'mi_rolling' for a generic placeholder, or add more.
                schema.append(bigquery.SchemaField(f"mi_{self.window_size}", "FLOAT", mode="NULLABLE"))

            table = bigquery.Table(table_ref, schema=schema)
            self.bq_client.create_table(table)
            print(f"Created table {full_table} with columns {[f.name for f in schema]}")

    def _get_minmax_date_in_table(self, table_name: str, signal_name: str):
        """
        Return (min_date, max_date) from table_name for the given signal_name.
        If none, returns (None, None).
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        query = f"""
        SELECT
          CAST(MIN(date) AS DATETIME) AS min_date,
          CAST(MAX(date) AS DATETIME) AS max_date
        FROM `{full_table}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("signal", "STRING", signal_name)]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    def _fetch_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        params = []

        if start_date:
            start_dt = pd.to_datetime(start_date)
            where_clauses.append("date >= @start_date_ts")
            params.append(bigquery.ScalarQueryParameter("start_date_ts", "TIMESTAMP", start_dt))
        if end_date:
            end_dt = pd.to_datetime(end_date)
            where_clauses.append("date <= @end_date_ts")
            params.append(bigquery.ScalarQueryParameter("end_date_ts", "TIMESTAMP", end_dt))

        w_clause = ""
        if where_clauses:
            w_clause = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT date, requestId, t1_returns
        FROM `{full_table}`
        {w_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return pd.DataFrame(columns=["date","requestId","t1_returns"])

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df

    def _append_to_table(self, df_mi: pd.DataFrame, table_name: str, daily=True):
        """
        Append new rows to table_name. We'll expect:
         - For daily => columns: [mi, signal], index=date
         - For rolling => columns: [mi_42, signal], index=date
        We'll rename and ensure a 'date' column is present, plus 'signal'.
        """
        df_mi = df_mi.copy()
        df_mi.reset_index(inplace=True)  # => [date, mi, signal] or [date, mi_42, signal] ...
        if daily:
            # ensure we have 'mi' if columns might differ
            if "mi" not in df_mi.columns:
                df_mi["mi"] = np.nan
            final_cols = ["date", "mi", "signal"]
        else:
            # rolling => e.g. 'mi_42'
            rolling_col = [c for c in df_mi.columns if c.startswith("mi_")]
            if not rolling_col:
                col_name = f"mi_{self.window_size}"
                rolling_col = [col_name]
                df_mi[col_name] = np.nan
            else:
                col_name = rolling_col[0]
            final_cols = ["date", col_name, "signal"]

        # Ensure any missing columns exist
        for c in final_cols:
            if c not in df_mi.columns:
                df_mi[c] = np.nan

        df_mi = df_mi[final_cols]

        # Convert date to Datetime
        df_mi["date"] = pd.to_datetime(df_mi["date"], errors="coerce")

        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        to_gbq(df_mi, full_table, project_id=self.project_id, if_exists="append")
        print(f"Appended {len(df_mi)} rows into {full_table}.")
        



class FvalueCalculator:
    """
    Calculates daily F-values for each date (like an OLS-based F-stat).
    Also supports a rolling-transformation method that can produce rolling medians,
    subtractions, etc. 
    """

    def __init__(self, df, signal_col, returns_col="t1_returns"):
        """
        df: DataFrame with MultiIndex [date, requestId], containing 
            columns [signal_col, returns_col].
        signal_col: e.g. the name of your feature column
        returns_col: e.g. 't1_returns'
        """
        self.df = df.copy()
        # ensure sorted by date if multiindex
        if "date" in self.df.index.names:
            self.df = self.df.sort_index(level="date")
        self.signal_col = signal_col
        self.returns_col = returns_col

    def compute_daily_fvalues(self) -> pd.DataFrame:
        """
        For each date in the index, do:
          1) rank the signal & returns columns (pct=True, ascending=False)
          2) run OLS(returns ~ signal), get model.fvalue as 'fvalue'
        Return a DataFrame with index=date, column='fvalue'.
        """
        if self.df.empty:
            return pd.DataFrame(columns=["fvalue"])

        # get unique dates
        date_level = self.df.index.get_level_values("date").unique().sort_values()
        results = []
        for dt in date_level:
            # sub-group
            day_data = self.df.xs(dt, level="date").dropna(subset=[self.signal_col, self.returns_col])
            if len(day_data) < 10:
                # skip if < 10 rows
                results.append({"date": dt, "fvalue": np.nan})
                continue
            if day_data[self.signal_col].nunique() < 10:
                # skip if < 10 unique values
                results.append({"date": dt, "fvalue": np.nan})
                continue

            # rank both columns
            day_ranked = day_data[[self.signal_col, self.returns_col]].rank(pct=True, ascending=False)
            X = day_ranked[self.signal_col]
            y = day_ranked[self.returns_col]

            # Add constant for OLS
            Xc = sm.add_constant(X)
            model = sm.OLS(y, Xc).fit()
            results.append({"date": dt, "fvalue": model.fvalue})

        if not results:
            return pd.DataFrame(columns=["fvalue"])

        out_df = pd.DataFrame(results).set_index("date").sort_index()
        return out_df

    def compute_rolling_interactions(self, df_daily_f: pd.DataFrame) -> pd.DataFrame:
        """
        Suppose df_daily_f has index=date, columns=[fvalue, signal].
        This function can do rolling transformations, e.g. rolling median of fvalue, 
        differences, etc.
        Return a DataFrame also indexed by date, containing extra columns.

        Example logic:
          - rolling medians for [21, 42]
          - subtractions of pairs of medians
        Modify as needed to suit your production logic.
        """
        if df_daily_f.empty:
            return pd.DataFrame()

        # ensure we have a 'fvalue' column
        if "fvalue" not in df_daily_f.columns:
            df_daily_f["fvalue"] = np.nan

        # We'll do rolling windows [21, 42]
        periods_list = [21, 42]

        df_roll = df_daily_f[["fvalue"]].copy()
        for w in periods_list:
            col_median = f"fvalue_median_{w}"
            df_roll[col_median] = df_roll["fvalue"].rolling(window=w).median()

        # we can do pairwise subtractions
        # for demonstration, subtract each pair of medians
        columns_for_sub = [c for c in df_roll.columns if c.startswith("fvalue_median_")]
        df_sub = pd.DataFrame(index=df_roll.index)
        for i in range(len(columns_for_sub)):
            for j in range(i+1, len(columns_for_sub)):
                col1 = columns_for_sub[i]
                col2 = columns_for_sub[j]
                new_col = f"{col1}_minus_{col2}"
                df_sub[new_col] = df_roll[col1] - df_roll[col2]

        # combine them
        df_out = pd.concat([df_roll, df_sub], axis=1)
        # join the original daily_f for referencing signal etc.
        df_out = df_out.join(df_daily_f.drop(columns=["fvalue"]), how="left")
        return df_out

class IncrementalFvalue:
    """
    Computes daily F-values + rolling transformations for a single signal,
    storing results in two BigQuery tables: table_daily_fval, table_rolling_fval.
    """

    def __init__(
        self,
        db,
        get_signal,
        project_id="issachar-feature-library",
        wmg_dataset="wmg",
        returns_table="t1_returns",
        table_daily_fval="fvalue_daily",
        table_rolling_fval="fvalue_rolling"
    ):
        """
        db: Database instance (manages BQ interactions)
        get_signal: GetSignal instance
        project_id, wmg_dataset, returns_table: Where returns & your table live.
        table_daily_fval, table_rolling_fval: Where to store daily vs rolling results.
        """
        self.db = db
        self.get_signal = get_signal
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.returns_table = returns_table
        self.table_daily_fval = table_daily_fval
        self.table_rolling_fval = table_rolling_fval

        self.bq_client = bigquery.Client(project=self.project_id)
        # Attempt to create the daily + rolling tables if they don't exist
        self._create_table_if_not_exists(self.table_daily_fval, daily=True)
        self._create_table_if_not_exists(self.table_rolling_fval, daily=False)

    def run(self, signal_name: str, refresh_data: bool = False):
        """
        Main flow:
          1) Check coverage in table_daily_fval => (min, max).
          2) Pull coverage from get_signal.
          3) Merge with returns, compute daily fvalues.
          4) Filter out old coverage => remove rows < new_start_date.
          5) Append daily rows => table_daily_fval.
          6) Rolling transformations => append => table_rolling_fval.
        """
        # 1) Check coverage
        min_cov, max_cov = self._get_minmax_date_in_table(self.table_daily_fval, signal_name)
        if max_cov is None:
            new_start_date = None
        else:
            new_start_date = (max_cov + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Get coverage from repository
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,
            end_date=None,
            refresh_data=refresh_data
        )
        if df_signal.empty:
            print(f"No new coverage for '{signal_name}'. Stopping.")
            return

        # 3) Merge with returns
        if new_start_date:
            dt_returns_start = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns = dt_returns_start.strftime("%Y-%m-%d")
        else:
            start_for_returns = None

        df_returns = self._fetch_returns_in_date_range(start_for_returns, None)
        if df_returns.empty:
            print("No returns found. Stopping.")
            return

        # set index
        df_signal.set_index(["date", "requestId"], inplace=True)
        df_returns.set_index(["date", "requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner").dropna()
        if df_merged.empty:
            print("No overlapping data after merges. Stopping.")
            return

        # 4) Build FvalueCalculator
        fcalc = FvalueCalculator(
            df_merged,
            signal_col=signal_name,
            returns_col="t1_returns"
        )

        # compute daily fvalues
        daily_f_df = fcalc.compute_daily_fvalues()  # index=date, col=[fvalue]
        if daily_f_df.empty:
            print("No daily fvalues computed. Stopping.")
            return
        daily_f_df["signal"] = signal_name

        # filter out old coverage if new_start_date
        if new_start_date:
            dt_ns = pd.to_datetime(new_start_date)
            # ensure index is datetime
            daily_f_df.index = pd.to_datetime(daily_f_df.index, errors="coerce")
            daily_f_df = daily_f_df.loc[daily_f_df.index >= dt_ns]
        if daily_f_df.empty:
            print("No new daily F-values after date filter. Stopping.")
            return

        # append daily
        self._append_to_table(daily_f_df, self.table_daily_fval, daily=True)

        # 5) rolling transformations
        rolling_df = fcalc.compute_rolling_interactions(daily_f_df)
        if rolling_df.empty:
            print("No new rolling transformations. Done.")
            return

        if new_start_date:
            # filter out old coverage
            rolling_df.index = pd.to_datetime(rolling_df.index, errors="coerce")
            rolling_df = rolling_df.loc[rolling_df.index >= dt_ns]
        if rolling_df.empty:
            print("No rolling transformations after date filter. Done.")
            return

        rolling_df["signal"] = signal_name
        self._append_to_table(rolling_df, self.table_rolling_fval, daily=False)
        print(f"Incremental Fvalue update complete for '{signal_name}'.")

    ###########################################################################
    # TABLE CREATION
    ###########################################################################
    def _create_table_if_not_exists(self, table_name: str, daily=True):
        """
        Create the table if it does not exist. 
        For daily => minimal columns: date TIMESTAMP, fvalue FLOAT, signal STRING
        For rolling => we can store columns for the transformations as needed.
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        table_ref = bigquery.TableReference.from_string(full_table)

        try:
            self.bq_client.get_table(table_ref)
            # table exists
        except NotFound:
            # create
            schema = [
                bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("signal", "STRING", mode="REQUIRED")
            ]
            if daily:
                # daily => 'fvalue'
                schema.append(bigquery.SchemaField("fvalue", "FLOAT", mode="NULLABLE"))
            else:
                # rolling => store columns like [fvalue_median_21, etc.]
                # We'll define some placeholder columns. You can expand as needed.
                schema.append(bigquery.SchemaField("fvalue_median_21", "FLOAT", mode="NULLABLE"))
                schema.append(bigquery.SchemaField("fvalue_median_42", "FLOAT", mode="NULLABLE"))
                schema.append(bigquery.SchemaField("fvalue_median_21_minus_fvalue_median_42", "FLOAT", mode="NULLABLE"))
            table = bigquery.Table(table_ref, schema=schema)
            self.bq_client.create_table(table)
            print(f"Created table {full_table} with columns {[f.name for f in schema]}")

    ###########################################################################
    # GET MIN/MAX DATE
    ###########################################################################
    def _get_minmax_date_in_table(self, table_name: str, signal_name: str):
        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        query = f"""
        SELECT
          CAST(MIN(date) AS DATETIME) AS min_date,
          CAST(MAX(date) AS DATETIME) AS max_date
        FROM `{full_table}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("signal", "STRING", signal_name)]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    ###########################################################################
    # FETCH RETURNS
    ###########################################################################
    def _fetch_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        params = []

        if start_date:
            start_dt = pd.to_datetime(start_date)
            where_clauses.append("date >= @start_date_ts")
            params.append(bigquery.ScalarQueryParameter("start_date_ts", "TIMESTAMP", start_dt))
        if end_date:
            end_dt = pd.to_datetime(end_date)
            where_clauses.append("date <= @end_date_ts")
            params.append(bigquery.ScalarQueryParameter("end_date_ts", "TIMESTAMP", end_dt))

        w_clause = ""
        if where_clauses:
            w_clause = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT date, requestId, t1_returns
        FROM `{full_table}`
        {w_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return pd.DataFrame(columns=["date","requestId","t1_returns"])

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df

    ###########################################################################
    # APPEND DATA
    ###########################################################################
    def _append_to_table(self, df_fval: pd.DataFrame, table_name: str, daily=True):
        """
        Append rows to the table_name. 
        If daily=True => columns: [date, fvalue, signal]
        else => includes rolling columns
        """
        df_fval = df_fval.copy()
        df_fval.reset_index(inplace=True)  # ensure 'date' is a column
        # For daily, keep at least [date, fvalue, signal]
        # For rolling, keep all columns but ensure we have 'date' & 'signal'

        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        to_gbq(df_fval, full_table, project_id=self.project_id, if_exists="append")
        print(f"Appended {len(df_fval)} rows into {full_table}.")
        
        
def get_unique_signals_from_gbq(project_id='issachar-feature-library', 
                                     dataset_name='wmg', 
                                     table_name='ic_daily2'):
    """
    Fetch all unique signal values from the 'signal' column in the daily_ic table.
    
    Args:
        project_id (str): The Google Cloud project ID.
        dataset_name (str): The BigQuery dataset name.
        table_name (str): The BigQuery table name containing the daily IC results.
    
    Returns:
        list: A list of unique signal names.
    """
    # Create a BigQuery client.
    client = bigquery.Client(project=project_id)
    
    # Construct the query to select distinct signals.
    query = f"""
    SELECT DISTINCT signal
    FROM `{project_id}.{dataset_name}.{table_name}`
    """
    
    # Execute the query.
    query_job = client.query(query)
    
    # Convert the result to a pandas DataFrame.
    df = query_job.to_dataframe()
    
    # Return the unique signals as a list.
    return df['signal'].unique().tolist()




class OscillationCalculator:
    """
    Given a DataFrame with columns [signal_col], plus possibly some returns columns,
    run a single pass of FFT + sine-wave fit to find amplitude, frequency, phase, offset, etc.
    """

    def __init__(self, df, signal_col):
        """
        df: DataFrame with index = date or [date, requestId]
            containing columns => [signal_col].
            For best results, ensure it's sorted by date or by index.
        signal_col: string for the column name we want to analyze
        """
        self.df = df
        self.signal_col = signal_col

    def _sine_wave(self, t, A, f, phi, C):
        # The model: A * sin(2 pi f t + phi) + C
        return A * np.sin(2 * np.pi * f * t + phi) + C

    def analyze_signal(self) -> pd.DataFrame:
        """
        1) Extract the signal as a series
        2) Do FFT to find dominant frequency & magnitude
        3) Fit a sine wave with initial guess -> (A=1, freq=dominant freq, phase=0, offset=0)
        4) Return a DataFrame with one row: 
           [dominant_magnitude, dominant_frequency, amplitude, frequency, phase, offset].
        """
        if self.df.empty or self.signal_col not in self.df.columns:
            # no data
            return pd.DataFrame([], columns=[
                "dominant_magnitude","dominant_frequency","amplitude","frequency","phase","offset"
            ])

        signal = self.df[self.signal_col].dropna()
        if signal.empty:
            return pd.DataFrame([], columns=[
                "dominant_magnitude","dominant_frequency","amplitude","frequency","phase","offset"
            ])

        # 2) FFT
        fft_values = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_values))
        fft_magnitude = np.abs(fft_values)

        dominant_freq_idx = np.argmax(fft_magnitude)
        dominant_frequency = frequencies[dominant_freq_idx]
        dominant_magnitude = fft_magnitude[dominant_freq_idx]

        # 3) We'll define t = range(len(signal)) for the curve_fit
        t = np.arange(len(signal))

        # initial guess
        initial_guess = [1, dominant_frequency, 0, 0]  # A=1, freq=dominant, phase=0, offset=0

        # fallback values
        amplitude, frequency, phase, offset = np.nan, np.nan, np.nan, np.nan

        # We need at least 4 data points for a 4-parameter fit
        if len(signal) >= 4:
            try:
                params, _ = curve_fit(self._sine_wave, t, signal, p0=initial_guess)
                amplitude, frequency, phase, offset = params
            except RuntimeError as e:
                # Fitting might fail
                print(f"Sine wave fit failed: {e}")
        else:
            print("Not enough data points to do a 4-parameter sine fit.")

        # Return as 1-row DataFrame
        results = {
            "dominant_magnitude": round(dominant_magnitude, 3),
            "dominant_frequency": round(dominant_frequency, 3),
            "amplitude": round(amplitude, 3),
            "frequency": round(frequency, 3),
            "phase": round(phase, 3),
            "offset": round(offset, 3)
        }
        return pd.DataFrame([results])
    
    
import pandas as pd
import numpy as np
from datetime import timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from pandas_gbq import to_gbq
from typing import Optional

def safe_qcut(series):
    """
    Attempt to qcut a group into 4 bins (quartiles).
    If not enough unique values, return NaNs.
    """
    if len(series) >= 4:
        try:
            return pd.qcut(series, 4, labels=False, duplicates='drop')
        except ValueError:
            return pd.Series(np.nan, index=series.index)
    else:
        return pd.Series(np.nan, index=series.index)

class SpreadReturns:
    """
    Incrementally calculates 'spread_returns' for a given signal, storing results
    in daily_spread_returns (which has columns: [date, signal, spread_returns]).
    
    Steps:
      1) Check existing coverage in daily_spread_returns => last date for that signal
      2) Pull coverage from get_signal for new coverage
      3) Merge coverage with t1_returns
      4) rank => safe_qcut => group => mean => unstack => define 'spread_returns'
      5) Upload new rows to daily_spread_returns
    """

    def __init__(
        self,
        db,                 # Database instance (managing BQ signal_lookup etc.)
        get_signal,         # The GetSignal class for pulling signals from repository
        project_id="issachar-feature-library",
        wmg_dataset="wmg",
        daily_spread_table="daily_spread_returns",
        returns_table="t1_returns"
    ):
        """
        db: your Database instance
        get_signal: your GetSignal instance
        daily_spread_table: The BQ table storing [date, signal, spread_returns]
        returns_table: The table name for t1_returns
        """
        self.db = db
        self.get_signal = get_signal
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.daily_spread_table = daily_spread_table
        self.returns_table = returns_table

        self.bq_client = bigquery.Client(project=self.project_id)
        self._create_table_if_not_exists()

    def run(self, signal_name: str, refresh_data: bool = False):
        """
        Main flow:
          1) Check coverage in daily_spread_returns => find last date for this signal
          2) Pull coverage from get_signal (start_date = last_date + 1 if partial)
          3) Also fetch t1_returns from last_date - 1 year if partial
          4) rank => safe_qcut => group => unstack => define daily spread
          5) insert new rows => daily_spread_returns
        """
        # 1) Check existing coverage
        max_date = self._get_max_date_in_spread(signal_name)
        if max_date is None:
            new_start_date = None
        else:
            new_start_date = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Pull coverage from get_signal
        df_signal = self.get_signal.run(
            signal_name, 
            start_date=new_start_date,
            end_date=None,
            refresh_data=refresh_data
        )
        if df_signal.empty:
            print(f"No coverage for '{signal_name}' after {new_start_date}. Nothing to do.")
            return

        # 3) Also fetch t1_returns (from new_start_date - 1 year for context)
        if new_start_date:
            dt_for_returns = pd.to_datetime(new_start_date) - timedelta(days=365)
            returns_start_str = dt_for_returns.strftime("%Y-%m-%d")
        else:
            returns_start_str = None

        df_returns = self._fetch_t1_returns_in_date_range(returns_start_str, None)
        if df_returns.empty:
            print("No returns data found. Stopping.")
            return

        # set index => [date, requestId], if not already
        df_signal.set_index(["date","requestId"], inplace=True)
        df_returns.set_index(["date","requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner").dropna()
        if df_merged.empty:
            print("No overlapping coverage with t1_returns. Stopping.")
            return

        # 4) Perform the daily rank => safe_qcut => group => mean => unstack
        # We'll store the difference between top quartile (3) - bottom quartile (0)
        # as the spread. Or you can store multiple quartiles if you prefer.
        df_merged.reset_index(inplace=True)
        df_merged["rank"] = df_merged.groupby("date")[signal_name].rank(pct=True, ascending=False)
        df_merged["eng_bin"] = df_merged.groupby("date")["rank"].transform(safe_qcut)

        df_quartile = df_merged.groupby(["date","eng_bin"])["t1_returns"].mean().unstack()

        # if eng_bin yields columns [0,1,2,3], we do:
        needed_bins = [0,1,2,3]
        for b in needed_bins:
            if b not in df_quartile.columns:
                df_quartile[b] = np.nan

        # define 'spread_returns' = top quartile (3) - bottom quartile (0)
        df_quartile["spread_returns"] = df_quartile[3] - df_quartile[0]
        # store a single spread per date
        df_spread = df_quartile[["spread_returns"]].copy()

        # 5) Filter out old coverage if new_start_date
        if new_start_date:
            dt_ns = pd.to_datetime(new_start_date)
            df_spread = df_spread.loc[df_spread.index >= dt_ns]

        if df_spread.empty:
            print("No new daily spreads. Stopping.")
            return

        # 6) Insert => daily_spread_returns, with columns [date, signal, spread_returns]
        # We'll index => date, store => one row per date
        df_spread.reset_index(inplace=True)
        df_spread.rename(columns={"spread_returns": "spread_returns"}, inplace=True)
        df_spread["signal"] = signal_name
        # final columns => [date, signal, spread_returns]
        final_cols = ["date","signal","spread_returns"]
        df_spread = df_spread[final_cols]
        # convert date
        df_spread["date"] = pd.to_datetime(df_spread["date"]).dt.tz_localize(None)

        # upload
        self._append_spread(df_spread)
        print(f"Inserted {len(df_spread)} new daily_spread rows for signal='{signal_name}'.")

    ############################################################################
    # Internal Helpers
    ############################################################################

    def _create_table_if_not_exists(self):
        """
        Ensures daily_spread_returns table has at least columns:
          date (TIMESTAMP), signal (STRING), spread_returns (FLOAT).
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.daily_spread_table}"
        table_ref = bigquery.TableReference.from_string(full_table)
        try:
            self.bq_client.get_table(table_ref)
            # table exists
        except NotFound:
            schema = [
                bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("signal", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("spread_returns", "FLOAT", mode="NULLABLE")
            ]
            table = bigquery.Table(table_ref, schema=schema)
            self.bq_client.create_table(table)
            print(f"Created table {full_table} with schema: date/timestamp, signal/string, spread_returns/float")

    def _get_max_date_in_spread(self, signal_name: str) -> Optional[pd.Timestamp]:
        """
        Return the maximum 'date' we have in daily_spread_returns for the given signal,
        or None if no coverage.
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.daily_spread_table}"
        query = f"""
        SELECT CAST(MAX(date) AS TIMESTAMP) as max_date
        FROM `{full_table}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("signal", "STRING", signal_name)
            ]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().any():
            return None
        return df.iloc[0]["max_date"]

    def _fetch_t1_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Query wmg.<returns_table> for columns [date, requestId, t1_returns] 
        from [start_date..end_date].
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        params = []
        if start_date:
            dt_start = pd.to_datetime(start_date)
            where_clauses.append("date >= @start_ts")
            params.append(bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", dt_start))
        if end_date:
            dt_end = pd.to_datetime(end_date)
            where_clauses.append("date <= @end_ts")
            params.append(bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", dt_end))

        w_clause = ""
        if where_clauses:
            w_clause = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT date, requestId, t1_returns
        FROM `{full_table}`
        {w_clause}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return pd.DataFrame(columns=["date","requestId","t1_returns"])

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df

    def _append_spread(self, df_spread: pd.DataFrame):
        """
        Append new rows to daily_spread_returns with columns => [date, signal, spread_returns].
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.daily_spread_table}"
        to_gbq(
            df_spread,
            destination_table=f"{self.wmg_dataset}.{self.daily_spread_table}",
            project_id=self.project_id,
            if_exists="append"
        )
        print(f"Appended {len(df_spread)} rows into {full_table}.")



import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.optimize import curve_fit
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from pandas_gbq import to_gbq

class OscillationCalculator:
    """
    Performs a single FFT + sine-wave fit on a single time series column,
    returning amplitude, frequency, etc.
    """
    def __init__(self, dates: pd.Series, values: pd.Series):
        """
        dates:  A pandas Series (or array) of timestamps (one per row).
        values: The corresponding numeric values (e.g. spread_returns).
                Must be the same length/order as dates.
        """
        self.dates = dates
        self.values = values

    def _sine_wave(self, t, A, f, phi, C):
        return A * np.sin(2*np.pi*f*t + phi) + C

    def analyze_signal(self) -> pd.DataFrame:
        """
        1) FFT to find dominant frequency & magnitude
        2) Sine-wave fit => amplitude, freq, phase, offset
        3) Return a 1-row DataFrame with columns:
           [dominant_magnitude, dominant_frequency, amplitude, frequency, phase, offset]
        """
        # Drop any NaN in 'values'
        valid_mask = self.values.notna()
        dates = self.dates[valid_mask]
        signal = self.values[valid_mask]
        if len(signal) < 4:
            # <4 data points => can't fit 4-parameter sine wave
            return pd.DataFrame(columns=[
                "dominant_magnitude","dominant_frequency","amplitude","frequency","phase","offset"
            ])

        # For an FFT, we need an evenly spaced time dimension. We'll assume each row is 1 step apart
        # or we just treat them as consecutive indices. If you want to account for actual date intervals,
        # you'd need to reindex or transform them.
        t = np.arange(len(signal))

        # 1) FFT
        fft_values = np.fft.fft(signal.values)
        freqs = np.fft.fftfreq(len(fft_values))
        fft_magnitude = np.abs(fft_values)

        dom_idx = np.argmax(fft_magnitude)
        dom_freq = freqs[dom_idx]
        dom_mag = fft_magnitude[dom_idx]

        # 2) Sine-wave fit
        initial_guess = [1.0, dom_freq, 0.0, 0.0]
        amplitude, frequency, phase, offset = np.nan, np.nan, np.nan, np.nan

        try:
            params, _ = curve_fit(self._sine_wave, t, signal.values, p0=initial_guess)
            amplitude, frequency, phase, offset = params
        except RuntimeError as e:
            print(f"Sine wave fit error: {e}")
            print("Using NaN for amplitude/freq/phase/offset.")

        results = {
            "dominant_magnitude": round(dom_mag, 3),
            "dominant_frequency": round(dom_freq, 3),
            "amplitude": round(amplitude, 3),
            "frequency": round(frequency, 3),
            "phase": round(phase, 3),
            "offset": round(offset, 3),
        }
        return pd.DataFrame([results])

class OscillationFromSpread:
    """
    Computes oscillation metrics for a single signal by:
      1) Querying daily_spread_returns for that signal => [date, spread_returns].
      2) Doing a single FFT + sine-wave fit => returns amplitude, frequency, etc.
      3) (Optionally) storing or returning that result.

    Because we no longer need (requestId, t1_returns), this class
    simply uses 'daily_spread_returns' as the time series for each signal.
    """
    def __init__(
        self,
        project_id: str = "issachar-feature-library",
        dataset_name: str = "wmg",
        daily_spread_table: str = "daily_spread_returns",
        oscillation_table: str = "daily_oscillation_metrics"  # optional
    ):
        """
        project_id, dataset_name: BigQuery references
        daily_spread_table: The BQ table storing [date, signal, spread_returns]
        oscillation_table: Optional table if you want to store the results
        """
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.daily_spread_table = daily_spread_table
        self.oscillation_table = oscillation_table  # optional, if you want to upload
        self.bq_client = bigquery.Client(project=self.project_id)

    def run(self, signal_name: str) -> pd.DataFrame:
        """
        1) Pull entire coverage for that signal from daily_spread_returns => [date, spread_returns].
        2) Sort by date => do a single pass FFT + sine wave fit => returns a 1-row DF with amplitude, freq, etc.
        3) Return that DF. (Optionally store it in self.oscillation_table).
        """
        df_spread = self._fetch_spread_for_signal(signal_name)
        if df_spread.empty:
            print(f"No spread coverage found for signal '{signal_name}'. Exiting.")
            return pd.DataFrame()

        # We now have columns => [date, signal, spread_returns], filter out the chosen signal
        df_spread = df_spread[df_spread["signal"] == signal_name].copy()
        df_spread.sort_values("date", inplace=True)
        df_spread.dropna(subset=["spread_returns"], inplace=True)

        # Build an OscillationCalculator
        calc = OscillationCalculator(
            dates=df_spread["date"],
            values=df_spread["spread_returns"]
        )
        res_df = calc.analyze_signal()
        if res_df.empty:
            print(f"Oscillation analysis produced no results (insufficient data).")
            return pd.DataFrame()

        # We'll optionally store 'date' as the last coverage date
        coverage_max_date = df_spread["date"].max()
        res_df["date"] = coverage_max_date
        res_df["signal"] = signal_name
        # => columns [dominant_magnitude, ..., offset, date, signal]

        # Optionally store in an 'oscillation_table' if desired
        # We'll do that below
        if self.oscillation_table:
            self._append_oscillation_results(res_df)

        return res_df

    # Helpers
    def _fetch_spread_for_signal(self, signal_name: str) -> pd.DataFrame:
        """
        Query daily_spread_returns for [date, signal, spread_returns], matching signal = @signal_name.
        Return as DataFrame => columns [date, signal, spread_returns].
        """
        full_table = f"{self.project_id}.{self.dataset_name}.{self.daily_spread_table}"
        query = f"""
        SELECT date, signal, spread_returns
        FROM `{full_table}`
        WHERE signal = @sig
        ORDER BY date
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("sig", "STRING", signal_name)]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty:
            return df
        # Convert date column to naive Python datetime
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    def _append_oscillation_results(self, df_res: pd.DataFrame):
        """
        If you want to store the final 1-row result in BQ, create
        an 'oscillation_table' with columns like:
          date TIMESTAMP
          signal STRING
          dominant_magnitude FLOAT
          dominant_frequency FLOAT
          amplitude FLOAT
          frequency FLOAT
          phase FLOAT
          offset FLOAT
        """
        full_table = f"{self.project_id}.{self.dataset_name}.{self.oscillation_table}"
        # optionally ensure table is created if not exist ...
        df_res = df_res.copy()
        df_res.reset_index(drop=True, inplace=True)

        # keep columns
        keep_cols = [
            "date","signal","dominant_magnitude","dominant_frequency",
            "amplitude","frequency","phase","offset"
        ]
        for c in keep_cols:
            if c not in df_res.columns:
                df_res[c] = np.nan
        df_res = df_res[keep_cols]
        df_res["date"] = pd.to_datetime(df_res["date"], errors="coerce")

        to_gbq(
            df_res,
            destination_table=f"{self.dataset_name}.{self.oscillation_table}",
            project_id=self.project_id,
            if_exists="append"
        )
        print(f"Appended 1 row of oscillation metrics to {full_table}.")

if __name__ == '__main__':
    
    # Pull all signals from GBQ
    all_signals = get_unique_signals_from_gbq()

    # 1) Create Database
    db = Database(project_id="issachar-feature-library")
    
    # 2) Create GetSignal
    get_signal = GetSignal(db=db)
    
    # Loop through each, and refresh any metrics, need be
    for signal_name in all_signals:

        # 3) Create IncrementalICCalculator
        ic_calc = IncrementalICCalculator(
            db=db, 
            get_signal=get_signal, 
            project_id="issachar-feature-library",
            wmg_dataset="wmg",
            returns_table="t1_returns",
            daily_ic2_table="daily_ic2"
        )

        # 4) Compute incremental IC
        ic_calc.run(signal_name, rank_signal=True, refresh_data = False)
        # => This triggers the get_signal logic to see if there's new coverage in baseline,
        #    rebuild if needed, then merges with returns data, calculates daily IC, 
        #    and appends new rows to daily_ic2.

        mon_calc = IncrementalMonotonic(
            db=db,
            get_signal=get_signal,
            project_id="issachar-feature-library",
            wmg_dataset="wmg",
            returns_table="t1_returns",
            monotonic_table="rolling_monotonic_tables_pct_st"
        )
        mon_calc.run(signal_name, refresh_data = False)

        # 2) Create IncrementalMI
        mi_calc = IncrementalMI(
            db=db,
            get_signal=get_signal,
            project_id="issachar-feature-library",
            wmg_dataset="wmg",
            returns_table="t1_returns",
            table_daily_mi="mi_daily",
            table_rolling_mi="mi_rolling",
            window_size=42
        )

        # 3) Compute incremental MI for a single signal
        mi_calc.run(signal_name, refresh_data = False)

        # 4) Fvalue related stats 
        fval_calc = IncrementalFvalue(
            db=db,
            get_signal=get_signal,
            project_id="issachar-feature-library",
            wmg_dataset="wmg",
            returns_table="t1_returns",
            table_daily_fval="daily_fvalues",
            table_rolling_fval="daily_fvalue_interactions"  # if you want the name from snippet
        )

        fval_calc.run(signal_name, refresh_data=False)        
        
        # 4) Daily spread returns
        spread_calc = SpreadReturns(
            db=db,                 # your Database instance
            get_signal=get_signal, # your GetSignal instance
            project_id="issachar-feature-library",
            wmg_dataset="wmg",
            daily_spread_table="daily_spread_returns",  # The BQ table to store final results
            returns_table="t1_returns"                  # Where we get t1_returns
        )

        spread_calc.run(signal_name, refresh_data=False)

        osc_calc = IncrementalOscillation(
            db=db,
            get_signal=get_signal,
            project_id="issachar-feature-library",
            wmg_dataset="wmg",
            spread_table="daily_spread_returns",
            oscillation_table="oscillation_metrics"
        )

        # Suppose we want to do "accel_21d" with daily_spread
        osc_calc.run(signal_name, refresh_data=False)