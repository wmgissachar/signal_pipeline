import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from pandas_gbq import to_gbq
from datetime import timedelta

# Need imports
from Database import Database
from GetSignal import GetSignal
from sklearn.feature_selection import mutual_info_regression


class IncrementalICCalculator:
    """
    A class that checks how much data for a given signal is already computed
    in daily_ic2. If there's new data in the feature_set table, it pulls
    that (plus one year prior for context), computes daily IC metrics,
    and appends them to wmg.daily_ic2.
    """

    def __init__(
        self,
        db,  # your Database instance
        get_signal,  # an instance of the new GetSignal class
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
        self.get_signal = get_signal  # we can now call get_signal.run(...)
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.returns_table = returns_table
        self.daily_ic2_table = daily_ic2_table

        self.bq_client = bigquery.Client(project=self.project_id)

    def run(self, signal_name: str, rank_signal: bool = True, refresh_data = False):
        """
        1) Check coverage in daily_ic2 for this signal.
        2) If there's new coverage in baseline, we want to compute daily metrics from
           [new_start - 1 year .. new_end].
        3) We use get_signal.run(...) with refresh_data to ensure we have updated coverage.
        4) Then we pull returns & do IC calculations, appending to daily_ic2.
        """
        sig_info = self.db.get_signal(signal_name)
        if not sig_info:
            raise ValueError(f"Signal '{signal_name}' not found in the database lookup.")

        # get coverage in daily_ic2
        min_date_ic2, max_date_ic2 = self._get_daily_ic2_minmax_date(signal_name)

        # figure out coverage in baseline via get_signal? Actually, the get_signal
        # code itself can rebuild coverage. But we need the max_date from baseline
        # to decide new_start. Let’s do that from the lookup's feature_set_location or
        # we can just rely on get_signal with refresh_data to do the logic.

        # Let's assume the signal has feature_set_location if needed. 
        feature_set_loc = sig_info.get("feature_set_location")
        if not feature_set_loc:
            print(f"No feature_set_location for '{signal_name}'. Possibly a baseline or custom.")
            # We'll proceed anyway, because get_signal can handle it if left_location is set.

        # min_date_fs, max_date_fs = ...
        # Actually let's skip repeating that logic and rely on get_signal to do the rebuild.

        # If daily_ic2 has no coverage for this signal, we do [the entire coverage].
        # Otherwise, if there's new coverage, we do from day after max_date_ic2.

        if max_date_ic2 is None:
            new_start_date = None  # means from the earliest coverage
        else:
            # day after the last coverage
            new_start_date = (max_date_ic2 + timedelta(days=1)).strftime("%Y-%m-%d")

        # We'll call get_signal.run(...) with refresh_data=True to ensure coverage is up to date
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,  # or None if no coverage
            end_date=None,             # fetch all up to the present
            refresh_data=refresh_data          # let it rebuild coverage if there's new data
        )

        if df_signal.empty:
            print(f"No signal data retrieved for '{signal_name}'. Nothing to compute.")
            return

        # we also want returns for the date range we care about, i.e. [start_for_fetch..].
        # We'll do from 1 year prior to new_start_date if new_start_date is not None.
        if new_start_date is not None:
            # parse to datetime
            start_for_returns = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns_str = start_for_returns.strftime("%Y-%m-%d")
        else:
            start_for_returns_str = None

        # fetch all returns from start_for_returns_str.. present
        df_returns = self._fetch_returns_in_date_range(start_for_returns_str, None)

        # Now merge
        # we have df_signal => [date, requestId, <signal_name>]
        # we have df_returns => [date, requestId, t1_returns]
        df_signal.set_index(["date","requestId"], inplace=True)
        df_returns.set_index(["date","requestId"], inplace=True)

        df_merged = df_signal.join(df_returns, how="inner")

        # rank if needed
        if rank_signal:
            df_merged[signal_name] = df_merged.groupby("date")[signal_name].rank(pct=True)

        daily_metrics_df = self._compute_daily_metrics(
            df_merged, signal_col=signal_name, returns_col="t1_returns"
        )

        # We only want from new_start_date.. if new_start_date is not None
        if new_start_date is not None:
            dt_new_start = pd.to_datetime(new_start_date)
            daily_metrics_df = daily_metrics_df.loc[daily_metrics_df.index >= dt_new_start]

        if daily_metrics_df.empty:
            print(f"No new daily metrics computed for '{signal_name}'.")
            return

        # upload to daily_ic2
        self._append_to_daily_ic2(daily_metrics_df, signal_name)
        print(f"Appended {len(daily_metrics_df)} new daily IC rows for '{signal_name}' into daily_ic2.")

    ###########################################################################
    # copy from your existing code for _get_daily_ic2_minmax_date, _fetch_returns_in_date_range, 
    # _compute_daily_metrics, _append_to_daily_ic2
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
            query_parameters=[
                bigquery.ScalarQueryParameter("signal", "STRING", signal_name)
            ]
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
        if start_date:
            where_clauses.append("date >= @start_date")
            query_params.append(bigquery.ScalarQueryParameter("start_date", "DATETIME", start_date))
        if end_date:
            where_clauses.append("date <= @end_date")
            query_params.append(bigquery.ScalarQueryParameter("end_date", "DATETIME", end_date))

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
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df

    def _compute_daily_metrics(self, df_merged: pd.DataFrame, signal_col: str, returns_col: str) -> pd.DataFrame:
        import numpy as np
        import statsmodels.api as sm
        from scipy.stats import spearmanr

        dates = df_merged.index.get_level_values("date").unique()
        results = []
        for dt in dates:
            day_data = df_merged.xs(dt, level="date")
            all_stats = self._compute_all_ic(day_data, signal_col, returns_col)
            topq_stats = self._compute_topq_ic(day_data, signal_col, returns_col)
            row = {
                "date": dt,
                **all_stats,
                **topq_stats
            }
            results.append(row)
        out_df = pd.DataFrame(results).set_index("date")
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

        import statsmodels.api as sm
        from scipy.stats import spearmanr

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
        df_metrics = df_metrics.copy()
        df_metrics.reset_index(inplace=True)
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
        from pandas_gbq import to_gbq
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
             rolling window proportions.
          4) Append new rows to monotonic_table.
        """
        # 1) Check coverage in monotonic table
        min_mono, max_mono = self._get_minmax_date_in_monotonic_table(signal_name)

        # We'll do from day after max_mono if any coverage
        if max_mono is None:
            new_start_date = None
        else:
            new_start_date = (max_mono + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Use get_signal to fetch coverage
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,  # or None
            end_date=None,
            refresh_data=refresh_data  # if True, will rebuild coverage if baseline has more data
        )
        if df_signal.empty:
            print(f"No data retrieved for signal '{signal_name}'. Stopping.")
            return

        # We'll now pull returns from new_start_date - 1 year if partial coverage,
        # or from earliest coverage if new_start_date is None
        if new_start_date is not None:
            start_for_returns = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns_str = start_for_returns.strftime("%Y-%m-%d")
        else:
            start_for_returns_str = None

        df_returns = self._fetch_returns_in_date_range(start_for_returns_str, None)
        if df_returns.empty:
            print("No returns data. Stopping.")
            return

        # 3) Merge the two DataFrames on [date, requestId]
        df_signal.set_index(["date","requestId"], inplace=True)
        df_returns.set_index(["date","requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner")
        df_merged.reset_index(inplace=True)

        # The snippet references columns: 
        #  [date, requestId, <signal_name>, t1_returns]
        # We'll compute quartile bins in descending order? The example uses ascending=False
        # => rank -> transform safe_qcut
        df_merged["rank"] = df_merged.groupby("date")[signal_name].rank(pct=True, ascending=False)
        df_merged["eng_bin"] = df_merged.groupby("date")["rank"].transform(self.safe_qcut)

        # group by [date, eng_bin] => compute mean t1_returns
        df_quartile = df_merged.groupby(["date","eng_bin"])["t1_returns"].mean().unstack()

        # daily monotonic check: we want quartile columns 0,1,2,3 => check if strictly ascending or descending
        # or neither
        df_quartile["monotonic_decreasing"] = df_quartile.apply(
            lambda x: x[0] > x[1] > x[2] > x[3] if len(x.dropna())==4 else False,
            axis=1
        )
        df_quartile["monotonic_increasing"] = df_quartile.apply(
            lambda x: x[0] < x[1] < x[2] < x[3] if len(x.dropna())==4 else False,
            axis=1
        )
        df_quartile["monotonic_neither"] = ~(
            df_quartile["monotonic_decreasing"] | df_quartile["monotonic_increasing"]
        )

        # We'll do rolling over windows, e.g. [21, 42]
        df_mono = pd.DataFrame(index=df_quartile.index)
        windows = [21, 42]
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

        # Add signal column
        df_mono["signal"] = signal_name

        # Filter out rows prior to new_start_date (if not None) => incremental approach
        if new_start_date is not None:
            dt_new_start = pd.to_datetime(new_start_date)
            df_mono = df_mono.loc[df_mono.index >= dt_new_start]

        if df_mono.empty:
            print("No new rolling monotonic data to append.")
            return

        # 4) Append new rows to monotonic_table
        self._append_to_monotonic_table(df_mono, signal_name)
        print(f"Appended {len(df_mono)} new rows for monotonic coverage of '{signal_name}'.")

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
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
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
        """
        Similar to the logic in IncrementalICCalculator. 
        Pull t1_returns from [start_date..end_date].
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        query_params = []
        if start_date:
            where_clauses.append("date >= @start_date")
            query_params.append(bigquery.ScalarQueryParameter("start_date", "DATETIME", start_date))
        if end_date:
            where_clauses.append("date <= @end_date")
            query_params.append(bigquery.ScalarQueryParameter("end_date", "DATETIME", end_date))

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
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date","requestId"], inplace=True)
        return df

    ###########################################################################
    # HELPER: APPEND NEW ROWS
    ###########################################################################
    def _append_to_monotonic_table(self, df_mono: pd.DataFrame, signal_name: str):
        """
        Save rows to rolling_monotonic_tables_pct_st. 
        We expect the DataFrame to have index=date, columns:
          monotonic_increasing_pct_21, monotonic_decreasing_pct_21, etc., plus 'signal'.
        We'll reset_index() to ensure 'date' is a column for BigQuery.
        """
        df_mono = df_mono.copy()
        df_mono.reset_index(inplace=True)
        # rename index col from 'date' if needed
        # but we've named the index date => it's already date

        # define the final columns
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
        # fill missing
        for c in final_cols:
            if c not in df_mono.columns:
                df_mono[c] = np.nan

        df_mono = df_mono[final_cols]

        # Upload (append)
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.monotonic_table}"
        to_gbq(
            df_mono,
            full_table,
            project_id=self.project_id,
            if_exists="append"
        )


class MutualInformationCalculator:
    """
    A lightweight version of your snippet's calculator for daily
    MI and entire-window rolling MI. You can modify as needed.
    """

    def __init__(self, df, x_col, y_col, window_size=42):
        """
        df: DataFrame with a MultiIndex [date, requestId] (ideally),
            or at least a 'date' index, plus columns x_col, y_col
        x_col: signal column
        y_col: returns column
        window_size: integer for rolling window calculations
        """
        self.df = df.sort_index(level="date")  # ensure date-sorted
        self.x_col = x_col
        self.y_col = y_col
        self.window_size = window_size
        self.unique_dates = self.df.index.get_level_values("date").unique()

    def compute_daily_mi(self) -> pd.DataFrame:
        """
        Compute daily mutual information for each date. Return DataFrame:
          index=date, columns=[mi], plus a 'signal' column for reference.
        """
        def compute_mi_for_group(group):
            X = group[self.x_col].values.reshape(-1,1)
            Y = group[self.y_col].values
            if len(X) < 5:
                return np.nan
            mi_val = mutual_info_regression(X, Y, discrete_features=False, random_state=0)
            return mi_val[0]

        daily_mi = self.df.groupby(level="date").apply(compute_mi_for_group)
        # daily_mi is a Series indexed by date
        out_df = pd.DataFrame(daily_mi, columns=["mi"])
        return out_df

    def compute_rolling_mi_entire_window(self) -> pd.DataFrame:
        """
        For each rolling window (of length self.window_size) of consecutive dates,
        compute MI using the entire combined data from those dates.
        Return DataFrame with index = end_date, column = [mi_<window_size>].
        """
        def compute_mi_for_window(df_window):
            X = df_window[self.x_col].values.reshape(-1,1)
            Y = df_window[self.y_col].values
            if len(X) < 5:
                return np.nan
            val = mutual_info_regression(X, Y, discrete_features=False, random_state=0)
            return val[0]

        results = []
        total_windows = len(self.unique_dates) - self.window_size + 1
        if total_windows < 1:
            return pd.DataFrame(columns=[f"mi_{self.window_size}"])

        for start_idx in range(total_windows):
            window_dates = self.unique_dates[start_idx:start_idx + self.window_size]
            window_data = self.df.loc[window_dates]
            window_mi = compute_mi_for_window(window_data)
            end_date = window_dates[-1]
            results.append({"date": end_date, f"mi_{self.window_size}": window_mi})

        out_df = pd.DataFrame(results).set_index("date")
        return out_df


class IncrementalMI:
    """
    A class that computes daily MI and rolling-window MI for a single signal,
    storing results incrementally in two tables: mi_daily, mi_rolling.
    """

    def __init__(
        self,
        db,              # Database instance
        get_signal,      # GetSignal instance
        project_id="issachar-feature-library",
        wmg_dataset="wmg",
        returns_table="t1_returns",
        table_daily_mi="mi_daily",
        table_rolling_mi="mi_rolling",
        window_size=42
    ):
        """
        Args:
            db: The Database instance for BQ table management
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

    def run(self, signal_name: str, refresh_data: bool = True):
        """
        Main incremental logic:
          1) Check coverage in mi_daily for 'signal_name' (min_date, max_date).
          2) If there's new coverage, we fetch from day after max_date - 1 year in GetSignal,
             or from the earliest coverage if none found, etc.
          3) Merge with returns, compute daily MI, compute rolling MI (entire window).
          4) Filter out rows prior to new_start_date
          5) Append new daily rows to mi_daily, new rolling rows to mi_rolling.
        """

        # 1) Check coverage in mi_daily
        min_mi, max_mi = self._get_minmax_date_in_table(self.table_daily_mi, signal_name)

        if max_mi is None:
            # means no coverage at all => from earliest coverage
            new_start_date = None
        else:
            # day after the last coverage date
            new_start_date = (max_mi + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Use get_signal to fetch coverage from repository (which might rebuild if refresh_data=True)
        df_signal = self.get_signal.run(
            signal_name,
            start_date=new_start_date,  # or None
            end_date=None,
            refresh_data=refresh_data
        )
        if df_signal.empty:
            print(f"No signal coverage for '{signal_name}' after date={new_start_date}. Stopping.")
            return

        # If we do an incremental approach, we might pull returns from new_start_date - 1 year
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
        df_merged = df_signal.join(df_returns, how="inner").dropna()

        # columns => [ <signal_name>, t1_returns ]
        # For daily MI, we might want a daily 'rank' or not? 
        # The snippet used 't1_returns_rank' or 't1_returns'? We'll assume raw returns here
        # If you want rank, you can do:
        # df_merged['t1_returns_rank'] = df_merged.groupby('date')['t1_returns'].rank(pct=True)

        # 3) Build the MI calculator
        signal_col = signal_name
        returns_col = "t1_returns"  # or "t1_returns_rank" if you prefer
        micalc = MutualInformationCalculator(
            df_merged,
            x_col=signal_col,
            y_col=returns_col,
            window_size=self.window_size
        )

        # Compute daily MI
        daily_mi_df = micalc.compute_daily_mi()  # columns=[mi], index=date
        daily_mi_df["signal"] = signal_name

        # Compute entire-window rolling MI
        rolling_mi_df = micalc.compute_rolling_mi_entire_window()  # columns=[mi_42], index=date
        rolling_mi_df["signal"] = signal_name

        # 4) Filter out rows prior to new_start_date
        if new_start_date:
            dt_ns = pd.to_datetime(new_start_date)
            daily_mi_df = daily_mi_df.loc[daily_mi_df.index >= dt_ns]
            rolling_mi_df = rolling_mi_df.loc[rolling_mi_df.index >= dt_ns]

        if daily_mi_df.empty and rolling_mi_df.empty:
            print(f"No new MI rows to upload for '{signal_name}'.")
            return

        # 5) Append daily results to 'mi_daily'
        if not daily_mi_df.empty:
            self._append_to_table(daily_mi_df, self.table_daily_mi, daily=True)
        # 6) Append rolling results to 'mi_rolling'
        if not rolling_mi_df.empty:
            self._append_to_table(rolling_mi_df, self.table_rolling_mi, daily=False)

        print(f"Incremental MI update complete for signal '{signal_name}'.")

    ###########################################################################
    # HELPERS
    ###########################################################################
    def _get_minmax_date_in_table(self, table_name: str, signal_name: str):
        """
        Return (min_date, max_date) from table_name for the given signal_name.
        Example for mi_daily or mi_rolling.
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
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    def _fetch_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Similar to your IncrementalICCalculator's approach.
        Pull t1_returns from wmg.returns_table in [start_date..end_date].
        Return DataFrame => columns=[date, requestId, t1_returns].
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        params = []
        if start_date:
            where_clauses.append("date >= @start_date")
            params.append(bigquery.ScalarQueryParameter("start_date", "DATETIME", start_date))
        if end_date:
            where_clauses.append("date <= @end_date")
            params.append(bigquery.ScalarQueryParameter("end_date", "DATETIME", end_date))

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
        df_mi.reset_index(inplace=True)  # now columns => [date, mi( or mi_42 ), signal]
        # define final columns
        if daily:
            # daily => 'mi'
            if "mi" not in df_mi.columns:
                df_mi["mi"] = np.nan
            final_cols = ["date", "mi", "signal"]
        else:
            # rolling => e.g. 'mi_42'
            rolling_col = [c for c in df_mi.columns if c.startswith("mi_")]
            if not rolling_col:
                rolling_col = ["mi_rolling"]
                df_mi["mi_rolling"] = np.nan
            else:
                rolling_col = rolling_col[0]
            final_cols = ["date", rolling_col, "signal"]

        # ensure any missing columns are present
        for c in final_cols:
            if c not in df_mi.columns:
                df_mi[c] = np.nan

        df_mi = df_mi[final_cols]

        # Now upload
        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"
        to_gbq(df_mi, full_table, project_id=self.project_id, if_exists="append")
        print(f"Appended {len(df_mi)} rows into {full_table}.")   
        
        

class IncrementalFvalue:
    """
    A class that calculates daily F-values for a single signal,
    plus rolling transformations, and appends them incrementally
    to 'fvalue_daily' and 'fvalue_rolling' (or whichever you specify).
    """

    def __init__(
        self,
        db,             # Database instance
        get_signal,     # GetSignal instance
        project_id="issachar-feature-library",
        wmg_dataset="wmg",
        returns_table="t1_returns",
        table_daily_fval="fvalue_daily",
        table_rolling_fval="fvalue_rolling"
    ):
        """
        Args:
            db: Database instance
            get_signal: GetSignal instance
            project_id: GCP Project ID
            wmg_dataset: BQ dataset name
            returns_table: name of table with t1_returns
            table_daily_fval: table for daily F-values
            table_rolling_fval: table for rolling transformations
        """
        self.db = db
        self.get_signal = get_signal
        self.project_id = project_id
        self.wmg_dataset = wmg_dataset
        self.returns_table = returns_table
        self.table_daily_fval = table_daily_fval
        self.table_rolling_fval = table_rolling_fval

        self.bq_client = bigquery.Client(project=self.project_id)

    def run(self, signal_name: str, refresh_data: bool = True):
        """
        1) Check coverage in table_daily_fval for 'signal_name' => (min, max)
        2) If new coverage is needed, fetch from get_signal
        3) Merge with returns, compute daily F-values
        4) Filter out old coverage
        5) Append new daily rows
        6) Compute rolling transformations => append new rolling rows
        """
        # 1) Check coverage in daily table
        min_cov, max_cov = self._get_minmax_date_in_table(self.table_daily_fval, signal_name)
        if max_cov is None:
            new_start_date = None
        else:
            new_start_date = (max_cov + timedelta(days=1)).strftime("%Y-%m-%d")

        # 2) Pull coverage from repository
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
        # We'll fetch returns from new_start_date - 1y if partial coverage
        if new_start_date:
            start_for_returns_dt = pd.to_datetime(new_start_date) - timedelta(days=365)
            start_for_returns = start_for_returns_dt.strftime("%Y-%m-%d")
        else:
            start_for_returns = None

        df_returns = self._fetch_returns_in_date_range(start_for_returns, None)
        if df_returns.empty:
            print("No returns found. Stopping.")
            return

        df_signal.set_index(["date","requestId"], inplace=True)
        df_returns.set_index(["date","requestId"], inplace=True)
        df_merged = df_signal.join(df_returns, how="inner").dropna()

        # 4) Build an FvalueCalculator
        from your_module import FvalueCalculator  # or place it inline
        fcalc = FvalueCalculator(
            df_merged,
            signal_col=signal_name,
            returns_col="t1_returns"  # or 't1_returns_rank' if you want rank
        )

        # compute daily
        daily_f_df = fcalc.compute_daily_fvalues()  # index=date, col=[fvalue]
        daily_f_df["signal"] = signal_name

        # remove coverage < new_start_date
        if new_start_date:
            dt_ns = pd.to_datetime(new_start_date)
            daily_f_df = daily_f_df.loc[daily_f_df.index >= dt_ns]
        if daily_f_df.empty:
            print("No new daily F-values. Stopping.")
            return

        # 5) append daily
        self._append_to_table(daily_f_df, self.table_daily_fval, daily=True)

        # 6) rolling transformations
        # e.g. the snippet: rolling median / subtractions
        rolling_df = fcalc.compute_rolling_interactions(daily_f_df)
        # rolling_df => includes daily_fvalue, rolling medians, etc.
        # filter out coverage < new_start_date
        if new_start_date:
            rolling_df = rolling_df.loc[rolling_df.index >= dt_ns]
        if rolling_df.empty:
            print("No new rolling transformations. Done.")
            return

        rolling_df["signal"] = signal_name
        self._append_to_table(rolling_df, self.table_rolling_fval, daily=False)
        print(f"Incremental Fvalue update complete for '{signal_name}'.")

    ###########################################################################
    # HELPERS
    ###########################################################################
    def _get_minmax_date_in_table(self, table_name: str, signal_name: str):
        """
        Return (min_date, max_date) from table_name for the given signal_name.
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
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
        )
        df = self.bq_client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    def _fetch_returns_in_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Similar to your other incremental classes: fetch t1_returns from wmg.returns_table
        in [start_date..end_date].
        Return DataFrame => [date, requestId, t1_returns].
        """
        full_table = f"{self.project_id}.{self.wmg_dataset}.{self.returns_table}"
        where_clauses = []
        params = []
        if start_date:
            where_clauses.append("date >= @start_date")
            params.append(bigquery.ScalarQueryParameter("start_date", "DATETIME", start_date))
        if end_date:
            where_clauses.append("date <= @end_date")
            params.append(bigquery.ScalarQueryParameter("end_date", "DATETIME", end_date))

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

    def _append_to_table(self, df_fval: pd.DataFrame, table_name: str, daily=True):
        """
        Append new rows to the specified table. For daily => columns [fvalue, signal].
        For rolling => might have [daily_fvalue, fvalue_median_21, etc., signal].
        We'll flatten to a consistent format if needed. Or we can store wide.
        """
        df_upload = df_fval.reset_index().copy()  # ensure 'date' is a column
        full_table = f"{self.project_id}.{self.wmg_dataset}.{table_name}"

        # decide which columns to keep
        # For daily, we at least expect [date, fvalue, signal].
        # For rolling, we might have many columns. Let's just store them all for now.
        if daily:
            # ensure columns
            if "fvalue" not in df_upload.columns:
                df_upload["fvalue"] = np.nan
            if "signal" not in df_upload.columns:
                df_upload["signal"] = "unknown"
            keep_cols = ["date", "fvalue", "signal"]
            df_upload = df_upload[keep_cols]
        else:
            # rolling => we'll store everything. But we must have 'date' and 'signal'
            if "signal" not in df_upload.columns:
                df_upload["signal"] = "unknown"
            # Just append all columns. If you want to limit them, define keep_cols.

        to_gbq(
            df_upload,
            destination_table=f"{self.wmg_dataset}.{table_name}",
            project_id=self.project_id,
            if_exists="append"
        )
        print(f"Appended {len(df_upload)} rows into {full_table}.")
        
        
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

        
        
        
if __name__ == '__main__':
    
    # Pull all signals from GBQ
    all_signals = get_unique_signals_from_gbq()
    
    # Loop through each, and refresh any metrics, need be
    for signal_name in all_signals:
    
        # 1) Create Database
        db = Database(project_id="issachar-feature-library")

        # 2) Create GetSignal
        get_signal = GetSignal(db=db)

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
        mi_calc.run("accel_21d", refresh_data = False)

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

        fval_calc.run("some_signal_name", refresh_data=False)