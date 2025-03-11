import pandas as pd
from typing import Union, List, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

class Database:
    """
    Class to manage signals in BigQuery.

    Primary responsibilities:
      1) Manage the 'signal_lookup' table (wmg.signal_lookup),
         which has columns:
           signal_name, signal_type, interaction_type,
           left_baseline, right_baseline,
           left_location, right_location, feature_set_location
      2) Provide reading/writing to a 'signal_repository' table (wmg.signal_repository)
         that stores actual signal values in the format [date, requestId, signal, value].
      3) Provide reading/writing to a 'daily_ic2' table (wmg.daily_ic2) for metrics
         coverage (e.g. all_IC, topq_IC, etc.).
      4) Provide methods to read baseline signals from core_raw.

    NOTE: You must adjust the exact queries to match your real dataset/table/column naming.
    """

    # 1) The table where we store signal definitions
    LOOKUP_TABLE_ID = "issachar-feature-library.wmg.signal_lookup"

    # 2) The repository table: store actual signal data
    REPOSITORY_TABLE = "issachar-feature-library.wmg.signal_repository"

    # 3) daily_ic2 table: store IC metrics
    DAILY_IC2_TABLE = "issachar-feature-library.wmg.daily_ic2"

    def __init__(self, project_id="issachar-feature-library"):
        self.project_id = project_id
        self.client = bigquery.Client(project=self.project_id)
        self._create_lookup_table_if_not_exists()

    ############################################################################
    #  LOOKUP TABLE (signal_lookup) METHODS
    ############################################################################

    def _create_lookup_table_if_not_exists(self):
        """
        Internal helper that checks for existence of the table:
          'issachar-feature-library.wmg.signal_lookup'.
        If it doesn't exist, create it with the expected schema (8 columns).
        """
        table_ref = bigquery.TableReference.from_string(self.LOOKUP_TABLE_ID)
        try:
            self.client.get_table(table_ref)
        except NotFound:
            schema = [
                bigquery.SchemaField("signal_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("signal_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("interaction_type", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("left_baseline", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("right_baseline", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("left_location", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("right_location", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("feature_set_location", "STRING", mode="NULLABLE"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            print(f"Created table {self.LOOKUP_TABLE_ID} with 8 columns.")

    def get_signal(self, signal_name: str) -> Optional[dict]:
        """
        Return a dictionary describing the signal from signal_lookup, or None if not found.
        """
        query = f"""
            SELECT signal_name, signal_type, interaction_type,
                   left_baseline, right_baseline,
                   left_location, right_location, feature_set_location
            FROM `{self.LOOKUP_TABLE_ID}`
            WHERE signal_name = @signal_name
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[ bigquery.ScalarQueryParameter("signal_name", "STRING", signal_name) ]
        )
        df = self.client.query(query, job_config=job_config).to_dataframe()
        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        return {
            "signal_name":          row["signal_name"],
            "signal_type":          row["signal_type"],
            "interaction_type":     row["interaction_type"],
            "left_baseline":        row["left_baseline"],
            "right_baseline":       row["right_baseline"],
            "left_location":        row["left_location"],
            "right_location":       row["right_location"],
            "feature_set_location": row["feature_set_location"],
        }

    def upload_signal(self, signal_data: dict):
        """
        Insert a single row into wmg.signal_lookup. The dict must have the 8 keys:
            - signal_name, signal_type, interaction_type,
            - left_baseline, right_baseline,
            - left_location, right_location,
            - feature_set_location
        """
        rows_to_insert = [(
            signal_data["signal_name"],
            signal_data["signal_type"],
            signal_data["interaction_type"],
            signal_data["left_baseline"],
            signal_data["right_baseline"],
            signal_data["left_location"],
            signal_data["right_location"],
            signal_data["feature_set_location"],
        )]
        table_ref = bigquery.TableReference.from_string(self.LOOKUP_TABLE_ID)
        errors = self.client.insert_rows(self.client.get_table(table_ref), rows_to_insert)
        if errors:
            raise RuntimeError(
                f"Failed to insert row for signal {signal_data['signal_name']}. Errors: {errors}"
            )

    def update_signal(self, signal_data: dict):
        """
        Updates an existing row in signal_lookup by deleting the old and inserting a new one.
        """
        existing = self.get_signal(signal_data["signal_name"])
        if existing is None:
            raise ValueError(f"Signal '{signal_data['signal_name']}' not found; cannot update.")

        # delete old row
        delete_query = f"""
            DELETE FROM `{self.LOOKUP_TABLE_ID}`
            WHERE signal_name = @signal_name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[ bigquery.ScalarQueryParameter("signal_name", "STRING", signal_data["signal_name"]) ]
        )
        self.client.query(delete_query, job_config=job_config).result()

        # insert new row
        self.upload_signal(signal_data)

    def download_all_signals(self) -> list:
        """
        Return all rows from wmg.signal_lookup as a list of dicts.
        """
        query = f"""
            SELECT signal_name, signal_type, interaction_type,
                   left_baseline, right_baseline,
                   left_location, right_location, feature_set_location
            FROM `{self.LOOKUP_TABLE_ID}`
        """
        df = self.client.query(query).to_dataframe()
        results = []
        for _, row in df.iterrows():
            results.append({
                "signal_name":          row["signal_name"],
                "signal_type":          row["signal_type"],
                "interaction_type":     row["interaction_type"],
                "left_baseline":        row["left_baseline"],
                "right_baseline":       row["right_baseline"],
                "left_location":        row["left_location"],
                "right_location":       row["right_location"],
                "feature_set_location": row["feature_set_location"],
            })
        return results

    def get_or_upload_signal(self, signal_data: dict) -> dict:
        """
        If a signal with signal_data['signal_name'] is in the table, return it.
        Otherwise, insert the new row and return it.
        """
        existing = self.get_signal(signal_data["signal_name"])
        if existing is not None:
            return existing
        self.upload_signal(signal_data)
        return self.get_signal(signal_data["signal_name"])

    ############################################################################
    #  SIGNAL REPOSITORY METHODS (wmg.signal_repository)
    ############################################################################

    def get_signal_data_from_repository(
        self,
        signal_name: Union[str, List[str]],
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Query wmg.signal_repository for one or more signals, optionally filtering by date range.
        Returns [date, requestId, signal, value].
        """
        repository_table = self.REPOSITORY_TABLE
        where_clauses = ["date IS NOT NULL"]
        query_params = []

        if isinstance(signal_name, str):
            where_clauses.append("signal = @signal_name")
            query_params.append(
                bigquery.ScalarQueryParameter("signal_name", "STRING", signal_name)
            )
        else:
            # list of signals
            where_clauses.append("signal IN UNNEST(@signal_list)")
            query_params.append(
                bigquery.ArrayQueryParameter("signal_list", "STRING", signal_name)
            )

        if start_date:
            start_dt = pd.to_datetime(start_date)
            where_clauses.append("date >= @start_date_ts")
            query_params.append(bigquery.ScalarQueryParameter("start_date_ts", "TIMESTAMP", start_dt))

        if end_date:
            end_dt = pd.to_datetime(end_date)
            where_clauses.append("date <= @end_date_ts")
            query_params.append(bigquery.ScalarQueryParameter("end_date_ts", "TIMESTAMP", end_dt))

        where_clause = " AND ".join(where_clauses)
        query = f"""
            SELECT date, requestId, signal, value
            FROM `{repository_table}`
            WHERE {where_clause}
        """

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        df = self.client.query(query, job_config=job_config).to_dataframe()
        return df

    def get_minmax_dates_from_repository(self, signal_name: str) -> (Optional[pd.Timestamp], Optional[pd.Timestamp]):
        """
        Returns (min_date, max_date) from wmg.signal_repository for the given signal,
        or (None, None) if no coverage.
        """
        query = f"""
        SELECT
          CAST(MIN(date) AS TIMESTAMP) as min_date,
          CAST(MAX(date) AS TIMESTAMP) as max_date
        FROM `{self.REPOSITORY_TABLE}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
        )
        df = self.client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    def delete_signal_from_repository(self, signal_name: str):
        """
        Removes all rows in wmg.signal_repository for the specified signal.
        """
        query = f"""
        DELETE FROM `{self.REPOSITORY_TABLE}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
        )
        self.client.query(query, job_config=job_config).result()

    def upload_signal_to_repository(self, df: pd.DataFrame, signal_name: str):
        """
        Appends rows to wmg.signal_repository.
        `df` must have columns: [date, requestId, <signal_name>].
        We'll convert it to [date, requestId, signal, value].
        """
        # Convert to the shape we need
        df = df.copy()
        if signal_name not in df.columns:
            raise ValueError(f"DataFrame missing column '{signal_name}' for upload.")

        # rename <signal_name> => 'value'
        df.rename(columns={signal_name: "value"}, inplace=True)
        df["signal"] = signal_name

        # Ensure date is a suitable type
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # keep only necessary columns
        df_upload = df[["date", "requestId", "signal", "value"]].copy()

        # Insert using BQ's load_table_from_dataframe or insert_rows
        table_id = self.REPOSITORY_TABLE
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = self.client.load_table_from_dataframe(df_upload, table_id, job_config=job_config)
        job.result()  # wait
        print(f"Uploaded {len(df_upload)} rows to {table_id} for signal='{signal_name}'.")

    ############################################################################
    #  CORE_RAW DATA METHODS
    ############################################################################

    def get_minmax_dates_in_core_raw(self, table_name: str, column_name: str) -> (Optional[pd.Timestamp], Optional[pd.Timestamp]):
        """
        Returns (min_date, max_date) from `core_raw.{table_name}` 
        for the given column_name, ignoring NULLs. 
        Table_name is e.g. 'fundamentals_full', 'fql_ptx_rev', etc.
        """
        full_table = f"{self.project_id}.core_raw.{table_name}"
        query = f"""
        SELECT 
          CAST(MIN(date) AS TIMESTAMP) as min_date,
          CAST(MAX(date) AS TIMESTAMP) as max_date
        FROM `{full_table}`
        WHERE `{column_name}` IS NOT NULL
        """
        df = self.client.query(query).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    def pull_data_from_core_raw(self, table_name: str, column_name: str) -> pd.DataFrame:
        """
        Select [date, requestId, column_name] from core_raw.{table_name}
        where {column_name} is NOT NULL. Return a DataFrame with columns:
            date, requestId, {column_name}
        """
        full_table = f"{self.project_id}.core_raw.{table_name}"
        query = f"""
        SELECT date, requestId, `{column_name}` as col_val
        FROM `{full_table}`
        WHERE `{column_name}` IS NOT NULL
        """
        df = self.client.query(query).to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["date", "requestId", column_name])

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.drop_duplicates(["date", "requestId"], inplace=True)
        df.rename(columns={"col_val": column_name}, inplace=True)
        return df[["date", "requestId", column_name]]

    ############################################################################
    #  daily_ic2 METHODS (for metrics coverage)
    ############################################################################

    def get_minmax_dates_from_daily_ic2(self, signal_name: str) -> (Optional[pd.Timestamp], Optional[pd.Timestamp]):
        """
        Returns (min_date, max_date) from daily_ic2 for the given signal.
        If not present, returns (None, None).
        """
        full_table = self.DAILY_IC2_TABLE
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
        df = self.client.query(query, job_config=job_config).to_dataframe()
        if df.empty or df.iloc[0].isnull().all():
            return (None, None)
        row = df.iloc[0]
        return (row["min_date"], row["max_date"])

    def get_data_from_daily_ic2(self, signal_name: str) -> pd.DataFrame:
        """
        Pull all rows for the given signal from daily_ic2.
        Returns a DataFrame with columns matching daily_ic2 schema:
           date (DATETIME), all_IC, all_tstat, ..., signal (STRING).
        """
        query = f"""
        SELECT date, all_IC, all_tstat, all_pvalue, all_fstat,
               all_IC_spearmanr, all_pvalue_spearmanr,
               topq_IC, topq_tstat, topq_pvalue, topq_fstat,
               topq_IC_spearmanr, topq_pvalue_spearmanr,
               signal
        FROM `{self.DAILY_IC2_TABLE}`
        WHERE signal = @signal
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[ bigquery.ScalarQueryParameter("signal", "STRING", signal_name) ]
        )
        df = self.client.query(query, job_config=job_config).to_dataframe()
        return df

    def append_to_daily_ic2(self, df: pd.DataFrame):
        """
        Append rows to daily_ic2. The df must have columns like:
            [date, all_IC, all_tstat, ..., signal].
        """
        full_table = self.DAILY_IC2_TABLE
        # We can do a load job
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = self.client.load_table_from_dataframe(df, full_table, job_config=job_config)
        job.result()
        print(f"Appended {len(df)} rows into {full_table}.")
