import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import uuid

class BaselineSignals:
    """
    A class that fetches all known baseline signals from the 'core_raw' dataset,
    and can parse an input signal to determine if it is:
      - baseline,
      - interaction (and if so, which two baseline signals + which interaction prefix),
      - or neither.

    It also returns additional metadata on where each piece is found:
      - left_location, right_location (core_raw tables)
      - feature_set_location (the wmg.feature_set* table containing the entire signal)

    Optional features:
      - copy_to_new_table: if True, automatically copies data for the signal into
        'signal_repository' after parsing.
      - deduplicate: if True, prevents duplicate rows in 'signal_repository' via MERGE.

    The 'signal_repository' table schema is (date TIMESTAMP, requestId STRING, signal STRING, value FLOAT).
    """

    KNOWN_INTERACTION_PREFIXES = {
        'interaction_product',
        'interaction_sum',
        'interaction_ratio',
        'interaction_squared_sum',
        'interaction_diff_squared',
        'interaction_harmonic_mean',
        'interaction_geometric_mean',
        'interaction_high_order',
    }

    def __init__(
        self,
        project_id: str = "issachar-feature-library",
        core_raw_dataset: str = "core_raw",
        wmg_dataset: str = "wmg",
        copy_to_new_table: bool = False,
        deduplicate: bool = False
    ):
        """
        Args:
            project_id (str): GCP project ID.
            core_raw_dataset (str): The dataset name for 'core_raw'.
            wmg_dataset (str): The dataset name for the 'feature_set...' tables.
            copy_to_new_table (bool): If True, after run() we also copy the data to
                                      'signal_repository' (if found).
            deduplicate (bool): If True, we'll use a MERGE-based approach to avoid
                                inserting duplicate rows (date, requestId, signal).
        """
        self.project_id = project_id
        self.core_raw_dataset = core_raw_dataset
        self.wmg_dataset = wmg_dataset
        self.copy_to_new_table = copy_to_new_table
        self.deduplicate = deduplicate

        # 1) Gather all baseline signals from core_raw
        self.baseline_signals = self._gather_all_baseline_signals()

        # 2) Build dict: baseline_column -> first core_raw table that has it
        self.core_raw_locations = self._map_columns_to_tables_first_only(
            project_id=self.project_id,
            dataset_name=self.core_raw_dataset
        )

        # 3) Build dict: any signal found in wmg.feature_set* -> first table
        self.feature_set_locations = self._map_feature_set_columns_first_only(
            project_id=self.project_id,
            dataset_name=self.wmg_dataset,
            table_prefix="feature_set"
        )

        # 4) If copy_to_new_table => ensure 'signal_repository' table
        if self.copy_to_new_table:
            self._create_signal_repository_if_not_exists()

    def run(self, signal: str) -> dict:
        """
        Takes in a single 'signal' string.

        Returns a dict with:
          - 'signal_name': str
          - 'signal_type': "baseline", "interaction", or "neither"
          - 'interaction_type': e.g. 'interaction_sum' or None
          - 'left_baseline': str or None
          - 'right_baseline': str or None
          - 'left_location': str or None
          - 'right_location': str or None
          - 'feature_set_location': str or None
        """
        result = {
            'signal_name': signal,
            'signal_type': 'neither',
            'interaction_type': None,
            'left_baseline': None,
            'right_baseline': None,
            'left_location': None,
            'right_location': None,
            'feature_set_location': None,
        }

        # CASE 1: If the signal is a direct baseline signal
        if signal in self.baseline_signals:
            result['signal_type'] = 'baseline'
            # 1a) fill in left_baseline, left_location
            result['left_baseline'] = signal
            result['left_location'] = self.core_raw_locations.get(signal, None)

            # 1b) feature_set_location if found
            fs_table = self.feature_set_locations.get(signal, None)
            result['feature_set_location'] = fs_table

            # 1c) copy to repository if needed
            if self.copy_to_new_table and fs_table:
                self._copy_signal_data_to_repository(signal, fs_table)
            return result

        # CASE 2: Check if it starts with a known interaction prefix
        possible_prefix = None
        for prefix in self.KNOWN_INTERACTION_PREFIXES:
            if signal.startswith(prefix + "_"):
                possible_prefix = prefix
                break

        if not possible_prefix:
            # It's 'neither'
            fs_table = self.feature_set_locations.get(signal, None)
            result['feature_set_location'] = fs_table

            if self.copy_to_new_table and fs_table:
                self._copy_signal_data_to_repository(signal, fs_table)
            return result

        # If we get here, it's an 'interaction' type
        remainder = signal[len(possible_prefix) + 1:]
        for i in range(1, len(remainder)):
            left_part = remainder[:i].strip("_")
            right_part = remainder[i:].strip("_")

            if (left_part in self.baseline_signals) and (right_part in self.baseline_signals):
                result['signal_type'] = 'interaction'
                result['interaction_type'] = possible_prefix
                result['left_baseline'] = left_part
                result['right_baseline'] = right_part
                # fill left_location, right_location from core_raw
                result['left_location'] = self.core_raw_locations.get(left_part, None)
                result['right_location'] = self.core_raw_locations.get(right_part, None)

                fs_table = self.feature_set_locations.get(signal, None)
                result['feature_set_location'] = fs_table

                if self.copy_to_new_table and fs_table:
                    self._copy_signal_data_to_repository(signal, fs_table)
                return result

        # If no valid split => 'neither'
        fs_table = self.feature_set_locations.get(signal, None)
        result['feature_set_location'] = fs_table
        if self.copy_to_new_table and fs_table:
            self._copy_signal_data_to_repository(signal, fs_table)
        return result

    ############################################################################
    #      Private method: create 'signal_repository' if it does not exist
    ############################################################################

    def _create_signal_repository_if_not_exists(self):
        """
        Ensures there is a table named: <project_id>.<wmg_dataset>.signal_repository
        with schema: date (TIMESTAMP), requestId (STRING), signal (STRING), value (FLOAT).
        If it doesn't exist, create it. Otherwise do nothing.
        """
        self.repository_table_id = f"{self.project_id}.{self.wmg_dataset}.signal_repository"

        client = bigquery.Client(project=self.project_id)
        table_ref = bigquery.TableReference.from_string(self.repository_table_id)

        try:
            client.get_table(table_ref)
            # Table exists
        except NotFound:
            # Create table
            schema = [
                bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("requestId", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("signal", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("value", "FLOAT", mode="NULLABLE"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"Created table {self.repository_table_id}.")

    ############################################################################
    #  Private method: copy data for a single signal from feature_set table => repository
    ############################################################################

    def _copy_signal_data_to_repository(self, signal_name: str, full_table_id: str):
        """
        Pulls 'date', 'requestId', and the column <signal_name> from the table
        'full_table_id', then cleans up duplicates in memory (pandas),
        and loads/merges into <project_id>.<wmg_dataset>.signal_repository
        as (date, requestId, signal, value).
        """
        qualified_table_id = f"{self.project_id}.{full_table_id}"

        client = bigquery.Client(project=self.project_id)

        query = f"""
        SELECT
            date,
            requestId,
            `{signal_name}` AS col_value
        FROM `{qualified_table_id}`
        WHERE `{signal_name}` IS NOT NULL
        """
        df = client.query(query).to_dataframe()

        if df.empty:
            print(f"No rows found for signal '{signal_name}' in table '{full_table_id}'. Nothing to copy.")
            return

        # Convert date to timezone-naive
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Drop duplicates in-memory
        df.drop_duplicates(["date", "requestId"], inplace=True)

        # Insert the "signal" column
        df["signal"] = signal_name

        # Rename col_value -> value
        df.rename(columns={"col_value": "value"}, inplace=True)

        # Keep only [date, requestId, signal, value]
        df = df[["date", "requestId", "signal", "value"]]

        # Upsert into signal_repository
        self._upsert_into_signal_repository(df)

    ############################################################################
    #      Private method: upsert / deduplicate into signal_repository
    ############################################################################

    def _upsert_into_signal_repository(self, df: pd.DataFrame):
        """
        If self.deduplicate == True, do a staging-table + MERGE approach so that
        rows with the same (date, requestId, signal) are not duplicated.

        Otherwise, just do an append load.
        """
        if not self.deduplicate:
            # Just do a simple append
            self._append_to_signal_repository(df)
        else:
            # MERGE approach
            self._merge_into_signal_repository(df)

    def _append_to_signal_repository(self, df: pd.DataFrame):
        """
        Simple append of all rows in df to the signal_repository table.
        """
        table_id = f"{self.project_id}.{self.wmg_dataset}.signal_repository"
        client = bigquery.Client(project=self.project_id)
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        print(f"Appended {len(df)} rows into {table_id}.")

    def _merge_into_signal_repository(self, df: pd.DataFrame):
        """
        1) Create a temporary staging table with random name.
        2) Load df to that table.
        3) MERGE from staging table into final table on matching (date, requestId, signal).
        4) Insert only if not matched.
        5) Drop staging table.
        """
        client = bigquery.Client(project=self.project_id)
        repo_table_id = f"{self.project_id}.{self.wmg_dataset}.signal_repository"

        # Step 1: Create staging table
        temp_table_name = f"_temp_{uuid.uuid4().hex[:8]}"
        temp_table_id = f"{self.project_id}.{self.wmg_dataset}.{temp_table_name}"

        schema = [
            bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("requestId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("signal", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("value", "FLOAT", mode="NULLABLE"),
        ]
        table_ref = bigquery.Table(temp_table_id, schema=schema)
        client.create_table(table_ref)
        print(f"Created staging table {temp_table_id} for deduplication merge.")

        # Step 2: Load df to staging table
        load_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        load_job = client.load_table_from_dataframe(df, temp_table_id, job_config=load_config)
        load_job.result()
        print(f"Loaded {len(df)} rows to staging table {temp_table_id}.")

        # Step 3: MERGE into final
        # We'll only insert rows that don't match on (date, requestId, signal).
        # If you want to update 'value' on match, you can do so with:
        # "WHEN MATCHED THEN UPDATE SET value = S.value"
        # But here, we do nothing if matched, to avoid duplicates.
        merge_query = f"""
        MERGE `{repo_table_id}` AS T
        USING `{temp_table_id}` AS S
        ON T.date = S.date
           AND T.requestId = S.requestId
           AND T.signal = S.signal
        WHEN NOT MATCHED THEN
          INSERT (date, requestId, signal, value)
          VALUES (S.date, S.requestId, S.signal, S.value)
        """
        merge_job = client.query(merge_query)
        merge_job.result()
        print(f"Merge complete. Rows not already present have been inserted into {repo_table_id}.")

        # Step 4: Drop staging table
        client.delete_table(temp_table_id)
        print(f"Dropped staging table {temp_table_id}.")

    ############################################################################
    #              Internal Helper: gather baseline signals from core_raw
    ############################################################################

    def _gather_all_baseline_signals(self) -> set:
        """
        Gather all columns from the core_raw dataset (across all tables).
        Those columns are considered 'baseline' signals.
        """
        client = bigquery.Client(project=self.project_id)
        dataset_ref = bigquery.DatasetReference(self.project_id, self.core_raw_dataset)
        all_baselines = set()

        tables = client.list_tables(dataset_ref)
        for table_item in tables:
            table_ref = dataset_ref.table(table_item.table_id)
            table_obj = client.get_table(table_ref)
            for field in table_obj.schema:
                all_baselines.add(field.name)

        return all_baselines

    ############################################################################
    #       Internal Helper: find the first table in which each column appears
    #                (for the 'core_raw' dataset, for baseline signals)
    ############################################################################

    def _map_columns_to_tables_first_only(self, project_id, dataset_name) -> dict:
        """
        Creates a dict: column_name -> first table_id in the specified dataset
        that has a field named column_name.
        """
        client = bigquery.Client(project=project_id)
        dataset_ref = bigquery.DatasetReference(project_id, dataset_name)
        tables = client.list_tables(dataset_ref)

        col_map = {}
        for table_item in tables:
            table_id = table_item.table_id
            table_ref = dataset_ref.table(table_id)
            table_obj = client.get_table(table_ref)

            for field in table_obj.schema:
                if field.name not in col_map:
                    col_map[field.name] = f"{dataset_name}.{table_id}"
        return col_map

    ############################################################################
    #  Internal Helper: find the first 'feature_set*' table in wmg that has each
    #                    signal (baseline or interaction) as a column
    ############################################################################

    def _map_feature_set_columns_first_only(self, project_id, dataset_name, table_prefix="feature_set") -> dict:
        """
        Returns a dict: column_name -> first table_id (within the 'wmg' dataset,
        whose name starts with table_prefix) in which column_name is found.
        """
        client = bigquery.Client(project=project_id)
        dataset_ref = bigquery.DatasetReference(project_id, dataset_name)
        tables = client.list_tables(dataset_ref)

        col_map = {}
        for table_item in tables:
            if not table_item.table_id.startswith(table_prefix):
                continue

            table_ref = dataset_ref.table(table_item.table_id)
            table_obj = client.get_table(table_ref)

            for field in table_obj.schema:
                if field.name not in col_map:
                    col_map[field.name] = f"{dataset_name}.{table_item.table_id}"
        return col_map

    ############################################################################
    #                   Optional public function to gather daily ic
    ############################################################################

    def get_unique_signals_from_daily_ic(
        self,
        project_id='issachar-feature-library',
        dataset_name='wmg',
        table_name='ic_daily'
    ):
        """
        Fetch all unique signal values from the 'signal' column in the daily_ic table.
        """
        client = bigquery.Client(project=project_id)
        query = f"""
            SELECT DISTINCT signal
            FROM `{project_id}.{dataset_name}.{table_name}`
        """
        df = client.query(query).to_dataframe()
        return df['signal'].unique().tolist()
