import pandas as pd
import numpy as np
from typing import Union, List

class GetSignal:
    """
    Class to fetch signal data from 'signal_repository'.
    Optionally, if 'refresh_data=True', it checks baseline coverage
    and rebuilds coverage if more data is available in left/right
    tables (for an interaction) or the single baseline table.
    """

    def __init__(self, db, verbose = True):
        """
        db: a Database instance
        """
        self.db = db
        self.verbose = verbose

    def run(
        self,
        signal_name: Union[str, List[str]],
        start_date: str = None,
        end_date: str = None,
        refresh_data: bool = False
    ) -> pd.DataFrame:
        """
        If a single string:
          - If refresh_data=True, do the coverage check & rebuild logic
            (pull from left/right baseline if there's new data).
          - If refresh_data=False, just pull from repository, ignoring
            whether baseline tables might have more coverage.
          - Finally, apply optional [start_date, end_date] filtering.

        If a list of strings, do the old multi-signal approach (no rebuild).
        """

        if isinstance(signal_name, list):
            return self._handle_multiple_signals(signal_name, start_date, end_date)

        # Single signal flow
        meta = self.db.get_signal(signal_name)
        if meta is None:
            print(f"Signal '{signal_name}' not found in lookup table.")
            return pd.DataFrame()

        if not refresh_data:
            # SKIP coverage rebuild, just pull from repository
            return self._fetch_from_repository(signal_name, start_date, end_date)

        # Otherwise, do the coverage check & potential rebuild
        return self._rebuild_if_needed_and_fetch(signal_name, start_date, end_date)

    ###########################################################################
    # Multi-signal approach: old logic (no rebuild coverage)
    ###########################################################################
    def _handle_multiple_signals(self, signals_list, start_date, end_date) -> pd.DataFrame:
        final_signals = []
        for s in signals_list:
            meta = self.db.get_signal(s)
            if meta is not None:
                final_signals.append(s)
            else:
                print(f"Signal '{s}' not found in lookup table. Skipping...")

        if not final_signals:
            print("No valid signals found in lookup table.")
            return pd.DataFrame()

        df = self.db.get_signal_data_from_repository(final_signals, start_date, end_date)
        if df.empty:
            print(f"No data found for signals {final_signals} after date filters.")
            return df

        # Convert date to timezone-naive
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # pivot wide
        df_pivot = df.pivot_table(
            index=["date", "requestId"], columns="signal", values="value"
        ).reset_index()

        col_order = ["date", "requestId"] + sorted(final_signals)
        df_pivot = df_pivot.reindex(columns=col_order, fill_value=None)
        return df_pivot

    ###########################################################################
    # If refresh_data=True, rebuild coverage if needed
    ###########################################################################
    def _rebuild_if_needed_and_fetch(
        self, signal_name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Check coverage in repository vs. baseline. If baseline is bigger, rebuild.
        Then fetch final coverage from repository. Finally, apply optional 
        [start_date, end_date] filter if provided.
        """
        meta = self.db.get_signal(signal_name)
        
        # Because BaselineSignals now always sets these fields for a baseline:
        #   meta["left_baseline"] = <the baseline signal name>
        #   meta["left_location"] = <the first table in core_raw that has it>
        #   meta["right_baseline"] = None
        #   meta["right_location"] = None
        #
        # For interaction:
        #   meta["left_baseline"], meta["right_baseline"] etc.
        
        left_table = meta.get("left_location")
        right_table = meta.get("right_location")

        # 1) Check coverage in repository
        repo_min, repo_max = self.db.get_minmax_dates_from_repository(signal_name)

        # 2) Check coverage in baseline (left + right).
        #    For baseline signals, we only have left_table.
        #    For interaction signals, we have both.
        left_min, left_max = (None, None)
        if left_table:
            # e.g. left_table = "core_raw.fql_adl"
            # => table_name_left = "fql_adl"
            table_name_left = left_table.split(".")[1]
            col_left = meta["left_baseline"]  # <-- use the newly guaranteed field
            left_min, left_max = self.db.get_minmax_dates_in_core_raw(table_name_left, col_left)

        right_min, right_max = (None, None)
        if meta["signal_type"]=="interaction" and right_table:
            table_name_right = right_table.split(".")[1]
            col_right = meta["right_baseline"]
            right_min, right_max = self.db.get_minmax_dates_in_core_raw(table_name_right, col_right)

        new_cov_min, new_cov_max = self._union_coverage(left_min, left_max, right_min, right_max)
        if new_cov_min is None or new_cov_max is None:
            # no coverage in baseline
            if self.verbose:
                print(f"No coverage found in baseline for '{signal_name}'. Returning repository data.")
            return self._fetch_from_repository(signal_name, start_date, end_date)

        # If baseline coverage is not bigger than repository coverage, skip
        if repo_max and new_cov_max <= repo_max:
            if self.verbose:
                print(f"No new coverage to rebuild for '{signal_name}'. Using repository data only.")
            return self._fetch_from_repository(signal_name, start_date, end_date)

        # Rebuild coverage from left/right
        if self.verbose:
            print(f"Rebuilding coverage for '{signal_name}' from baseline tables...")

        # 3) Pull from left and (if needed) right
        df_left = pd.DataFrame()
        if left_table:
            table_name_left = left_table.split(".")[1]
            col_left = meta["left_baseline"]
            df_left = self.db.pull_data_from_core_raw(table_name_left, col_left)

        df_right = pd.DataFrame()
        if meta["signal_type"]=="interaction" and right_table:
            table_name_right = right_table.split(".")[1]
            col_right = meta["right_baseline"]
            df_right = self.db.pull_data_from_core_raw(table_name_right, col_right)

        # 4) build final coverage
        df_signal = self._build_signal(meta, df_left, df_right)

        # 5) delete old coverage, upload new
        self.db.delete_signal_from_repository(signal_name)
        self.db.upload_signal_to_repository(df_signal, signal_name)

        # 6) fetch the new coverage from repository, applying date filter
        return self._fetch_from_repository(signal_name, start_date, end_date)

    def _union_coverage(self, lmin, lmax, rmin, rmax):
        """
        Union coverage for left + right. 
        """
        if lmin is None and rmin is None:
            return (None, None)

        mins = []
        if lmin: mins.append(lmin)
        if rmin: mins.append(rmin)

        maxs = []
        if lmax: maxs.append(lmax)
        if rmax: maxs.append(rmax)

        if not mins or not maxs:
            return (None, None)

        return (min(mins), max(maxs))

    def _build_signal(self, meta: dict, df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
        """
        If baseline => rename df_left -> final. 
        If interaction => merge + apply interaction function.
        """
        signal_name = meta["signal_name"]
        sig_type = meta["signal_type"]

        # For baseline, we use meta["left_baseline"] for the column rename
        # For interaction, we do the same for left_col / right_col
        if sig_type != "interaction":
            # baseline
            df_left = df_left.copy()
            baseline_col = meta["left_baseline"]  # same as signal_name
            if baseline_col in df_left.columns:
                df_left.rename(columns={baseline_col: signal_name}, inplace=True)
            # final => [date, requestId, signal_name]
            return df_left[["date", "requestId", signal_name]]

        # interaction
        left_col_name = meta["left_baseline"]
        right_col_name = meta["right_baseline"]
        interact_type = meta.get("interaction_type", "interaction_sum")

        df_left = df_left.copy()
        df_right = df_right.copy()
        df_left.rename(columns={left_col_name: "left_col"}, inplace=True)
        df_right.rename(columns={right_col_name: "right_col"}, inplace=True)
        merged = pd.merge(
            df_left[["date", "requestId", "left_col"]],
            df_right[["date", "requestId", "right_col"]],
            on=["date", "requestId"], how="inner"
        )
        merged[signal_name] = self._apply_interaction(interact_type, merged["left_col"], merged["right_col"])
        return merged[["date", "requestId", signal_name]]

    def _apply_interaction(self, itype: str, s1: pd.Series, s2: pd.Series) -> pd.Series:
        if itype == "interaction_product":
            return s1 * s2
        elif itype == "interaction_sum":
            return s1 + s2
        elif itype == "interaction_ratio":
            return s1 / (s2 + 1e-5)
        elif itype == "interaction_squared_sum":
            return s1**2 + s2**2
        elif itype == "interaction_diff_squared":
            return (s1 - s2)**2
        elif itype == "interaction_harmonic_mean":
            return 2 / (1/(s1+1e-5) + 1/(s2+1e-5))
        elif itype == "interaction_geometric_mean":
            return (s1 * s2) ** 0.5
        elif itype == "interaction_high_order":
            return s1 * (s2**2)
        else:
            print(f"Unknown interaction type '{itype}'. Returning NaN.")
            return np.nan

    def _fetch_from_repository(self, signal_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.db.get_signal_data_from_repository(signal_name, start_date, end_date)
        if df.empty:
            print(f"No data found in repository for '{signal_name}'.")
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.rename(columns={"value": signal_name}, inplace=True)
        df.drop(columns=["signal"], inplace=True)
        return df[["date", "requestId", signal_name]]
