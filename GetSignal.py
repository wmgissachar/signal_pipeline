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

    def __init__(self, db):
        """
        db: a Database instance
        """
        self.db = db

    def run(
        self,
        signal_name: Union[str, List[str]],
        start_date: str = None,
        end_date: str = None,
        refresh_data: bool = False) -> pd.DataFrame:
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

        # Otherwise, do the new coverage logic (as before)
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
        Similar logic to before: check coverage in repository vs. baseline.
        If baseline is bigger, rebuild. Then fetch final coverage from repository.
        Finally, apply [start_date, end_date] filter if provided.
        """
        meta = self.db.get_signal(signal_name)
        left_table = meta.get("left_location")
        right_table = meta.get("right_location")

        # 1) Check coverage in repository
        repo_min, repo_max = self.db.get_minmax_dates_from_repository(signal_name)

        # 2) Check coverage in baseline (left + right). For baseline signals,
        #    only left_table is relevant. For interactions, union coverage.
        left_min, left_max = (None, None)
        if left_table:
            # e.g. 'core_raw.fql_ptx_rev' => table_name = fql_ptx_rev
            table_name_left = left_table.split(".")[1]
            col_name_left = meta["left_baseline"] if meta["signal_type"]=="interaction" else meta["signal_name"]
            left_min, left_max = self.db.get_minmax_dates_in_core_raw(table_name_left, col_name_left)

        right_min, right_max = (None, None)
        if meta["signal_type"]=="interaction" and right_table:
            table_name_right = right_table.split(".")[1]
            col_name_right = meta["right_baseline"]
            right_min, right_max = self.db.get_minmax_dates_in_core_raw(table_name_right, col_name_right)

        new_cov_min, new_cov_max = self._union_coverage(left_min, left_max, right_min, right_max)
        if new_cov_min is None or new_cov_max is None:
            # no coverage in baseline
            print(f"No coverage found in baseline for '{signal_name}'. Returning repository data.")
            return self._fetch_from_repository(signal_name, start_date, end_date)

        # If baseline coverage is not bigger than repository coverage, skip
        if repo_max and new_cov_max <= repo_max:
            print(f"No new coverage to rebuild for '{signal_name}'. Using repository data only.")
            return self._fetch_from_repository(signal_name, start_date, end_date)

        # Rebuild coverage from left/right
        print(f"Rebuilding coverage for '{signal_name}' from baseline tables...")

        df_left = pd.DataFrame()
        if left_table:
            table_name_left = left_table.split(".")[1]
            col_left = meta["left_baseline"] if meta["signal_type"]=="interaction" else signal_name
            df_left = self.db.pull_data_from_core_raw(table_name_left, col_left)

        df_right = pd.DataFrame()
        if meta["signal_type"]=="interaction" and right_table:
            table_name_right = right_table.split(".")[1]
            col_right = meta["right_baseline"]
            df_right = self.db.pull_data_from_core_raw(table_name_right, col_right)

        # build final coverage
        df_signal = self._build_signal(meta, df_left, df_right)

        # delete old coverage, upload new
        self.db.delete_signal_from_repository(signal_name)
        self.db.upload_signal_to_repository(df_signal, signal_name)

        # finally, fetch the new coverage from repository, applying date filter
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
        if sig_type != "interaction":
            # baseline
            df_left = df_left.copy()
            if meta["left_baseline"] in df_left.columns:
                df_left.rename(columns={meta["left_baseline"]: signal_name}, inplace=True)
            # final => [date, requestId, signal_name]
            return df_left[["date", "requestId", signal_name]]

        # interaction
        left_col = meta["left_baseline"]
        right_col = meta["right_baseline"]
        interact_type = meta.get("interaction_type", "interaction_sum")  # default ?

        df_left = df_left.copy()
        df_right = df_right.copy()
        df_left.rename(columns={left_col: "left_col"}, inplace=True)
        df_right.rename(columns={right_col: "right_col"}, inplace=True)
        merged = pd.merge(
            df_left[["date", "requestId", "left_col"]],
            df_right[["date", "requestId", "right_col"]],
            on=["date", "requestId"], how="inner"
        )
        # apply
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



# import pandas as pd
# import numpy as np
# from typing import Union, List

# class GetSignal:
#     """
#     Class to fetch/update signal data from the repository. 
#     If the repository is missing some dates (compared to the baseline tables in core_raw), 
#     we rebuild the coverage from the original tables. 
#     """

#     def __init__(self, db):
#         """
#         Constructor requires a Database instance (the same class that manages
#         the signal_lookup table and can query signal_repository).
#         """
#         self.db = db

#     def run(
#         self,
#         signal_name: Union[str, List[str]],
#         start_date: str = None,
#         end_date: str = None) -> pd.DataFrame:
#         """
#         Entry point to fetch a single signal (string) or multiple signals (list).
#         In either case, we check if there's more coverage in the baseline tables
#         than the repository. If so, we rebuild coverage.

#         For multiple signals, we still do a simpler approach (like the old code).
#         This new 'rebuild coverage if more dates exist' is primarily for single-signal usage.
#         """

#         # 0) If it's multiple signals, do the old multi-signal approach for now (unchanged).
#         if isinstance(signal_name, list):
#             return self._handle_multiple_signals(signal_name, start_date, end_date)

#         # 1) Single signal approach
#         meta = self.db.get_signal(signal_name)
#         if meta is None:
#             print(f"Signal '{signal_name}' not found in lookup table.")
#             return pd.DataFrame()

#         # 2) Check what coverage is in the repository
#         repo_min, repo_max = self.db.get_minmax_dates_from_repository(signal_name)
#         # e.g. returns (datetime.date or datetime, datetime.date or None) or (None, None)

#         # 3) Check what coverage is in the baseline tables
#         #    If it's a baseline signal => left_location is the only table we need
#         #    If it's an interaction => we also have right_location
#         left_table = meta.get("left_location")  # e.g. 'core_raw.fql_ptx_rev'
#         right_table = meta.get("right_location")  # might be None if baseline
#         signal_type = meta.get("signal_type")

#         # We'll see min/max coverage in left_location
#         left_min, left_max = None, None
#         if left_table:
#             left_min, left_max = self.db.get_minmax_dates_in_core_raw(
#                 table_name=left_table.split(".")[1],  # e.g. 'fql_ptx_rev'
#                 column_name=meta["left_baseline"] if signal_type == "interaction" else signal_name
#             )

#         right_min, right_max = None, None
#         if signal_type == "interaction" and right_table:
#             right_min, right_max = self.db.get_minmax_dates_in_core_raw(
#                 table_name=right_table.split(".")[1],
#                 column_name=meta["right_baseline"]
#             )

#         # The "combined" coverage for an interaction is the union of left coverage and right coverage
#         new_coverage_min, new_coverage_max = self._union_coverage(
#             left_min, left_max, right_min, right_max
#         )
#         if new_coverage_min is None or new_coverage_max is None:
#             print(f"No coverage found in baseline tables for '{signal_name}'.")
#             return pd.DataFrame()

#         # Compare with repository coverage
#         if repo_max and new_coverage_max <= repo_max:
#             # Means we do not have any additional coverage in the baseline tables
#             # => No need to rebuild
#             print(f"No new coverage for '{signal_name}'. Using existing repository data.")
#             return self._fetch_from_repository(signal_name, start_date, end_date)

#         # Otherwise, we do a rebuild:
#         print(f"Rebuilding coverage for '{signal_name}' from baseline tables...")

#         # 4) Pull full coverage from left and (if needed) right tables
#         df_left = self.db.pull_data_from_core_raw(
#             table_name=left_table.split(".")[1], 
#             column_name=meta["left_baseline"] if signal_type == "interaction" else signal_name
#         ) if left_table else pd.DataFrame()

#         df_right = None
#         if signal_type == "interaction" and right_table:
#             df_right = self.db.pull_data_from_core_raw(
#                 table_name=right_table.split(".")[1],
#                 column_name=meta["right_baseline"]
#             )

#         # 5) Build the final signal coverage
#         df_signal = self._build_signal_coverage(
#             meta, df_left, df_right
#         )
#         # => columns [date, requestId, <signal_name>]

#         # 6) Delete old coverage from repository
#         self.db.delete_signal_from_repository(signal_name)

#         # 7) Upload new coverage
#         self.db.upload_signal_to_repository(df_signal, signal_name)

#         # 8) Return the coverage (optionally apply start_date/end_date filter)
#         if start_date or end_date:
#             mask = pd.Series(True, index=df_signal.index)
#             if start_date:
#                 mask &= df_signal["date"] >= pd.to_datetime(start_date)
#             if end_date:
#                 mask &= df_signal["date"] <= pd.to_datetime(end_date)
#             df_signal = df_signal[mask]

#         return df_signal

#     ###########################################################################
#     # Multi-signal approach: old logic
#     ###########################################################################
#     def _handle_multiple_signals(self, signals_list, start_date, end_date) -> pd.DataFrame:
#         """
#         If user calls run() with a list of signals, do the old approach:
#         Query the repository, pivot, etc. 
#         This does NOT do the new 'pull from baseline if more coverage' approach.
#         """
#         final_signals = []
#         for s in signals_list:
#             meta = self.db.get_signal(s)
#             if meta is not None:
#                 final_signals.append(s)
#             else:
#                 print(f"Signal '{s}' not found in lookup table. Skipping...")

#         if not final_signals:
#             print("No valid signals found in lookup table.")
#             return pd.DataFrame()

#         df = self.db.get_signal_data_from_repository(final_signals, start_date, end_date)
#         if df.empty:
#             print(f"No data found for signals {final_signals} after date filters.")
#             return df

#         df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

#         # pivot
#         df_pivot = df.pivot_table(
#             index=["date", "requestId"], columns="signal", values="value"
#         ).reset_index()

#         # reorder
#         col_order = ["date", "requestId"] + sorted(final_signals)
#         df_pivot = df_pivot.reindex(columns=col_order, fill_value=None)
#         return df_pivot

#     ###########################################################################
#     # For an interaction, build the signal by applying the function
#     ###########################################################################
#     def _build_signal_coverage(
#         self,
#         meta: dict,
#         df_left: pd.DataFrame,
#         df_right: pd.DataFrame = None
#     ) -> pd.DataFrame:
#         """
#         If meta['signal_type'] == 'baseline', we just rename df_left -> final signal.
#         If it's 'interaction', we create it from df_left & df_right using the 
#         'interaction_type' specified, e.g. 'interaction_squared_sum', 'interaction_product', etc.

#         Returns a DataFrame with columns [date, requestId, meta['signal_name']].
#         """
#         signal_name = meta["signal_name"]
#         sig_type = meta["signal_type"]
#         interact_type = meta.get("interaction_type")

#         # Convert date columns, ensure no duplicates
#         for df_ in [df_left, df_right]:
#             if df_ is not None and not df_.empty:
#                 if "date" in df_.columns:
#                     df_["date"] = pd.to_datetime(df_["date"]).dt.tz_localize(None)
#                 df_.drop_duplicates(["date", "requestId"], inplace=True)

#         if sig_type != "interaction":
#             # Baseline signal: just rename the column to signal_name
#             if "date" not in df_left.columns:
#                 df_left.reset_index(inplace=True)
#             if meta["left_baseline"] in df_left.columns:
#                 df_left.rename(columns={meta["left_baseline"]: signal_name}, inplace=True)
#             return df_left[["date", "requestId", signal_name]].copy()

#         # Otherwise, build an interaction
#         # We'll do a join on [date, requestId]
#         left_baseline = meta["left_baseline"]
#         right_baseline = meta["right_baseline"]
#         if df_right is None:
#             print("Right table is missing for an interaction. Returning empty DataFrame.")
#             return pd.DataFrame(columns=["date", "requestId", signal_name])

#         # Make sure data is in [date, requestId, left_baseline], same for df_right
#         if left_baseline != signal_name:  # i.e. we only rename if it's baseline alone
#             df_left.rename(columns={left_baseline: "left_col"}, inplace=True)
#         else:
#             df_left.rename(columns={signal_name: "left_col"}, inplace=True)

#         df_right.rename(columns={right_baseline: "right_col"}, inplace=True)

#         # We do an inner join
#         merged = pd.merge(
#             df_left[["date", "requestId", "left_col"]],
#             df_right[["date", "requestId", "right_col"]],
#             on=["date", "requestId"], how="inner"
#         )

#         # Now apply the interaction function
#         merged[signal_name] = self._apply_interaction(
#             interact_type, merged["left_col"], merged["right_col"]
#         )

#         return merged[["date", "requestId", signal_name]]

#     def _apply_interaction(self, interact_type: str, s1: pd.Series, s2: pd.Series) -> pd.Series:
#         """
#         Implementation for each known interaction type:
#           'interaction_product', 'interaction_sum', etc.
#         Example code from your FeatureEngineer class.
#         """
#         if interact_type == "interaction_product":
#             return s1 * s2
#         elif interact_type == "interaction_sum":
#             return s1 + s2
#         elif interact_type == "interaction_ratio":
#             return s1 / (s2 + 1e-5)
#         elif interact_type == "interaction_squared_sum":
#             return s1**2 + s2**2
#         elif interact_type == "interaction_diff_squared":
#             return (s1 - s2)**2
#         elif interact_type == "interaction_harmonic_mean":
#             return 2 / (1/(s1+1e-5) + 1/(s2+1e-5))
#         elif interact_type == "interaction_geometric_mean":
#             return (s1 * s2) ** 0.5
#         elif interact_type == "interaction_high_order":
#             return s1 * (s2**2)
#         else:
#             print(f"Unknown interaction type '{interact_type}'. Returning NaN.")
#             return np.nan

#     ###########################################################################
#     # Combine coverage for left + right (for an interaction) to find union
#     ###########################################################################
#     def _union_coverage(
#         self, left_min, left_max, right_min, right_max
#     ):
#         """
#         Return (combined_min, combined_max) as the union of left + right coverage.
#         If either side is missing, we use the other side's min/max.
#         """
#         # Convert None or NaN to sentinel
#         if left_min is None and right_min is None:
#             return (None, None)

#         possible_mins = []
#         if left_min: 
#             possible_mins.append(left_min)
#         if right_min: 
#             possible_mins.append(right_min)

#         possible_maxs = []
#         if left_max:
#             possible_maxs.append(left_max)
#         if right_max:
#             possible_maxs.append(right_max)

#         if not possible_mins or not possible_maxs:
#             return (None, None)

#         combined_min = min(possible_mins)
#         combined_max = max(possible_maxs)
#         return (combined_min, combined_max)

#     ###########################################################################
#     # A helper to simply fetch from repository (start_date, end_date)
#     ###########################################################################
#     def _fetch_from_repository(
#         self, signal_name: str, start_date: str = None, end_date: str = None
#     ) -> pd.DataFrame:
#         df = self.db.get_signal_data_from_repository(signal_name, start_date, end_date)
#         if df.empty:
#             print(f"No data found in repository for '{signal_name}'.")
#             return df

#         df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
#         df.rename(columns={"value": signal_name}, inplace=True)
#         df.drop(columns=["signal"], inplace=True)
#         return df[["date", "requestId", signal_name]]
