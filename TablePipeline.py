import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from google.cloud import bigquery

# Optional: Only import catboost if you plan to use it. 
# Otherwise, you can lazily import it inside the class below.
from catboost import CatBoostRegressor

catboost_params = {
    'iterations': 1000,
    'depth': 10,
    'border_count': 128,
    'boosting_type': 'Ordered',
    'task_type': 'GPU',
    'devices': '0',
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 0.5,
    'learning_rate': 0.03,
    'l2_leaf_reg': 1,
    'scale_pos_weight': 2.964,
    'eval_metric': 'F1',   # Typically for classification; adjust if you're doing regression
    'random_seed': 42,
    'verbose': 0,
    'use_best_model': True
}


class DataFetcher:
    """
    A helper class to interact with BigQuery and fetch the necessary data for the pipeline.
    """
    def __init__(self, 
                 project_id: str = 'issachar-feature-library', 
                 dataset_name: str = 'wmg'):
        """
        Initialize the DataFetcher with BigQuery connection details.
        
        Args:
            project_id (str): Google Cloud project ID.
            dataset_name (str): BigQuery dataset name.
        """
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.client = bigquery.Client(project=self.project_id)
    
    def get_unique_signals_from_daily_ic(self, table_name: str = 'ic_daily2'):
        """
        Fetch all unique signal values from the 'signal' column of the daily_ic table.

        Args:
            table_name (str): BigQuery table name containing the daily IC results.

        Returns:
            list: A list of unique signal names.
        """
        query = f"""
        SELECT DISTINCT signal
        FROM `{self.project_id}.{self.dataset_name}.{table_name}`
        """
        query_job = self.client.query(query)
        df = query_job.to_dataframe()
        return df['signal'].unique().tolist()

    def fetch_data_from_gbq(self, table_name: str):
        """
        Fetches all rows & columns from a specified BigQuery table.
        Converts the 'date' column to datetime and sets a MultiIndex of (date, signal).
        
        Args:
            table_name (str): The BigQuery table name.

        Returns:
            pandas.DataFrame: A DataFrame with a MultiIndex of (date, signal).
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_name}.{table_name}`
        """
        query_job = self.client.query(query)
        df = query_job.to_dataframe()

        # Convert date to datetime, drop timezone info
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Make (date, signal) a MultiIndex if both columns exist
        if 'signal' in df.columns and 'date' in df.columns:
            df.set_index(['date', 'signal'], inplace=True)

        return df.sort_index()
    
    def fetch_data_from_gbq_fvalues(self, 
                                   table_name: str, 
                                   signals: list):
        """
        Fetches data from a BigQuery table and filters rows to include only those
        where the "signal" column matches one of the provided signals.
        
        Args:
            table_name (str): The BigQuery table name.
            signals (list): A list of signal values to filter the rows.

        Returns:
            pandas.DataFrame: A DataFrame containing the filtered rows.
        """
        # Prepare the signals list for SQL IN clause by quoting each signal
        signals_str = ", ".join(f"'{signal}'" for signal in signals)

        query = f"""
        SELECT date, signal, 
               fvalue, 
               fvalue_median_21_minus_fvalue_median_252,
               fvalue_median_21
        FROM `{self.project_id}.{self.dataset_name}.{table_name}`
        WHERE signal IN ({signals_str})
        """
        query_job = self.client.query(query)
        df = query_job.to_dataframe()

        # Convert date to datetime, drop timezone info
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df.set_index(['date', 'signal'], inplace=True)

        return df.sort_index()


class FeatureEngineer:
    """
    A helper class to engineer features from the fetched data.
    """
    @staticmethod
    def compute_rolling_ic(df_ic_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Append rolling IC metrics (21-day rolling mean) to the daily IC DataFrame.
        
        Args:
            df_ic_daily (pd.DataFrame): Input DataFrame of daily IC, indexed by (date, signal).
        
        Returns:
            pd.DataFrame: The updated DataFrame with new rolling columns.
        """
        # 21-day rolling for all_IC_spearmanr
        daily_rolling_ic = df_ic_daily.groupby('signal')['all_IC_spearmanr'].rolling(21).mean()
        daily_rolling_ic.index = daily_rolling_ic.index.droplevel(0)

        # 21-day rolling for topq_IC_spearmanr
        daily_rolling_ic_top = df_ic_daily.groupby('signal')['topq_IC_spearmanr'].rolling(21).mean()
        daily_rolling_ic_top.index = daily_rolling_ic_top.index.droplevel(0)

        df_ic_daily['all_IC_spearmanr_rolling_21'] = daily_rolling_ic
        df_ic_daily['topq_IC_spearmanr_rolling_21'] = daily_rolling_ic_top
        
        # Diff feature
        df_ic_daily['rolling_diff'] = daily_rolling_ic - daily_rolling_ic_top
        
        return df_ic_daily

    @staticmethod
    def compute_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
        """
        Compute various rolling transformations (averages, vol, squared terms, etc.)
        on a per-signal basis.

        Args:
            group (pd.DataFrame): Subset of DataFrame belonging to one signal.

        Returns:
            pd.DataFrame: The group with additional rolling feature columns.
        """
        group = group.sort_index()
        
        # Rolling means of all_IC_spearmanr
        group['IC_all_5'] = group['all_IC_spearmanr'].rolling(window=5, min_periods=1).mean()
        group['IC_all_20'] = group['all_IC_spearmanr'].rolling(window=20, min_periods=1).mean()
        group['IC_all_5_diff_IC_all_20'] = group['IC_all_5'] - group['IC_all_20']
        
        # Rolling means of topq_IC_spearmanr
        group['IC_top_5'] = group['topq_IC_spearmanr'].rolling(window=5, min_periods=1).mean()
        group['IC_top_20'] = group['topq_IC_spearmanr'].rolling(window=20, min_periods=1).mean()
        group['IC_top_5_diff_IC_top_20'] = group['IC_top_5'] - group['IC_top_20']
        
        # Rolling volatility (std)
        group['IC_all_vol_5'] = group['all_IC_spearmanr'].rolling(window=5, min_periods=1).std()
        group['IC_all_vol_20'] = group['all_IC_spearmanr'].rolling(window=20, min_periods=1).std()
        group['IC_all_vol_5_diff_IC_all_vol_20'] = group['IC_all_vol_5'] - group['IC_all_vol_20']
        
        group['IC_top_vol_5'] = group['topq_IC_spearmanr'].rolling(window=5, min_periods=1).std()
        group['IC_top_vol_20'] = group['topq_IC_spearmanr'].rolling(window=20, min_periods=1).std()
        group['IC_top_vol_5_diff_IC_top_vol_20'] = group['IC_top_vol_5'] - group['IC_top_vol_20']
        
        # Squared terms
        group['IC_all_sq'] = group['all_IC_spearmanr'] ** 2
        group['IC_top_sq'] = group['topq_IC_spearmanr'] ** 2
        group['IC_all_sq_diff_IC_top_sq'] = group['IC_all_sq'] - group['IC_top_sq']
        
        group['IC_all_vol_5_sq'] = group['IC_all_vol_5'] ** 2
        group['IC_top_vol_5_sq'] = group['IC_top_vol_5'] ** 2
        group['IC_all_vol_5_sq_minus_IC_top_vol_5_sq'] = group['IC_all_vol_5_sq'] - group['IC_top_vol_5_sq']
        
        # Log transform of 5-day vol (small constant to avoid log(0))
        group['log_IC_all_vol_5'] = np.log(group['IC_all_vol_5_sq'] + 1e-8)
        group['log_IC_top_vol_5'] = np.log(group['IC_top_vol_5_sq'] + 1e-8)
        group['log_IC_all_vol_5_diff_log_IC_top_vol_5'] = group['log_IC_all_vol_5'] - group['log_IC_top_vol_5']
        
        # Interaction: product of 5-day rolling average and 5-day volatility
        group['IC_5_x_vol_5_all'] = group['IC_all_5'] * group['IC_all_vol_5']
        group['IC_20_x_vol_20_all'] = group['IC_all_20'] * group['IC_all_vol_20']
        group['IC_5_x_vol_5_top'] = group['IC_top_5'] * group['IC_top_vol_5']
        group['IC_20_x_vol_20_top'] = group['IC_top_20'] * group['IC_top_vol_20']
        
        # Volatility-adjusted IC
        group['IC_all_vol_adj'] = group['IC_all_5'] / (group['IC_all_vol_5'] + 1e-8)
        group['IC_top_vol_adj'] = group['IC_top_5'] / (group['IC_top_vol_5'] + 1e-8)
        
        # Trend indicators
        group['IC_all_trend'] = group['IC_all_5'] - group['IC_all_20']
        group['IC_top_trend'] = group['IC_top_5'] - group['IC_top_20']

        # Rate of change of monotonicity
        group['monotonicity_21_diff_accel_5'] = group['monotonic_21_diff'].pct_change().rolling(5).sum()
        group['monotonic_increasing_pct_21_accel_5'] = group['monotonic_increasing_pct_21'].pct_change().rolling(5).sum()
        group['monotonic_decreasing_pct_21_accel_5'] = group['monotonic_decreasing_pct_21'].pct_change().rolling(5).sum()

        return group

    @staticmethod
    def fix_df(df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a DataFrame or Series to a standard (date, signal) MultiIndex with sorted index.

        Args:
            df_input (pd.DataFrame or pd.Series): Input data.

        Returns:
            pd.DataFrame: Reindexed/sorted DataFrame with MultiIndex (date, signal).
        """
        df_input = df_input.copy()
        if isinstance(df_input, pd.Series):
            col_name = df_input.name
            df_input = df_input.reset_index()
            df_input.columns = ['date', 'signal', col_name]
        else:
            cols = df_input.columns.tolist()
            df_input = df_input.reset_index()
            df_input.columns = ['date', 'signal'] + cols

        df_input.drop_duplicates(['date', 'signal'], inplace=True)
        df_input = df_input.set_index(['date', 'signal']).sort_index()
        return df_input


class ModelTrainer:
    """
    Class responsible for defining and training models in a rolling-window fashion
    and returning the final predictions DataFrame.

    By default, uses LightGBM. If use_catboost=True, uses CatBoost instead.
    """
    def __init__(self, 
                 lookback: int = 252, 
                 forecast_horizon: int = 5,
                 target_col: str = 'target',
                 use_catboost: bool = False,
                 catboost_params: dict = None):
        """
        Initialize the ModelTrainer.

        Args:
            lookback (int): Number of days for rolling training window.
            forecast_horizon (int): Horizon used to define the target window.
            target_col (str): The name of the target column.
            use_catboost (bool): Whether to use CatBoost instead of LightGBM.
            catboost_params (dict): Parameters to pass to CatBoost if use_catboost is True.
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.cpu_count = 2  # or cpu_count(), your choice

        self.use_catboost = use_catboost
        self.catboost_params = catboost_params or {}

    def _process_signal(self, 
                        signal: str, 
                        group: pd.DataFrame, 
                        target_series: pd.Series) -> tuple:
        """
        For a given signal, train a model (LightGBM or CatBoost) using a rolling window approach
        and produce predictions (and feature importances if desired).

        Args:
            signal (str): The signal identifier.
            group (pd.DataFrame): Predictors for this signal over time (indexed by date).
            target_series (pd.Series): Target data for this signal over time (indexed by date).

        Returns:
            tuple: (predictions_list, importances_list)
                   - predictions_list: list of dicts with 'date', 'signal', 'predicted_target', 'actual_target'
                   - importances_list: list of feature importance dicts
        """
        predictions_signal = []
        importances_signal = []

        # Ensure sorted by date
        group = group.sort_index(level='date')
        target_series = target_series.sort_index(level='date')
        n = len(group)

        # Not enough data to do rolling
        if n < self.lookback + self.forecast_horizon:
            return predictions_signal, importances_signal

        # Lazy import CatBoost if actually needed
        if self.use_catboost:
            from catboost import CatBoostRegressor

        for t in range(self.lookback + self.forecast_horizon, n - self.forecast_horizon):
            # Training set
            X_train = group.drop(columns=[self.target_col]).iloc[t - self.lookback - self.forecast_horizon : t - self.forecast_horizon].copy()
            y_train = target_series.iloc[t - self.lookback - self.forecast_horizon : t - self.forecast_horizon].copy()

            # Test set (single row at time t)
            X_test = group.drop(columns=[self.target_col]).iloc[t : t+1].copy()

            # Choose model
            if self.use_catboost:
                model = CatBoostRegressor(**self.catboost_params)
                # For regression tasks, we often just call fit without additional params
                # If you need cat_features, text_features, etc., you must specify them.
                model.fit(X_train, y_train, verbose=0)
            else:
                model = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)[0]
            actual_value = target_series.iloc[t]
            pred_date = group.index[t]

            predictions_signal.append({
                'date': pred_date,
                'signal': signal,
                'predicted_target': y_pred,
                'actual_target': actual_value
            })

            # Feature importances
            if self.use_catboost:
                # CatBoost has feature_importances_ but it's obtained differently
                importance_values = model.get_feature_importance()
                importance_dict = dict(zip(X_train.columns, importance_values))
            else:
                importance_dict = dict(zip(X_train.columns, model.feature_importances_))
            importances_signal.append(importance_dict)

        return predictions_signal, importances_signal

    def train_and_predict(self, df_model: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrate rolling-window training for all signals in parallel and build
        a final predictions DataFrame.

        Args:
            df_model (pd.DataFrame): Contains both features and target, 
                                     indexed by (date, signal).

        Returns:
            pd.DataFrame: Predictions with columns ['predicted_target', 'actual_target'] 
                          and MultiIndex (date, signal).
        """
        # Drop rows without a target
        df_model = df_model.dropna(subset=[self.target_col]).copy()

        # Parallel over signals
        grouped = df_model.groupby(level='signal')
        results = []
        with tqdm_joblib(tqdm(desc="Processing Signals", total=len(grouped))):
            results = Parallel(n_jobs=self.cpu_count)(
                delayed(self._process_signal)(
                    signal,
                    group,
                    df_model.xs(signal, level='signal')[self.target_col]
                )
                for signal, group in grouped
            )

        # Flatten the results
        all_predictions = [pred for res in results for pred in res[0]]
        all_importances = [imp for res in results for imp in res[1]]  # if you want to use them

        df_predictions2 = pd.DataFrame(all_predictions)

        # Clean "date" column in case it's a tuple
        df_predictions2['date'] = df_predictions2['date'].apply(
            lambda x: x[0] if isinstance(x, tuple) else x
        )

        # If "actual_target" is still a Series, fix it
        df_predictions2['actual_target'] = df_predictions2['actual_target'].apply(
            lambda x: x.values[0] if isinstance(x, pd.Series) else x
        )

        # Convert "date" to datetime
        df_predictions2['date'] = pd.to_datetime(df_predictions2['date'])

        # Create MultiIndex
        df_predictions2.set_index(['date', 'signal'], inplace=True)

        return df_predictions2


class Pipeline:
    """
    An end-to-end pipeline that:
      1) Fetches raw data from BigQuery,
      2) Engineers features and targets,
      3) Trains the model in a rolling fashion,
      4) Returns the final predictions DataFrame (df_predictions2).
    """
    def __init__(self, 
                 project_id='issachar-feature-library', 
                 dataset_name='wmg',
                 invalid_signals=None,
                 use_catboost=False,
                 catboost_params=None):
        """
        Initialize the pipeline.

        Args:
            project_id (str): Google Cloud project ID.
            dataset_name (str): BigQuery dataset name.
            invalid_signals (list): A list of signals to exclude, if any.
            use_catboost (bool): Whether to use CatBoost (True) or LightGBM (False, default).
            catboost_params (dict): Parameters for CatBoost if use_catboost=True.
        """
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.invalid_signals = invalid_signals if invalid_signals else []
        
        self.data_fetcher = DataFetcher(project_id=project_id, 
                                        dataset_name=dataset_name)
        self.feature_engineer = FeatureEngineer()

        # We will instantiate ModelTrainer in run_pipeline()
        self.model_trainer = None  

        # Placeholders
        self.all_signals = []
        self.df_ic_daily = None
        self.df_fvalues = None
        self.df_mi_rolling = None
        self.df_mi_daily = None
        self.df_mono = None
        self.df_combined = None
        self.df_features_enhanced = None
        self.df_target = None

        self.use_catboost = use_catboost
        self.catboost_params = catboost_params or {}

    def run_pipeline(self,
                     table_ic_daily='daily_ic2',
                     table_fvalues='daily_fvalue_interactions',
                     table_mi_rolling='mi_rolling',
                     table_mi_daily='mi_daily',
                     table_monotonicity='daily_rolling_monotonic_pct_st',
                     target_duration='regular',  # or 'rolling'
                     target_abs=True):
        """
        Execute the entire pipeline, from data fetch to final predictions. 
        Returns df_predictions2.

        Args:
            table_ic_daily (str): Table name for daily IC data.
            table_fvalues (str): Table name for f-values data.
            table_mi_rolling (str): Table name for rolling MI data.
            table_mi_daily (str): Table name for daily MI data.
            table_monotonicity (str): Table name for daily rolling monotonicity data.
            target_duration (str): 'regular' or 'rolling' target.
            target_abs (bool): Whether to use absolute IC for target in rolling mode.

        Returns:
            pd.DataFrame: The final predictions DataFrame (df_predictions2).
        """
        # 1. Fetch signals and filter them
        all_signals_raw = self.data_fetcher.get_unique_signals_from_daily_ic(table_ic_daily)
        self.all_signals = [s for s in all_signals_raw if s not in self.invalid_signals]

        # 2. Fetch daily IC data
        self.df_ic_daily = self.data_fetcher.fetch_data_from_gbq(table_ic_daily)
        self.df_ic_daily = self.df_ic_daily[~self.df_ic_daily.index.duplicated(keep='first')]

        # Compute 21-day rolling
        self.df_ic_daily = self.feature_engineer.compute_rolling_ic(self.df_ic_daily)

        # 3. Fetch f-values, mi_rolling, mi_daily, monotonicity
        self.df_fvalues = self.data_fetcher.fetch_data_from_gbq_fvalues(table_fvalues, self.all_signals)
        self.df_fvalues = self.df_fvalues[~self.df_fvalues.index.duplicated(keep='first')]
        
        self.df_mi_rolling = self.data_fetcher.fetch_data_from_gbq(table_mi_rolling)
        self.df_mi_rolling = self.df_mi_rolling[~self.df_mi_rolling.index.duplicated(keep='first')]

        self.df_mi_daily = self.data_fetcher.fetch_data_from_gbq(table_mi_daily)
        self.df_mi_daily = self.df_mi_daily[~self.df_mi_daily.index.duplicated(keep='first')]

        self.df_mono = self.data_fetcher.fetch_data_from_gbq(table_monotonicity)
        self.df_mono = self.df_mono[~self.df_mono.index.duplicated(keep='first')]
        self.df_mono['monotonic_21_diff'] = (
            self.df_mono['monotonic_increasing_pct_21'] - 
            self.df_mono['monotonic_decreasing_pct_21']
        )

        # 4. Combine everything
        self.df_combined = self.df_ic_daily \
            .join(self.df_fvalues) \
            .join(self.df_mi_rolling) \
            .join(self.df_mi_daily) \
            .join(self.df_mono) \
            .copy()

        # 5. Engineer rolling features on combined data
        self.df_features_enhanced = self.df_combined.groupby(level='signal') \
            .apply(self.feature_engineer.compute_rolling_features).copy()
        # The grouping above can shift the signal level up, so drop it
        self.df_features_enhanced.index = self.df_features_enhanced.index.droplevel(0)

        # 6. Define the target
        if target_duration == 'regular':
            self.df_target = self.df_combined['all_IC_spearmanr'].copy()
        elif target_duration == 'rolling':
            roll = 5
            shift_n = 4
            if not target_abs:
                # Rolling std of future 5 days
                df_forward_5day_std_ic = self.df_combined.groupby(level='signal')['all_IC_spearmanr'] \
                    .apply(lambda x: x.shift(-1).rolling(window=roll).std().shift(-shift_n))
                df_forward_5day_std_ic.index = df_forward_5day_std_ic.index.droplevel(0)
                df_forward_5day_std_ic.name = 'target'
                self.df_target = df_forward_5day_std_ic.copy()
            else:
                # Rolling std of future 5 days of abs(IC)
                df_forward_5day_std_ic_abs = self.df_combined.groupby(level='signal')['all_IC_spearmanr'] \
                    .apply(lambda x: x.shift(-1).abs().rolling(window=roll).std().shift(-shift_n))
                df_forward_5day_std_ic_abs.index = df_forward_5day_std_ic_abs.index.droplevel(0)
                df_forward_5day_std_ic_abs.name = 'target'
                self.df_target = df_forward_5day_std_ic_abs.copy()

        # Standardize target name
        self.df_target.name = 'target'

        # 7. Shift features by 2 days to avoid lookahead
        df_features_enhanced_shifted = self.df_features_enhanced.groupby(level='signal').shift(2).copy()
        df_features_enhanced_shifted = FeatureEngineer.fix_df(df_features_enhanced_shifted)

        # 8. Combine shifted features + target
        df_model = pd.concat([df_features_enhanced_shifted, 
                              FeatureEngineer.fix_df(self.df_target)], 
                             axis=1)

        # 9. Train the model in a rolling-window and return predictions
        self.model_trainer = ModelTrainer(
            lookback=252, 
            forecast_horizon=5, 
            target_col='target',
            use_catboost=self.use_catboost,
            catboost_params=self.catboost_params
        )
        df_predictions2 = self.model_trainer.train_and_predict(df_model)

        return df_predictions2


if __name__ == '__main__':
    
    catboost = False
    
    if not catboost:
        # Example usage:

        # By default, this uses LightGBM:
        pipeline_lgb = Pipeline(
            project_id='issachar-feature-library', 
            dataset_name='wmg',
            invalid_signals=[],  # or pass a list of signals to exclude
            use_catboost=False   # <--- LightGBM is the default
        )

        df_predictions2_lgb = pipeline_lgb.run_pipeline(
            table_ic_daily='daily_ic2',
            table_fvalues='daily_fvalue_interactions',
            table_mi_rolling='mi_rolling',
            table_mi_daily='mi_daily',
            table_monotonicity='daily_rolling_monotonic_pct_st',
            target_duration='regular',   # or 'rolling'
            target_abs=False
        )

        df_predictions2_lgb.to_gbq(
            destination_table='wmg.ic_predictions_v2_lgb', 
            project_id='issachar-feature-library', 
            if_exists='replace'
        )
        print("Predictions uploaded to BigQuery table 'wmg.ic_predictions_v2' successfully!")
        
    else:

        # Example: Use CatBoost instead:
        pipeline_cb = Pipeline(
            project_id='issachar-feature-library', 
            dataset_name='wmg',
            invalid_signals=[], 
            use_catboost=True,
            catboost_params=catboost_params  # your CatBoost parameter dict
        )

        df_predictions2_cb = pipeline_cb.run_pipeline(
            table_ic_daily='daily_ic2',
            table_fvalues='daily_fvalue_interactions',
            table_mi_rolling='mi_rolling',
            table_mi_daily='mi_daily',
            table_monotonicity='daily_rolling_monotonic_pct_st',
            target_duration='regular',
            target_abs=False
        )

        df_predictions2_cb.to_gbq(
            destination_table='wmg.ic_predictions_v2_catboost', 
            project_id='issachar-feature-library', 
            if_exists='replace'
        )
        print("CatBoost Predictions uploaded to BigQuery table 'wmg.ic_predictions_v2_catboost' successfully!")
