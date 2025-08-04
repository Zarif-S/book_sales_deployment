import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import Dict, Any, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import warnings

warnings.filterwarnings('ignore')

def create_time_series_from_df(df: pd.DataFrame, target_col: str = "volume",
                              date_col: str = "date") -> pd.Series:
    """
    Convert DataFrame to time series for ARIMA modeling.
    Assumes data is already filtered and prepared from your prep step.
    """
    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work[date_col]):
        df_work[date_col] = pd.to_datetime(df_work[date_col])

    df_work = df_work.sort_values(date_col)
    time_series = df_work.groupby(date_col)[target_col].sum()

    return time_series

def split_time_series(series: pd.Series, test_size: int = 32) -> tuple:
    """Split time series into train/test sets."""
    if len(series) <= test_size:
        raise ValueError(f"Not enough data for test split. Need > {test_size}, got {len(series)}")

    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]

    return train_series, test_series

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics with error handling."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}

def objective(trial, train_series: pd.Series, test_series: pd.Series) -> float:
    """Optuna objective function to minimize RMSE."""
    p = trial.suggest_int("p", 0, 2)
    d = trial.suggest_int("d", 1, 2)
    q = trial.suggest_int("q", 2, 4)
    P = trial.suggest_int("P", 0, 2)
    D = trial.suggest_int("D", 0, 1)
    Q = trial.suggest_int("Q", 1, 3)
    s = 52

    try:
        model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        fitted = model.fit(disp=False, maxiter=50)
        forecast = fitted.forecast(steps=len(test_series))
        metrics = evaluate_forecast(test_series.values, forecast.values)
        return metrics["rmse"]
    except Exception as e:
        return float("inf")

def run_optuna_optimization(train_series: pd.Series, test_series: pd.Series,
                           n_trials: int, study_name: str = "arima_optimization") -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization with persistent storage."""
    storage_dir = os.path.expanduser("~/zenml_optuna_storage")
    os.makedirs(storage_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(storage_dir, f'{study_name}.db')}"

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize"
        )

        study.optimize(
            lambda trial: objective(trial, train_series, test_series),
            n_trials=n_trials,
            timeout=1800,
            n_jobs=1
        )

        return {
            "best_params": dict(study.best_params),
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "study_name": study_name,
            "storage_url": storage_url
        }

    except Exception as e:
        return {
            "best_params": {"p": 1, "d": 1, "q": 2, "P": 1, "D": 0, "Q": 1}, # Formerly 
            "best_value": float("inf"),
            "n_trials": 0,
            "study_name": study_name,
            "storage_url": storage_url,
            "error": str(e)
        }

def train_final_arima_model(series: pd.Series, best_params: Dict[str, int]):
    """Train final ARIMA model with best parameters."""
    p, d, q = best_params['p'], best_params['d'], best_params['q']
    P, D, Q = best_params['P'], best_params['D'], best_params['Q']
    s = 52

    model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    fitted_model = model.fit(disp=False, maxiter=100)

    return fitted_model

def parse_hyperparameters_json(json_string: str) -> Dict[str, Any]:
    """
    Helper function to parse hyperparameters JSON string back to dict.
    """
    return json.loads(json_string)
