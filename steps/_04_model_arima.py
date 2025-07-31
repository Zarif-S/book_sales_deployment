import json
import pandas as pd
import numpy as np
import optuna
import pickle
from typing import Tuple, Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# -----------------------
# TRAIN/TEST SPLIT FUNCTION
# -----------------------
def train_test_split_series(series: pd.Series, forecast_horizon: int = 32):
    train = series[:-forecast_horizon]
    test = series[-forecast_horizon:]
    return train, test

# -----------------------
# OPTUNA OBJECTIVE FUNCTION
# -----------------------
def arima_objective(trial, train: pd.Series, test: pd.Series, seasonal_period: int):
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)

    try:
        model = SARIMAX(train,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.forecast(steps=len(test))
        mae = mean_absolute_error(test, forecast)
        return mae
    except Exception as e:
        return np.inf  # penalize failed fits

# -----------------------
# HYPERPARAM TUNING
# -----------------------
def tune_arima(series: pd.Series, seasonal_period: int = 52, n_trials: int = 30) -> Dict[str, Any]:
    train, test = train_test_split_series(series)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: arima_objective(trial, train, test, seasonal_period),
                   n_trials=n_trials)
    return study.best_params, study.best_value

# -----------------------
# FINAL MODEL FIT
# -----------------------
def fit_final_arima(series: pd.Series, best_params: Dict[str, Any], seasonal_period: int = 52):
    model = SARIMAX(series,
                    order=(best_params['p'], best_params['d'], best_params['q']),
                    seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    residuals = series - results.fittedvalues
    return results, residuals
