import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

from zenml import step
from zenml.logger import get_logger
from zenml.steps import get_step_context
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.client import Client
from .modelling_prep import prepare_data_after_2012

logger = get_logger(__name__)

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}

def objective(trial, train_series: pd.Series, test_series: pd.Series):
    """Optuna objective to minimize RMSE."""
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)
    s = 52  # weekly seasonality

    try:
        model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=len(test_series))
        metrics = evaluate_forecast(test_series.values, forecast.values)
        return metrics["rmse"]
    except Exception as e:
        logger.warning(f"Trial failed with error: {e}")
        return float("inf")

def run_optuna(train_series: pd.Series, test_series: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
    """Run Optuna hyperparameter tuning."""
    logger.info("Starting Optuna hyperparameter tuning for ARIMA...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_series, test_series), n_trials=n_trials)
    logger.info(f"Best trial params: {study.best_params} with value {study.best_value}")
    return study.best_params

def train_best_arima(series: pd.Series, best_params: Dict[str, int]) -> SARIMAX:
    """Train ARIMA using best parameters."""
    p, d, q = best_params['p'], best_params['d'], best_params['q']
    P, D, Q = best_params['P'], best_params['D'], best_params['Q']
    s = 52
    model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    return model.fit(disp=False)

@step
def train_arima_optuna_step(df_merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Hyperparameter tuning using Optuna, final ARIMA model training,
    evaluation, and MLflow logging, returning model, forecast, residuals.
    """
    logger.info("Running ARIMA + Optuna step")

    target_col = "Volume"  # adjust for your dataframe
    
    # Use the new data preparation function for data after 2012
    train_series, test_series = prepare_data_after_2012(df_merged, target_col, split_size=32)

    # Run Optuna tuning on train/test split
    best_params = run_optuna(train_series, test_series, n_trials=30)

    # Train model on train split with best params for evaluation
    fitted_model = train_best_arima(train_series, best_params)
    forecast = fitted_model.forecast(steps=len(test_series))
    metrics = evaluate_forecast(test_series.values, forecast.values)
    logger.info(f"ARIMA metrics on test split: {metrics}")

    # Train final model on full series with best params
    final_model_fit = train_best_arima(series, best_params)

    # Extract residuals from full model fit
    residuals = final_model_fit.resid

    # Serialize final model for passing as artifact
    final_model_bytes = pickle.dumps(final_model_fit)

    # MLflow tracking (optional)
    client = Client()
    if isinstance(client.active_stack.experiment_tracker, MLFlowExperimentTracker):
        import mlflow
        with mlflow.start_run(run_name="arima_optuna_final"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)

    context = get_step_context()
    context.add_output_metadata("arima_model", {**best_params, **metrics})

    return {
        "best_params": best_params,
        "metrics": metrics,
        "model": final_model_bytes,
        "forecast": forecast.to_list(),
        "test_series": test_series.to_list(),
        "residuals": residuals.to_list()
    }
