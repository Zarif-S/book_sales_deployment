import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, Tuple, Annotated
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from typing import Any
import warnings
warnings.filterwarnings('ignore')

from zenml import step, ArtifactConfig
from zenml.logger import get_logger
from zenml.steps import get_step_context
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.client import Client
from zenml.integrations.pandas.materializers.pandas_materializer import PandasMaterializer
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)


class DictMaterializer(BaseMaterializer):
    """Custom materializer for dictionary objects."""
    ASSOCIATED_TYPES = (dict,)

    def load(self, data_type: type) -> dict:
        """Load a dictionary from the artifact store."""
        import json
        with self.artifact_store.open(self.uri, "r") as f:
            return json.load(f)

    def save(self, data: dict) -> None:
        """Save a dictionary to the artifact store."""
        import json
        with self.artifact_store.open(self.uri, "w") as f:
            json.dump(data, f, indent=2, default=str)

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
    logger.info(f"Created time series with {len(time_series)} data points")
    logger.info(f"Date range: {time_series.index.min()} to {time_series.index.max()}")
    return time_series


def split_time_series(series: pd.Series, test_size: int = 32) -> tuple:
    """Split time series into train/test sets."""
    if len(series) <= test_size:
        raise ValueError(f"Not enough data for test split. Need > {test_size}, got {len(series)}")
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
    logger.info(f"Split: {len(train_series)} train, {len(test_series)} test periods")
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
        logger.warning(f"Trial failed with params p={p},d={d},q={q},P={P},D={D},Q={Q}: {str(e)}")
        return float("inf")


def run_optuna_optimization(train_series: pd.Series, test_series: pd.Series,
                            n_trials: int = 5, study_name: str = "arima_optimization") -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization with persistent storage."""
    logger.info(f"Starting Optuna optimization with {n_trials} trials (sequential processing)...")
    storage_dir = os.path.expanduser("~/zenml_optuna_storage")
    os.makedirs(storage_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(storage_dir, f'{study_name}.db')}"
    logger.info(f"Using persistent Optuna storage: {storage_url}")
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
        logger.info(f"Best params: {study.best_params}, Best RMSE: {study.best_value:.4f}")
        logger.info(f"Total trials completed: {len(study.trials)}")
        return {
            "best_params": dict(study.best_params),
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "study_name": study_name,
            "storage_url": storage_url
        }
    except Exception as e:
        logger.error(f"Optuna optimization failed: {str(e)}")
        return {
            "best_params": {"p": 1, "d": 1, "q": 2, "P": 1, "D": 0, "Q": 1},
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


@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def train_arima_optuna_step(
    modelling_data: pd.DataFrame,
    n_trials: int = 3,
    study_name: str = "arima_optimization"
) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="arima_results")],
    Annotated[str, ArtifactConfig(name="best_hyperparameters_json")],
    Annotated[Any, ArtifactConfig(name="trained_model")]
]:
    """
    ARIMA training step with persistent Optuna storage and ZenML caching.
    """
    logger.info("Starting ARIMA + Optuna training step")
    logger.info(f"Input data shape: {modelling_data.shape}")
    logger.info(f"Optuna study name: {study_name}, n_trials: {n_trials}")

    required_cols = ['book_name', 'date', 'volume', 'data_type', 'isbn']
    missing_cols = [col for col in required_cols if col not in modelling_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_work = modelling_data.copy()
    df_work['isbn'] = df_work['isbn'].astype(str)

    try:
        time_series = create_time_series_from_df(df_work, target_col="volume", date_col="date")
        train_series, test_series = split_time_series(time_series, test_size=32)
        optimization_results = run_optuna_optimization(
            train_series,
            test_series,
            n_trials=n_trials,
            study_name=study_name
        )
        best_params = optimization_results["best_params"]

        eval_model = train_final_arima_model(train_series, best_params)
        eval_forecast = eval_model.forecast(steps=len(test_series))
        eval_metrics = evaluate_forecast(test_series.values, eval_forecast.values)
        logger.info(f"Evaluation metrics on test set: {eval_metrics}")

        final_model = train_final_arima_model(time_series, best_params)

        try:
            client = Client()
            if isinstance(client.active_stack.experiment_tracker, MLFlowExperimentTracker):
                import mlflow
                with mlflow.start_run(run_name="arima_optuna_training"):
                    mlflow.log_params(best_params)
                    for metric_name, metric_value in eval_metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    mlflow.log_metric("data_points", len(time_series))
                    mlflow.log_metric("training_books", df_work['book_name'].nunique())
                    mlflow.log_metric("optuna_trials", optimization_results["n_trials"])
                    mlflow.log_metric("best_rmse", optimization_results["best_value"])
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        context = get_step_context()
        metadata_to_log = {
        "best_params": str(best_params),
        "eval_metrics": str(eval_metrics),
        "training_periods": len(time_series),
        "books_count": df_work['book_name'].nunique(),
        "study_name": study_name,
        "optuna_trials": optimization_results["n_trials"],
        "best_rmse": optimization_results["best_value"]
        }
        # Add each piece of metadata individually for safety
        context.add_output_metadata(
        output_name="arima_results",
        metadata=metadata_to_log
        )
        context.add_output_metadata(
            output_name="best_hyperparameters_json",
            metadata=metadata_to_log
        )
        # Correct way to log individual metadata items
        context.add_output_metadata(
            output_name="trained_model",
            metadata={"model_type": "SARIMAX", "best_params": str(best_params)}
        )

        results_data = []
        results_data.append({
            'result_type': 'model_config',
            'component': 'hyperparameters',
            'parameter': 'arima_order',
            'value': f"({best_params['p']},{best_params['d']},{best_params['q']})",
            'timestamp': pd.Timestamp.now(),
            'metadata': str(best_params)
        })
        results_data.append({
            'result_type': 'model_config',
            'component': 'hyperparameters',
            'parameter': 'seasonal_order',
            'value': f"({best_params['P']},{best_params['D']},{best_params['Q']},52)",
            'timestamp': pd.Timestamp.now(),
            'metadata': 'weekly_seasonality'
        })
        results_data.append({
            'result_type': 'optimization',
            'component': 'optuna_study',
            'parameter': 'study_info',
            'value': f"study_name:{study_name},trials:{optimization_results['n_trials']},best_rmse:{optimization_results['best_value']:.4f}",
            'timestamp': pd.Timestamp.now(),
            'metadata': str(optimization_results)
        })
        for metric_name, metric_value in eval_metrics.items():
            results_data.append({
                'result_type': 'evaluation',
                'component': 'test_metrics',
                'parameter': metric_name,
                'value': f"{metric_value:.4f}",
                'timestamp': pd.Timestamp.now(),
                'metadata': 'test_set_performance'
            })
        for i, (date, actual, predicted) in enumerate(zip(
            test_series.index[-10:], test_series.values[-10:], eval_forecast.values[-10:]
        )):
            results_data.append({
                'result_type': 'forecast',
                'component': 'evaluation_forecast',
                'parameter': f'period_{i+1}',
                'value': f"actual:{actual:.2f},predicted:{predicted:.2f}",
                'timestamp': date,
                'metadata': f'test_period_{i+1}'
            })
        results_data.append({
            'result_type': 'summary',
            'component': 'training_info',
            'parameter': 'data_summary',
            'value': f"periods:{len(time_series)},books:{df_work['book_name'].nunique()},total_volume:{time_series.sum():.0f}",
            'timestamp': pd.Timestamp.now(),
            'metadata': 'training_data_summary'
        })

        results_df = pd.DataFrame(results_data)
        training_summary = {
            "data_periods": len(time_series),
            "books_count": df_work['book_name'].nunique(),
            "total_volume": float(time_series.sum())
        }
        hyperparameters_dict = {
            "best_params": best_params,
            "optimization_results": optimization_results,
            "eval_metrics": eval_metrics,
            "study_name": study_name,
            "training_summary": training_summary,
            "model_signature": f"SARIMAX_({best_params['p']},{best_params['d']},{best_params['q']})_({best_params['P']},{best_params['D']},{best_params['Q']},52)"
        }
        best_hyperparameters_json = json.dumps(hyperparameters_dict, indent=2, default=str)

        logger.info(f"ARIMA training completed successfully!")
        logger.info(f"Results DataFrame shape: {results_df.shape}")
        logger.info(f"Best ARIMA parameters: {best_params}")
        logger.info(f"Test performance - RMSE: {eval_metrics['rmse']:.2f}, MAE: {eval_metrics['mae']:.2f}")
        logger.info(f"Trained on {len(time_series)} periods from {df_work['book_name'].nunique()} books")

        return results_df, best_hyperparameters_json, final_model

    except Exception as e:
        logger.error(f"ARIMA training failed: {str(e)}")
        error_df = pd.DataFrame([{
            'result_type': 'error',
            'component': 'training_error',
            'parameter': 'error_message',
            'value': str(e),
            'timestamp': pd.Timestamp.now(),
            'metadata': 'training_failure'
        }])
        error_hyperparameters_dict = {
            "error": str(e),
            "best_params": {"p": 1, "d": 1, "q": 2, "P": 1, "D": 0, "Q": 1},
            "study_name": study_name,
            "model_signature": "ERROR_MODEL"
        }
        error_hyperparameters_json = json.dumps(error_hyperparameters_dict, indent=2, default=str)
        return error_df, error_hyperparameters_json, None


def parse_hyperparameters_json(json_string: str) -> Dict[str, Any]:
    """
    Helper function to parse hyperparameters JSON string back to dict.
    """
    return json.loads(json_string)
