import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import warnings
warnings.filterwarnings('ignore')

from zenml import step
from zenml.logger import get_logger
from zenml.steps import get_step_context
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.client import Client

logger = get_logger(__name__)

def create_time_series_from_df(df: pd.DataFrame, target_col: str = "volume", 
                              date_col: str = "date") -> pd.Series:
    """
    Convert DataFrame to time series for ARIMA modeling.
    Assumes data is already filtered and prepared from your prep step.
    
    Args:
        df: Prepared DataFrame from your modelling prep step
        target_col: Target column name (volume)
        date_col: Date column name
        
    Returns:
        pd.Series with datetime index and volume values
    """
    df_work = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[date_col]):
        df_work[date_col] = pd.to_datetime(df_work[date_col])
    
    # Sort by date
    df_work = df_work.sort_values(date_col)
    
    # Since your prep already aggregates by date/book, we'll aggregate across all books per date
    # This creates a total volume time series
    time_series = df_work.groupby(date_col)[target_col].sum()
    
    logger.info(f"Created time series with {len(time_series)} data points")
    logger.info(f"Date range: {time_series.index.min()} to {time_series.index.max()}")
    
    return time_series

def split_time_series(series: pd.Series, test_size: int = 32) -> tuple:
    """
    Split time series into train/test sets.
    
    Args:
        series: Time series data
        test_size: Number of periods for test set
        
    Returns:
        Tuple of (train_series, test_series)
    """
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
    
    # Handle MAPE calculation with zero values
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
        
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}

def objective(trial, train_series: pd.Series, test_series: pd.Series) -> float:
    """Optuna objective function to minimize RMSE."""
    # Suggest hyperparameters
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)
    
    # Determine seasonality based on data frequency
    # Since your data looks weekly, use weekly seasonality
    s = 52  # weekly seasonality (52 weeks per year)

    try:
        # Fit SARIMAX model
        model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        fitted = model.fit(disp=False, maxiter=50)
        
        # Generate forecast
        forecast = fitted.forecast(steps=len(test_series))
        
        # Calculate RMSE
        metrics = evaluate_forecast(test_series.values, forecast.values)
        return metrics["rmse"]
        
    except Exception as e:
        logger.warning(f"Trial failed with params p={p},d={d},q={q},P={P},D={D},Q={Q}: {str(e)}")
        return float("inf")

def run_optuna_optimization(train_series: pd.Series, test_series: pd.Series, 
                          n_trials: int = 30) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization."""
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_series, test_series), 
                  n_trials=n_trials, timeout=1800)  # 30 min timeout
    
    logger.info(f"Best params: {study.best_params}, Best RMSE: {study.best_value:.4f}")
    
    return study.best_params

def train_final_arima_model(series: pd.Series, best_params: Dict[str, int]):
    """Train final ARIMA model with best parameters."""
    p, d, q = best_params['p'], best_params['d'], best_params['q']
    P, D, Q = best_params['P'], best_params['D'], best_params['Q']
    s = 52  # weekly seasonality
    
    model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    fitted_model = model.fit(disp=False, maxiter=100)
    
    return fitted_model

@step
def train_arima_optuna_step(modelling_data: pd.DataFrame) -> pd.DataFrame:
    """
    ARIMA training step that works with your prepared modelling_data DataFrame.
    
    Args:
        modelling_data: DataFrame with columns [book_name, date, volume, data_type, isbn]
                       Already prepared and filtered by your prep step
        
    Returns:
        DataFrame containing model results, parameters, forecasts, and residuals
    """
    logger.info("Starting ARIMA + Optuna training step")
    logger.info(f"Input data shape: {modelling_data.shape}")
    logger.info(f"Input columns: {list(modelling_data.columns)}")
    
    # Validate input DataFrame structure
    required_cols = ['book_name', 'date', 'volume', 'data_type', 'isbn']
    missing_cols = [col for col in required_cols if col not in modelling_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Preserve ISBN as string and make working copy
    df_work = modelling_data.copy()
    df_work['isbn'] = df_work['isbn'].astype(str)
    
    try:
        # Convert DataFrame to time series (aggregating across all books by date)
        time_series = create_time_series_from_df(df_work, target_col="volume", date_col="date")
        
        # Split into train/test for optimization
        train_series, test_series = split_time_series(time_series, test_size=32)
        
        # Run Optuna optimization
        best_params = run_optuna_optimization(train_series, test_series, n_trials=30)
        
        # Train model on training data for evaluation
        eval_model = train_final_arima_model(train_series, best_params)
        eval_forecast = eval_model.forecast(steps=len(test_series))
        eval_metrics = evaluate_forecast(test_series.values, eval_forecast.values)
        
        logger.info(f"Evaluation metrics on test set: {eval_metrics}")
        
        # Train final model on full time series
        final_model = train_final_arima_model(time_series, best_params)
        
        # Extract model components
        residuals = final_model.resid
        fitted_values = final_model.fittedvalues
        
        # Serialize model
        model_bytes = pickle.dumps(final_model)
        
        # MLflow tracking
        try:
            client = Client()
            if isinstance(client.active_stack.experiment_tracker, MLFlowExperimentTracker):
                import mlflow
                with mlflow.start_run(run_name="arima_optuna_training"):
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(eval_metrics)
                    mlflow.log_metric("data_points", len(time_series))
                    mlflow.log_metric("training_books", df_work['book_name'].nunique())
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
        
        # Add metadata to step context
        context = get_step_context()
        context.add_output_metadata("arima_model", {
            **best_params, 
            **eval_metrics,
            "training_periods": len(time_series),
            "books_count": df_work['book_name'].nunique()
        })
        
        # Create comprehensive results DataFrame
        results_data = []
        
        # Model configuration
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
        
        # Evaluation metrics
        for metric_name, metric_value in eval_metrics.items():
            results_data.append({
                'result_type': 'evaluation',
                'component': 'test_metrics',
                'parameter': metric_name,
                'value': f"{metric_value:.4f}",
                'timestamp': pd.Timestamp.now(),
                'metadata': 'test_set_performance'
            })
        
        # Forecast evaluation results
        for i, (date, actual, predicted) in enumerate(zip(test_series.index, test_series.values, eval_forecast.values)):
            results_data.append({
                'result_type': 'forecast',
                'component': 'evaluation_forecast',
                'parameter': f'period_{i+1}',
                'value': f"actual:{actual:.2f},predicted:{predicted:.2f}",
                'timestamp': date,
                'metadata': f'test_period_{i+1}'
            })
        
        # Model diagnostics (residuals and fitted values)
        for i, (date, resid, fitted) in enumerate(zip(time_series.index, residuals, fitted_values)):
            results_data.append({
                'result_type': 'diagnostics',
                'component': 'model_fit',
                'parameter': f'period_{i+1}',
                'value': f"residual:{resid:.4f},fitted:{fitted:.2f},actual:{time_series.iloc[i]:.2f}",
                'timestamp': date,
                'metadata': f'training_period_{i+1}'
            })
        
        # Store serialized model
        results_data.append({
            'result_type': 'model_artifact',
            'component': 'trained_model',
            'parameter': 'pickled_model',
            'value': model_bytes.hex(),  # Convert to hex string for DataFrame storage
            'timestamp': pd.Timestamp.now(),
            'metadata': f'sarimax_model_{len(time_series)}_periods'
        })
        
        # Model summary statistics
        results_data.append({
            'result_type': 'summary',
            'component': 'training_info',
            'parameter': 'data_summary',
            'value': f"periods:{len(time_series)},books:{df_work['book_name'].nunique()},total_volume:{time_series.sum():.0f}",
            'timestamp': pd.Timestamp.now(),
            'metadata': 'training_data_summary'
        })
        
        # Create final results DataFrame
        results_df = pd.DataFrame(results_data)
        
        logger.info(f"ARIMA training completed successfully!")
        logger.info(f"Results DataFrame shape: {results_df.shape}")
        logger.info(f"Best ARIMA parameters: {best_params}")
        logger.info(f"Test performance - RMSE: {eval_metrics['rmse']:.2f}, MAE: {eval_metrics['mae']:.2f}")
        logger.info(f"Trained on {len(time_series)} periods from {df_work['book_name'].nunique()} books")
        
        return results_df
        
    except Exception as e:
        logger.error(f"ARIMA training failed: {str(e)}")
        # Return error DataFrame to maintain pipeline flow
        error_df = pd.DataFrame([{
            'result_type': 'error',
            'component': 'training_error',
            'parameter': 'error_message',
            'value': str(e),
            'timestamp': pd.Timestamp.now(),
            'metadata': 'training_failure'
        }])
        return error_df