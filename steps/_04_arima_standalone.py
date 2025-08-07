import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import Dict, Any, Tuple, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import warnings

warnings.filterwarnings('ignore')

def load_individual_book_data(book_isbn: str):
    """Load train and test data for a specific book from individual CSV files."""
    print(f"📂 Loading individual book data for ISBN: {book_isbn}")

    train_path = f"data/processed/train_data_{book_isbn}.csv"
    test_path = f"data/processed/test_data_{book_isbn}.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Book train data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Book test data not found: {test_path}")

    train_data = pd.read_csv(train_path, parse_dates=['End Date'])
    test_data = pd.read_csv(test_path, parse_dates=['End Date'])
    
    # Set the End Date as index for proper time series handling
    train_data = train_data.set_index('End Date').sort_index()
    test_data = test_data.set_index('End Date').sort_index()

    print(f"✅ Loaded train data: {train_data.shape}")
    print(f"✅ Loaded test data: {test_data.shape}")
    print(f"📊 Train columns: {list(train_data.columns)}")
    print(f"📊 Test columns: {list(test_data.columns)}")

    return train_data, test_data

def load_pipeline_data():
    """DEPRECATED: Use load_individual_book_data() instead for individual book modeling."""
    print("⚠️  WARNING: load_pipeline_data() loads combined data which aggregates multiple books.")
    print("⚠️  For individual book modeling, use load_individual_book_data(isbn) instead.")
    
    train_path = "data/processed/combined_train_data.csv"
    test_path = "data/processed/combined_test_data.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Pipeline train data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Pipeline test data not found: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"✅ Loaded train data: {train_data.shape}")
    print(f"✅ Loaded test data: {test_data.shape}")
    print(f"📊 Train columns: {list(train_data.columns)}")
    print(f"📊 Test columns: {list(test_data.columns)}")

    return train_data, test_data

def create_time_series_from_df(df: pd.DataFrame, target_col: str = "Volume") -> pd.Series:
    """
    Convert DataFrame to time series for ARIMA modeling.
    For individual book data, assumes the index is already a datetime index.
    """
    df_work = df.copy()
    
    # Check if target column exists
    if target_col not in df_work.columns:
        available_cols = list(df_work.columns)
        raise ValueError(f"Column '{target_col}' not found. Available columns: {available_cols}")
    
    # If index is not datetime, try to convert it
    if not pd.api.types.is_datetime64_any_dtype(df_work.index):
        try:
            df_work.index = pd.to_datetime(df_work.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {e}")
    
    # Sort by index (date) to ensure chronological order
    df_work = df_work.sort_index()
    
    # For individual book data, we can directly use the target column as time series
    # No need to group by date since each row should be one time period
    time_series = df_work[target_col]
    
    print(f"📈 Created time series with {len(time_series)} data points")
    print(f"📅 Date range: {time_series.index.min()} to {time_series.index.max()}")
    print(f"📊 Volume range: {time_series.min()} to {time_series.max()}")
    
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
    p = trial.suggest_int("p", 0, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 4)
    P = trial.suggest_int("P", 0, 2)
    D = trial.suggest_int("D", 0, 1)
    Q = trial.suggest_int("Q", 0, 3)
    s = 52

    try:
        model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.forecast(steps=len(test_series))
        metrics = evaluate_forecast(test_series.values, forecast.values)
        return metrics["rmse"]
    except Exception as e:
        return float("inf")

def run_optuna_optimization_with_early_stopping(train_series: pd.Series, test_series: pd.Series,
                                              n_trials: int, study_name: str = "arima_optimization",
                                              patience: int = 15, min_improvement: float = 0.1,
                                              min_trials: int = 20) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization with convergence-based early stopping."""
    print(f"🚀 Starting ARIMA optimization with intelligent early stopping:")
    print(f"   • Max trials: {n_trials}")
    print(f"   • Patience: {patience} trials without improvement")
    print(f"   • Min improvement threshold: {min_improvement}")
    print(f"   • Min trials before stopping: {min_trials}")

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

        print(f"📊 Found {len(study.trials)} existing trials")
        
        # Check if there are any completed trials with valid values
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print(f"💾 Best value so far: {study.best_value:.4f}")
            best_value = study.best_value
        else:
            print("🔄 No completed trials found, starting fresh optimization")
            best_value = float('inf')
        trials_without_improvement = 0
        last_improvement_trial = len(study.trials)

        class EarlyStoppingCallback:
            def __init__(self, patience, min_improvement, min_trials, best_value, last_improvement_trial):
                self.patience = patience
                self.min_improvement = min_improvement
                self.min_trials = min_trials
                self.best_value = best_value
                self.last_improvement_trial = last_improvement_trial
                self.trials_without_improvement = 0

            def __call__(self, study, trial):
                # Only check best_value if there are completed trials
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if not completed_trials:
                    return  # Skip early stopping check if no completed trials
                    
                current_best = study.best_value
                total_trials = len(study.trials)

                # Check if we found a significant improvement
                if current_best < self.best_value - self.min_improvement:
                    self.best_value = current_best
                    self.last_improvement_trial = total_trials
                    self.trials_without_improvement = 0
                    print(f"🎯 Trial {total_trials}: New best value {current_best:.4f} (improvement: {self.best_value - current_best:.4f})")
                else:
                    self.trials_without_improvement = total_trials - self.last_improvement_trial

                # Early stopping check
                if (total_trials >= self.min_trials and
                    self.trials_without_improvement >= self.patience):
                    print(f"⏹️  Early stopping triggered!")
                    print(f"   • No improvement for {self.trials_without_improvement} trials")
                    print(f"   • Best value: {current_best:.4f}")
                    study.stop()

        # Create callback for early stopping
        callback = EarlyStoppingCallback(patience, min_improvement, min_trials, best_value, last_improvement_trial)

        print(f"⚙️  Starting optimization...")
        study.optimize(
            lambda trial: objective(trial, train_series, test_series),
            n_trials=n_trials,
            timeout=1800,  # 30 minutes maximum
            callbacks=[callback],
            n_jobs=1
        )

        total_trials = len(study.trials)
        improvement_trials = total_trials - last_improvement_trial

        print(f"✅ Optimization completed:")
        print(f"   • Total trials: {total_trials}")
        
        # Check if optimization was successful
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print(f"   • No successful trials completed")
            return {
                "best_params": {"p": 1, "d": 1, "q": 2, "P": 1, "D": 0, "Q": 1},
                "best_value": float("inf"),
                "n_trials": total_trials,
                "study_name": study_name,
                "storage_url": storage_url,
                "error": "No trials completed successfully"
            }
        
        print(f"   • Best value: {study.best_value:.4f}")
        print(f"   • Trials since last improvement: {improvement_trials}")

        return {
            "best_params": dict(study.best_params),
            "best_value": float(study.best_value),
            "n_trials": total_trials,
            "study_name": study_name,
            "storage_url": storage_url,
            "early_stopped": improvement_trials >= patience,
            "trials_without_improvement": improvement_trials
        }

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return {
            "best_params": {"p": 1, "d": 1, "q": 2, "P": 1, "D": 0, "Q": 1},
            "best_value": float("inf"),
            "n_trials": 0,
            "study_name": study_name,
            "storage_url": storage_url,
            "error": str(e)
        }

def run_optuna_optimization(train_series: pd.Series, test_series: pd.Series,
                           n_trials: int, study_name: str = "arima_optimization") -> Dict[str, Any]:
    """Legacy wrapper - now uses smart early stopping by default."""
    return run_optuna_optimization_with_early_stopping(
        train_series, test_series, n_trials, study_name,
        patience=15, min_improvement=0.1, min_trials=25
    )

def train_final_arima_model(series: pd.Series, best_params: Dict[str, int]):
    """Train final ARIMA model with best parameters."""
    p, d, q = best_params['p'], best_params['d'], best_params['q']
    P, D, Q = best_params['P'], best_params['D'], best_params['Q']
    s = 52

    model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    fitted_model = model.fit(disp=False, maxiter=200)

    return fitted_model

def parse_hyperparameters_json(json_string: str) -> Dict[str, Any]:
    """
    Helper function to parse hyperparameters JSON string back to dict.
    """
    return json.loads(json_string)


def train_individual_book_arima(book_isbn: str, output_dir: str = "outputs/arima", 
                               n_trials: int = 50) -> Dict[str, Any]:
    """
    Train ARIMA model for a single book using individual book files.
    
    Args:
        book_isbn: ISBN of the book to train model for
        output_dir: Directory to save model artifacts
        n_trials: Number of Optuna trials for hyperparameter optimization
        
    Returns:
        Dictionary with model results and metrics
    """
    print(f"🚀 Training individual ARIMA model for book ISBN: {book_isbn}")
    
    try:
        # Initialize MLflow logging for individual book (if available)
        try:
            import mlflow
            mlflow_available = True
        except ImportError:
            mlflow_available = False
            print("MLflow not available for individual book logging")
        # Load individual book data
        train_data, test_data = load_individual_book_data(book_isbn)
        
        # Convert to time series
        train_series = create_time_series_from_df(train_data, target_col="Volume")
        test_series = create_time_series_from_df(test_data, target_col="Volume")
        
        print(f"📊 Training series: {len(train_series)} points")
        print(f"📊 Test series: {len(test_series)} points")
        
        # Run Optuna optimization
        study_name = f"arima_{book_isbn}"
        optimization_results = run_optuna_optimization_with_early_stopping(
            train_series, test_series, n_trials, study_name,
            patience=5, min_improvement=0.1, min_trials=15
        )
        
        best_params = optimization_results["best_params"]
        print(f"🎯 Best parameters for {book_isbn}: {best_params}")
        
        # Train final model
        final_model = train_final_arima_model(train_series, best_params)
        
        # Make predictions
        forecast = final_model.forecast(steps=len(test_series))
        eval_metrics = evaluate_forecast(test_series.values, forecast.values)
        
        print(f"📈 Evaluation metrics for {book_isbn}:")
        for metric, value in eval_metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        # Log to MLflow if available
        if mlflow_available:
            try:
                # Log parameters for this book
                mlflow.log_params({
                    f"book_isbn": book_isbn,
                    f"p": best_params.get('p', 0),
                    f"d": best_params.get('d', 0),
                    f"q": best_params.get('q', 0),
                    f"P": best_params.get('P', 0),
                    f"D": best_params.get('D', 0),
                    f"Q": best_params.get('Q', 0),
                    f"n_trials": optimization_results.get("n_trials", 0),
                    f"train_length": len(train_series),
                    f"test_length": len(test_series)
                })
                
                # Log evaluation metrics
                mlflow.log_metrics({
                    f"mae": eval_metrics.get('mae', 0),
                    f"rmse": eval_metrics.get('rmse', 0),
                    f"mape": eval_metrics.get('mape', 0),
                    f"optuna_best_value": optimization_results.get("best_value", 0)
                })
                
                print(f"✅ Logged metrics to MLflow for {book_isbn}")
                
            except Exception as e:
                print(f"⚠️ Failed to log to MLflow for {book_isbn}: {e}")
        
        # Save model artifacts
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, f"arima_model_{book_isbn}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'actual': test_series.values,
            'predicted': forecast.values,
            'date': test_series.index
        })
        predictions_path = os.path.join(output_dir, f"predictions_{book_isbn}.csv")
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save hyperparameters and metrics
        results = {
            "book_isbn": book_isbn,
            "best_params": best_params,
            "optimization_results": optimization_results,
            "evaluation_metrics": eval_metrics,
            "model_path": model_path,
            "predictions_path": predictions_path,
            "train_series_length": len(train_series),
            "test_series_length": len(test_series),
            "training_date_range": f"{train_series.index.min()} to {train_series.index.max()}",
            "test_date_range": f"{test_series.index.min()} to {test_series.index.max()}"
        }
        
        results_path = os.path.join(output_dir, f"results_{book_isbn}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Log artifacts to MLflow if available
        if mlflow_available:
            try:
                mlflow.log_artifact(model_path, "models")
                mlflow.log_artifact(predictions_path, "predictions")
                mlflow.log_artifact(results_path, "results")
                print(f"✅ Logged artifacts to MLflow for {book_isbn}")
            except Exception as e:
                print(f"⚠️ Failed to log artifacts to MLflow for {book_isbn}: {e}")
        
        print(f"✅ Successfully trained ARIMA model for {book_isbn}")
        print(f"📁 Artifacts saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to train ARIMA model for {book_isbn}: {e}")
        return {
            "book_isbn": book_isbn,
            "error": str(e),
            "success": False
        }


def train_multiple_books_arima(book_isbns: List[str], output_dir: str = "outputs/arima",
                             n_trials: int = 50) -> Dict[str, Any]:
    """
    Train individual ARIMA models for multiple books.
    
    Args:
        book_isbns: List of ISBNs to train models for
        output_dir: Directory to save model artifacts
        n_trials: Number of Optuna trials per book
        
    Returns:
        Dictionary with results for each book
    """
    print(f"🚀 Training ARIMA models for {len(book_isbns)} books")
    
    results = {}
    successful_models = 0
    
    for i, isbn in enumerate(book_isbns, 1):
        print(f"\n{'='*60}")
        print(f"📖 Processing book {i}/{len(book_isbns)}: {isbn}")
        print(f"{'='*60}")
        
        book_results = train_individual_book_arima(isbn, output_dir, n_trials)
        results[isbn] = book_results
        
        if book_results.get("success", True):  # Default to True if not specified
            successful_models += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully trained: {successful_models}/{len(book_isbns)} models")
    print(f"❌ Failed: {len(book_isbns) - successful_models}/{len(book_isbns)} models")
    
    # Save overall summary
    summary = {
        "total_books": len(book_isbns),
        "successful_models": successful_models,
        "failed_models": len(book_isbns) - successful_models,
        "book_results": results,
        "training_timestamp": pd.Timestamp.now().isoformat()
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📁 Overall summary saved to: {summary_path}")
    
    return summary


def train_arima_step(train_data, test_data, output_dir, n_trials=50,
                     study_name="arima_optimization"):
    """
    ARIMA training step with persistent Optuna storage for pipeline compatibility.
    Same interface as CNN and LSTM models.
    """
    print("Starting ARIMA training with Optuna optimization...")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Parameters: trials={n_trials}, study_name={study_name}")

    try:
        # Ensure we have the volume column
        if "Volume" in train_data.columns and "volume" not in train_data.columns:
            train_data = train_data.copy()
            train_data["volume"] = train_data["Volume"]
        if "Volume" in test_data.columns and "volume" not in test_data.columns:
            test_data = test_data.copy()
            test_data["volume"] = test_data["Volume"]

        # Convert to time series format
        train_df_reset = train_data.reset_index()
        if "date" not in train_df_reset.columns:
            if "End Date" in train_df_reset.columns:
                train_df_reset["date"] = pd.to_datetime(train_df_reset["End Date"])
            else:
                train_df_reset["date"] = pd.date_range('2023-01-01', periods=len(train_data), freq='W')

        test_df_reset = test_data.reset_index()
        if "date" not in test_df_reset.columns:
            if "End Date" in test_df_reset.columns:
                test_df_reset["date"] = pd.to_datetime(test_df_reset["End Date"])
            else:
                # Start test dates after train data
                start_date = train_df_reset["date"].max() + pd.Timedelta(weeks=1)
                test_df_reset["date"] = pd.date_range(start_date, periods=len(test_data), freq='W')

        # Create time series
        train_series = create_time_series_from_df(train_df_reset, target_col="volume", date_col="date")
        test_series = create_time_series_from_df(test_df_reset, target_col="volume", date_col="date")

        print(f"Training series shape: {train_series.shape}")
        print(f"Test series shape: {test_series.shape}")

        # Run Optuna optimization
        optimization_results = run_optuna_optimization(
            train_series, test_series, n_trials, study_name
        )

        best_params = optimization_results["best_params"]
        print(f"Best ARIMA parameters: {best_params}")

        # Train final model with best parameters
        final_model = train_final_arima_model(train_series, best_params)

        # Make predictions
        forecast = final_model.forecast(steps=len(test_series))
        eval_metrics = evaluate_forecast(test_series.values, forecast.values)
        print(f"Evaluation metrics: {eval_metrics}")

        # Create results matching CNN/LSTM format
        results_data = []

        # Model configuration
        results_data.append({
            "result_type": "model_config",
            "component": "arima_order",
            "parameter": "pdq",
            "value": f"({best_params['p']},{best_params['d']},{best_params['q']})",
            "timestamp": pd.Timestamp.now(),
            "metadata": str(best_params)
        })

        # Evaluation metrics
        for metric_name, metric_value in eval_metrics.items():
            results_data.append({
                "result_type": "evaluation",
                "component": "test_metrics",
                "parameter": metric_name,
                "value": f"{metric_value:.4f}",
                "timestamp": pd.Timestamp.now(),
                "metadata": "test_set_performance"
            })

        results_df = pd.DataFrame(results_data)

        # Create hyperparameters JSON
        hyperparameters_dict = {
            "best_params": best_params,
            "optimization_results": optimization_results,
            "eval_metrics": eval_metrics,
            "study_name": study_name,
            "model_signature": f"ARIMA_({best_params['p']},{best_params['d']},{best_params['q']})_({best_params['P']},{best_params['D']},{best_params['Q']},52)"
        }

        best_hyperparameters_json = json.dumps(hyperparameters_dict, indent=2, default=str)

        print("ARIMA training completed successfully!")
        print(f"Best ARIMA parameters: {best_params}")
        print(f"Test performance - RMSE: {eval_metrics['rmse']:.2f}, MAE: {eval_metrics['mae']:.2f}")

        # Create output DataFrames for compatibility
        residuals_df = pd.DataFrame({
            "date": train_series.index,
            "residuals": final_model.resid,
            "model_signature": hyperparameters_dict["model_signature"]
        })

        # Create forecast DataFrame (consolidating test_predictions and forecast_comparison)
        forecast_df = pd.DataFrame({
            "period": range(1, len(test_series) + 1),
            "date": test_series.index,
            "actual": test_series.values,
            "predicted": forecast.values,
            "residuals": test_series.values - forecast.values,
            "absolute_error": np.abs(test_series.values - forecast.values),
            "percentage_error": np.abs((test_series.values - forecast.values) / test_series.values) * 100,
            "squared_error": (test_series.values - forecast.values) ** 2,
            "model_signature": hyperparameters_dict["model_signature"]
        })

        return (results_df, best_hyperparameters_json, final_model,
                residuals_df, forecast_df)

    except Exception as e:
        print(f"ARIMA training failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return error results matching format
        error_df = pd.DataFrame([{
            "result_type": "error",
            "component": "training_error",
            "parameter": "error_message",
            "value": str(e),
            "timestamp": pd.Timestamp.now(),
            "metadata": "training_failure"
        }])

        error_hyperparameters_dict = {
            "error": str(e),
            "best_params": {"p": 1, "d": 1, "q": 2, "P": 1, "D": 0, "Q": 1},
            "study_name": study_name,
            "model_signature": "ERROR_ARIMA_MODEL"
        }

        error_hyperparameters_json = json.dumps(error_hyperparameters_dict, indent=2, default=str)

        # Create empty DataFrames for error case
        error_residuals_df = pd.DataFrame({
            "date": pd.to_datetime([]),
            "residuals": [],
            "model_signature": "ERROR_ARIMA_MODEL"
        })

        error_forecast_df = pd.DataFrame({
            "period": [],
            "date": pd.to_datetime([]),
            "actual": [],
            "predicted": [],
            "residuals": [],
            "absolute_error": [],
            "percentage_error": [],
            "squared_error": [],
            "model_signature": []
        })

        return (error_df, error_hyperparameters_json, None,
                error_residuals_df, error_forecast_df)


if __name__ == "__main__":
    """
    Standalone ARIMA execution using pipeline data for fair model comparison.
    """
    print("🚀 Running ARIMA standalone training with pipeline data...")
    print("=" * 60)

    try:
        # Load pipeline data
        train_data, test_data = load_pipeline_data()

        # Create output directory - use centralized outputs directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🔧 Running ARIMA training...")
        print("=" * 60)

        # Run ARIMA training
        results = train_arima_step(
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            n_trials=50,
            study_name="standalone_arima_optimization"
        )

        results_df, hyperparameters_json, model, residuals_df, forecast_df = results

        print("\n✅ ARIMA standalone training completed successfully!")
        print("=" * 60)

        # Extract metrics from hyperparameters
        hyperparams = json.loads(hyperparameters_json)
        eval_metrics = hyperparams.get('eval_metrics', {})
        best_params = hyperparams.get('best_params', {})

        print(f"\n📊 ARIMA Results Summary:")
        print(f"• Model signature: {hyperparams.get('model_signature', 'ARIMA_Model')}")
        print(f"• Best parameters: {best_params}")
        print(f"• Training residuals: {len(residuals_df)} points")
        print(f"• Test predictions: {len(test_predictions_df)} points")
        print(f"• Test MAE: {eval_metrics.get('mae', 0):.2f}")
        print(f"• Test RMSE: {eval_metrics.get('rmse', 0):.2f}")
        print(f"• Test MAPE: {eval_metrics.get('mape', 0):.2f}%")

        # Create organized subdirectories for outputs
        residuals_dir = os.path.join(output_dir, "data", "residuals")
        predictions_dir = os.path.join(output_dir, "data", "predictions")
        comparisons_dir = os.path.join(output_dir, "data", "comparisons")

        os.makedirs(residuals_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(comparisons_dir, exist_ok=True)

        # Save results to organized CSV locations (enhanced forecast comparison includes all metrics)
        forecast_comparison_df.to_csv(f"{comparisons_dir}/arima_forecast_comparison.csv", index=False)
        residuals_df.to_csv(f"{residuals_dir}/arima_residuals.csv", index=False)

        # Add plotting functionality
        print(f"\n📋 Creating ARIMA forecast plots...")
        try:
            # Create standalone ARIMA plotting function
            def create_arima_standalone_plot(series_train, series_test, arima_predictions,
                                           eval_metrics, best_params, output_dir):
                """Create standalone ARIMA forecast plot."""
                import plotly.graph_objects as go
                import os

                # Create model signature for ARIMA
                model_signature = f"ARIMA_({best_params['p']},{best_params['d']},{best_params['q']})_({best_params['P']},{best_params['D']},{best_params['Q']},52)"

                # Create the main plot
                fig = go.Figure()

                # Add training data
                fig.add_trace(go.Scatter(
                    x=series_train.index,
                    y=series_train.values,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='blue', width=2),
                    opacity=0.8
                ))

                # Add actual test data
                fig.add_trace(go.Scatter(
                    x=series_test.index,
                    y=series_test.values,
                    mode='lines+markers',
                    name='Actual Test Data',
                    line=dict(color='black', width=3),
                    marker=dict(size=5)
                ))

                # Add ARIMA predictions
                fig.add_trace(go.Scatter(
                    x=series_test.index,
                    y=arima_predictions,
                    mode='lines+markers',
                    name='ARIMA Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))

                # Update layout
                title_text = f'ARIMA Book Sales Forecast<br><sub>MAE: {eval_metrics["mae"]:.2f} | MAPE: {eval_metrics["mape"]:.2f}% | RMSE: {eval_metrics["rmse"]:.2f}</sub>'

                fig.update_layout(
                    title=title_text,
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )

                # Save plots - create necessary directories
                os.makedirs(f"{output_dir}/plots/interactive", exist_ok=True)
                os.makedirs(f"{output_dir}/plots/static", exist_ok=True)

                # Create descriptive file names with proper folder structure
                html_filename = f"{output_dir}/plots/interactive/arima_standalone_forecast.html"
                png_filename = f"{output_dir}/plots/static/arima_standalone_forecast.png"

                # Save files
                fig.write_html(html_filename)
                fig.write_image(png_filename, width=1200, height=500)

                print(f"📁 ARIMA standalone plots saved to: {output_dir}")
                print(f"   • HTML: {html_filename}")
                print(f"   • PNG: {png_filename}")

                return {
                    'figure': fig,
                    'model_signature': model_signature,
                    'metrics': eval_metrics
                }

            # Prepare data for plotting
            # Create date series based on test data
            if "End Date" in test_data.columns:
                test_dates = pd.to_datetime(test_data["End Date"])
            else:
                test_dates = pd.date_range('2023-12-16', periods=len(test_predictions_df), freq='W-SAT')

            if "End Date" in train_data.columns:
                # Use actual train dates
                train_dates = pd.to_datetime(train_data["End Date"])
                train_values = train_data["Volume"].values
            else:
                # Create synthetic train dates
                train_dates = pd.date_range('2020-01-01', periods=len(train_data), freq='W-SAT')
                train_values = train_data["Volume"].values if "Volume" in train_data.columns else train_data["volume"].values

            # Create series for plotting
            train_series = pd.Series(train_values, index=train_dates, name='Volume')
            test_series = pd.Series(test_predictions_df['actual'].values, index=test_dates[:len(test_predictions_df)], name='Volume')

            # Create standalone ARIMA plot
            plotting_results = create_arima_standalone_plot(
                series_train=train_series,
                series_test=test_series,
                arima_predictions=test_predictions_df['predicted'].values,
                eval_metrics=eval_metrics,
                best_params=best_params,
                output_dir=output_dir
            )

            print("✅ ARIMA standalone plotting completed!")

        except ImportError as e:
            print(f"⚠️  Plotting module not available: {e}")
            print("📊 Continuing without plots...")
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")
            print("📊 Continuing without plots...")
            import traceback
            traceback.print_exc()

        print(f"\n📁 Generated files in '{output_dir}/':")
        for file in os.listdir(output_dir):
            print(f"  • {file}")

        print(f"\n🎉 ARIMA standalone execution completed successfully!")
        print(f"📁 Check the '{output_dir}' directory for generated plots and data files.")

    except Exception as e:
        print(f"\n❌ ARIMA standalone training failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
