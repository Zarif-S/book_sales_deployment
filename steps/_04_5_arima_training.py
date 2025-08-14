"""
ARIMA Training Step

This module contains ARIMA model training functionality, including
individual model training, MLflow logging, and model management.
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
import time
from functools import wraps
import mlflow
import mlflow.statsmodels
from typing import Dict, List, Any, Annotated
from zenml import step
from zenml.steps import get_step_context
from zenml.logger import get_logger
from zenml import ArtifactConfig
from config.arima_training_config import ARIMATrainingConfig, get_arima_config
from utils.model_reuse import create_retraining_engine
from utils.zenml_helpers import _add_step_metadata, create_step_metadata

# Initialize logger
logger = get_logger(__name__)


def retry_mlflow_operation(max_retries=3, delay=1):
    """Decorator to retry MLflow operations on connection errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionResetError, ConnectionError, Exception) as e:
                    if attempt < max_retries - 1 and ("connection" in str(e).lower() or "remote" in str(e).lower()):
                        logger.warning(f"MLflow connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        if attempt == max_retries - 1:
                            logger.error(f"MLflow operation failed after {max_retries} attempts: {e}")
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cleanup_old_mlflow_models(max_models_per_book: int = 2) -> None:
    """
    Cleanup old MLflow model artifacts to prevent disk space issues.
    Keeps only the most recent `max_models_per_book` model.statsmodels files per book.
    """
    try:
        # Find all model.statsmodels files and extract book information
        model_files_by_book = {}

        for root, dirs, files in os.walk("mlruns"):
            for file in files:
                if file == "model.statsmodels":
                    full_path = os.path.join(root, file)
                    stat_info = os.stat(full_path)

                    # Extract book ISBN from run name if available
                    run_dir = full_path.split('/artifacts/')[0]
                    book_isbn = None

                    try:
                        # Try to get run name from tags
                        run_name_file = os.path.join(run_dir, 'tags', 'mlflow.runName')
                        if os.path.exists(run_name_file):
                            with open(run_name_file, 'r') as f:
                                run_name = f.read().strip()
                                # Extract ISBN from run name (format: book_9780123456789_Title_timestamp)
                                if 'book_' in run_name:
                                    parts = run_name.split('_')
                                    for i, part in enumerate(parts):
                                        if part == 'book' and i + 1 < len(parts):
                                            potential_isbn = parts[i + 1]
                                            if len(potential_isbn) == 13 and potential_isbn.isdigit():
                                                book_isbn = potential_isbn
                                                break
                    except Exception:
                        # If we can't extract from run name, skip this model
                        continue

                    if book_isbn:
                        if book_isbn not in model_files_by_book:
                            model_files_by_book[book_isbn] = []
                        model_files_by_book[book_isbn].append((full_path, stat_info.st_mtime, stat_info.st_size, run_dir))

        if not model_files_by_book:
            logger.info("✅ Model cleanup: No models found with identifiable book ISBNs")
            return

        total_removed = 0
        total_size_removed = 0

        # Process each book separately
        for book_isbn, book_models in model_files_by_book.items():
            # Sort by modification time (newest first)
            book_models.sort(key=lambda x: x[1], reverse=True)

            models_to_keep = len(book_models)
            models_to_remove = max(0, len(book_models) - max_models_per_book)

            logger.info(f"📚 Book {book_isbn}: Found {len(book_models)} models, keeping {min(len(book_models), max_models_per_book)}")

            if models_to_remove > 0:
                # Remove old models for this book (keep the newest max_models_per_book)
                for file_path, mod_time, size, run_dir in book_models[max_models_per_book:]:
                    try:
                        if os.path.exists(run_dir):
                            import shutil
                            shutil.rmtree(run_dir)
                            total_removed += 1
                            total_size_removed += size
                            logger.info(f"🗑️  Removed old model for {book_isbn}: {os.path.basename(run_dir)} ({size/(1024*1024):.1f}MB)")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to remove {file_path}: {e}")

        if total_removed > 0:
            size_mb = total_size_removed / (1024 * 1024)
            logger.info(f"✅ Model cleanup completed: Removed {total_removed} old model runs ({size_mb:.1f}MB total)")
        else:
            logger.info(f"✅ Model cleanup: All models within per-book limit of {max_models_per_book}")

    except Exception as e:
        logger.warning(f"⚠️  Model cleanup failed: {e}")


def train_models_from_consolidated_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    book_isbns: List[str],
    output_dir: str,
    config: ARIMATrainingConfig = None
) -> Dict[str, Any]:
    """
    Train individual ARIMA models for each book using consolidated DataFrames with smart retraining.
    This replaces the CSV file-based approach for Vertex AI deployment.

    Enhanced with:
    - Configuration-driven optimization parameters
    - Smart model reuse to avoid unnecessary retraining
    - Performance-based retraining triggers
    - Comprehensive logging and monitoring

    Note: Deprecated n_trials parameter has been removed. Use config.n_trials instead.
    """
    from steps._04_arima_standalone import (
        create_time_series_from_df,
        run_optuna_optimization_with_early_stopping,
        train_final_arima_model,
        evaluate_forecast
    )
    from steps._03_5_modelling_prep import filter_book_data

    # Initialize configuration if not provided
    if config is None:
        config = get_arima_config()
        logger.info(f"Using default configuration for environment: {config.environment}")

    # Log configuration
    config.log_configuration(logger)

    # Initialize model retraining decision engine
    retraining_engine = create_retraining_engine(config, output_dir)

    logger.info(f"🚀 Training models for {len(book_isbns)} books using consolidated artifacts")
    logger.info(f"📊 Configuration: {config.environment} mode, {config.n_trials} trials, force_retrain={config.force_retrain}")

    total_books = len(book_isbns)
    successful_models = 0
    failed_models = 0
    reused_models = 0
    book_results = {}

    for book_isbn in book_isbns:
        logger.info(f"📚 Processing ISBN: {book_isbn}")

        # Check if retraining is needed using smart decision engine
        should_retrain, reason, existing_model = retraining_engine.should_retrain_model(
            book_isbn, train_data, test_data
        )

        # Log retraining decision
        retraining_engine.log_retraining_decision(book_isbn, should_retrain, reason, existing_model)

        if not should_retrain and existing_model:
            # Reuse existing model
            logger.info(f"♻️  Reusing existing model for {book_isbn}: {reason}")
            book_results[book_isbn] = retraining_engine.load_existing_model_results(existing_model)
            successful_models += 1
            reused_models += 1
            continue

        # Proceed with training new model
        logger.info(f"🔄 Training new model for {book_isbn}: {reason}")

        # Reset evaluation_metrics for each book to avoid reusing metrics from previous books
        evaluation_metrics = None

        try:
            # Filter consolidated data by ISBN and prepare for modeling
            book_train_clean = filter_book_data(train_data, book_isbn, clean_for_modeling=True)
            book_test_clean = filter_book_data(test_data, book_isbn, clean_for_modeling=True)

            if book_train_clean.empty or book_test_clean.empty:
                logger.error(f"No data found for ISBN {book_isbn} in consolidated artifacts")
                book_results[book_isbn] = {"error": f"No data found for ISBN {book_isbn}"}
                failed_models += 1
                continue

            logger.info(f"Filtered data for {book_isbn}: train {book_train_clean.shape}, test {book_test_clean.shape}")
            logger.info(f"Train index datetime: {pd.api.types.is_datetime64_any_dtype(book_train_clean.index)}")
            logger.info(f"Test index datetime: {pd.api.types.is_datetime64_any_dtype(book_test_clean.index)}")

            # Convert to time series
            train_series = create_time_series_from_df(book_train_clean, target_col="Volume")
            test_series = create_time_series_from_df(book_test_clean, target_col="Volume")

            # Get book-specific parameter seeds based on domain expertise
            def get_seed_parameters_for_book(book_isbn):
                """Get initial parameter suggestions based on book-specific domain knowledge"""

                # Default seeds for most books
                default_seeds = [
                    {'p': 1, 'd': 0, 'q': 0, 'P': 1, 'D': 0, 'Q': 0},  # SARIMAX(1, 0, 0)x(1, 0, 0, 52)
                    {'p': 1, 'd': 0, 'q': 0, 'P': 2, 'D': 0, 'Q': 1},  # SARIMAX(1, 0, 0)x(2, 0, 1, 52)
                ]

                # Book-specific overrides based on domain expertise
                book_specific_seeds = {
                    '9780722532935': [  # Alchemist - your known good params
                        {'p': 1, 'd': 0, 'q': 0, 'P': 2, 'D': 0, 'Q': 1},  # Best performing
                        {'p': 1, 'd': 0, 'q': 0, 'P': 1, 'D': 0, 'Q': 0},  # Alternative
                    ],
                    '9780241003008': [  # Very Hungry Caterpillar - optimized params for better performance
                        {'p': 2, 'd': 1, 'q': 1, 'P': 1, 'D': 0, 'Q': 1},  # More complex pattern for seasonal data
                        {'p': 1, 'd': 1, 'q': 1, 'P': 2, 'D': 0, 'Q': 2},  # Alternative with more seasonality
                        {'p': 0, 'd': 2, 'q': 3, 'P': 2, 'D': 0, 'Q': 3},  # Previous best from your results
                    ],
                }

                return book_specific_seeds.get(book_isbn, default_seeds)

            # Get book-specific suggested parameters
            suggested_params = get_seed_parameters_for_book(book_isbn)
            logger.info(f"Testing {len(suggested_params)} seed parameters for {book_isbn}")

            suggested_results = []
            for i, params in enumerate(suggested_params):
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    model = SARIMAX(train_series, order=(params['p'], params['d'], params['q']),
                                   seasonal_order=(params['P'], params['D'], params['Q'], 52))
                    fitted = model.fit(disp=False, maxiter=200)
                    forecast = fitted.forecast(steps=len(test_series))
                    from steps._04_arima_standalone import evaluate_forecast
                    metrics = evaluate_forecast(test_series.values, forecast.values)
                    suggested_results.append((params, metrics['rmse'], metrics))
                    logger.info(f"  Suggested params {i+1}: {params} → RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.2f}%")
                except Exception as e:
                    logger.warning(f"  Suggested params {i+1}: {params} → Failed: {e}")

            # Optimize hyperparameters with Optuna using configuration-driven parameters
            logger.info(f"Starting Optuna optimization for {book_isbn} (config-driven)")

            # Use timestamp for unique study names to avoid database corruption
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            book_study_name = f"arima_optimization_{book_isbn}_{timestamp}"

            # Seed Optuna with our good parameters first
            import optuna
            from steps._04_arima_standalone import objective

            # Get storage configuration from config
            storage_config = config.get_optuna_storage_config()

            if storage_config['storage_type'] == 'memory':
                # Production: Use in-memory storage for reliability and speed
                logger.info(f"Using in-memory Optuna storage ({config.environment} mode)")
                study = optuna.create_study(
                    study_name=book_study_name,
                    direction="minimize"
                )
            else:
                # Development/testing: Use SQLite storage for persistence and debugging
                logger.info(f"Using SQLite Optuna storage ({config.environment} mode)")
                storage_dir = storage_config['storage_dir']
                storage_url = f"sqlite:///{os.path.join(storage_dir, f'{book_study_name}.db')}"

                study = optuna.create_study(
                    study_name=book_study_name,
                    storage=storage_url,
                    load_if_exists=storage_config['load_if_exists'],
                    direction="minimize"
                )

            # Check if this is a new study (no trials yet)
            if len(study.trials) == 0:
                logger.info(f"New study - seeding with {len(suggested_params)} parameter combinations")
                for params in suggested_params:
                    study.enqueue_trial(params)
            else:
                logger.info(f"Existing study with {len(study.trials)} trials - continuing optimization")

            # Run the optimization with robust error handling
            from steps._04_arima_standalone import run_optuna_optimization_with_early_stopping

            optuna_success = False
            optuna_best_params = None
            optuna_best_rmse = float('inf')

            try:
                logger.info(f"Starting Optuna optimization for {book_isbn} with config parameters")
                logger.info(f"  Trials: {config.n_trials}, Patience: {config.patience}, Min improvement: {config.min_improvement}")
                optimization_results = run_optuna_optimization_with_early_stopping(
                    train_series, test_series, config.n_trials, book_study_name,
                    patience=config.patience, min_improvement=config.min_improvement, min_trials=config.min_trials
                )

                # Validate optimization results
                if (optimization_results and
                    "best_params" in optimization_results and
                    "best_value" in optimization_results and
                    optimization_results["best_value"] != float('inf')):

                    optuna_best_params = optimization_results["best_params"]
                    optuna_best_rmse = optimization_results["best_value"]
                    optuna_success = True
                    logger.info(f"✅ Optuna optimization succeeded for {book_isbn}")
                else:
                    logger.warning(f"⚠️ Optuna returned invalid results for {book_isbn}")

            except Exception as e:
                logger.warning(f"⚠️ Optuna optimization failed for {book_isbn}: {str(e)}")
                logger.info(f"Falling back to suggested parameters for {book_isbn}")

            if not optuna_success:
                logger.info(f"Using fallback optimization strategy for {book_isbn}")

            # Compare suggested params vs Optuna results
            if suggested_results:
                best_suggested = min(suggested_results, key=lambda x: x[1])
                best_suggested_params, best_suggested_rmse, best_suggested_metrics = best_suggested

                logger.info(f"Best suggested params: {best_suggested_params} → RMSE: {best_suggested_rmse:.4f}")

                if optuna_success and optuna_best_params:
                    logger.info(f"Best Optuna params: {optuna_best_params} → RMSE: {optuna_best_rmse:.4f}")

                    if best_suggested_rmse < optuna_best_rmse:
                        logger.info(f"✅ Using suggested parameters (better RMSE: {best_suggested_rmse:.4f} vs {optuna_best_rmse:.4f})")
                        best_params = best_suggested_params
                        evaluation_metrics = best_suggested_metrics
                    else:
                        logger.info(f"✅ Using Optuna parameters (better RMSE: {optuna_best_rmse:.4f} vs {best_suggested_rmse:.4f})")
                        best_params = optuna_best_params
                        # Will evaluate later
                else:
                    logger.info(f"✅ Using suggested parameters (Optuna failed, RMSE: {best_suggested_rmse:.4f})")
                    best_params = best_suggested_params
                    evaluation_metrics = best_suggested_metrics
            else:
                if optuna_success and optuna_best_params:
                    logger.info(f"✅ Using Optuna parameters (suggested params failed, RMSE: {optuna_best_rmse:.4f})")
                    best_params = optuna_best_params
                else:
                    # Fallback to safe default parameters
                    logger.warning(f"⚠️ Both suggested and Optuna params failed for {book_isbn}, using safe defaults")
                    best_params = {'p': 1, 'd': 0, 'q': 0, 'P': 1, 'D': 0, 'Q': 0}

            # Validate final parameters
            required_keys = ['p', 'd', 'q', 'P', 'D', 'Q']
            if not all(key in best_params for key in required_keys):
                logger.warning(f"⚠️ Invalid parameters for {book_isbn}, using safe defaults")
                best_params = {'p': 1, 'd': 0, 'q': 0, 'P': 1, 'D': 0, 'Q': 0}

            # Train final model
            logger.info(f"Training final model for {book_isbn} with params: {best_params}")
            final_model = train_final_arima_model(train_series, best_params)

            # Evaluate model (if not already done for suggested params)
            if evaluation_metrics is None:
                train_predictions = final_model.fittedvalues
                test_predictions = final_model.forecast(steps=len(test_series))
                evaluation_metrics = evaluate_forecast(test_series.values, test_predictions.values)

            # Save model using MLflow format (production-ready)
            book_output_dir = os.path.join(output_dir, f'book_{book_isbn}')
            os.makedirs(book_output_dir, exist_ok=True)

            # Generate timestamp for unique model paths
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save model with MLflow (with timestamp to avoid conflicts)
            model_path = os.path.join(book_output_dir, f'arima_model_{book_isbn}_{timestamp}')

            # Create model signature and input example for MLflow
            try:
                import mlflow.models.signature as signature
                from mlflow.types.schema import Schema, ColSpec

                # Create input/output signature for SARIMA model
                input_schema = Schema([ColSpec("double", "steps")])
                output_schema = Schema([ColSpec("double", "forecast")])
                model_signature = signature.ModelSignature(inputs=input_schema, outputs=output_schema)

                # Save additional context for model loading
                model_info = {
                    "isbn": book_isbn,
                    "train_start": str(train_series.index.min()),
                    "train_end": str(train_series.index.max()),
                    "train_freq": str(train_series.index.freq),
                    "model_params": best_params,
                    "training_data_length": len(train_series)
                }

                # Save with MLflow
                mlflow.statsmodels.save_model(
                    statsmodels_model=final_model,
                    path=model_path,
                    signature=model_signature,
                    input_example=pd.DataFrame({"steps": [len(test_series)]}),
                    metadata=model_info
                )

                logger.info(f"✅ Saved MLflow model to: {model_path}")

                logger.info(f"💾 Model saved and ready for MLflow registration via run-based approach")

                # Clean up old models to prevent disk space issues
                cleanup_old_mlflow_models(max_models_per_book=2)

            except Exception as e:
                logger.warning(f"MLflow save failed, falling back to pickle: {e}")
                # Fallback to pickle if MLflow fails (with timestamp)
                model_path_pkl = os.path.join(book_output_dir, f'arima_model_{book_isbn}_{timestamp}.pkl')
                with open(model_path_pkl, 'wb') as f:
                    pickle.dump(final_model, f)
                model_path = model_path_pkl
                logger.info(f"📁 Saved pickle model to: {model_path_pkl}")

            # Save results
            results_path = os.path.join(book_output_dir, f'results_{book_isbn}.json')
            results_data = {
                'isbn': book_isbn,
                'best_params': best_params,
                'evaluation_metrics': evaluation_metrics,
                'optimization_results': optimization_results,  # Store the full Optuna results
                'model_path': model_path,
                'train_shape': book_train_clean.shape,
                'test_shape': book_test_clean.shape,
                'train_series_length': len(train_series),  # Store actual series lengths
                'test_series_length': len(test_series)
            }

            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            book_results[book_isbn] = results_data
            successful_models += 1

            # Register the newly trained model in the retraining engine
            try:
                data_hash = retraining_engine.calculate_data_hash(train_data, test_data, book_isbn)
                retraining_engine.register_model(
                    isbn=book_isbn,
                    model_path=model_path,
                    evaluation_metrics=evaluation_metrics,
                    model_params=best_params,
                    data_hash=data_hash,
                    train_length=len(train_series),
                    test_length=len(test_series),
                    mlflow_run_id=None,  # Will be set later in MLflow runs
                    mlflow_model_version=None,  # Will be set later in MLflow runs
                    metadata={'optimization_results': optimization_results, 'config_environment': config.environment}
                )
                logger.info(f"📝 Registered new model for {book_isbn} in retraining registry")
            except Exception as reg_error:
                logger.warning(f"⚠️ Failed to register model for {book_isbn} in retraining registry: {reg_error}")

            logger.info(f"✅ Model training completed for {book_isbn}")
            logger.info(f"   MAE: {evaluation_metrics.get('mae', 0):.4f}")
            logger.info(f"   RMSE: {evaluation_metrics.get('rmse', 0):.4f}")
            logger.info(f"   MAPE: {evaluation_metrics.get('mape', 0):.2f}%")

        except Exception as e:
            logger.error(f"❌ Training failed for {book_isbn}: {str(e)}")
            book_results[book_isbn] = {"error": str(e)}
            failed_models += 1

        # Create individual sequential MLflow run for each book (avoids ZenML conflicts)
        if book_isbn in book_results and "error" not in book_results[book_isbn]:
            try:
                # Get book title from train data
                book_title_rows = train_data[train_data['ISBN'] == book_isbn]['Title']
                book_title = book_title_rows.iloc[0] if not book_title_rows.empty else f"Book_{book_isbn}"

                logger.info(f"📖 Scheduling individual MLflow run for {book_isbn} post-pipeline")

                # Store book run data for later processing (after main pipeline run completes)
                if not hasattr(train_models_from_consolidated_data, '_book_run_data'):
                    train_models_from_consolidated_data._book_run_data = []

                book_run_data = {
                    'book_isbn': book_isbn,
                    'book_title': book_title,
                    'best_params': book_results[book_isbn]['best_params'],
                    'evaluation_metrics': book_results[book_isbn]['evaluation_metrics'],
                    'optimization_results': book_results[book_isbn].get('optimization_results', {}),
                    'train_series_length': book_results[book_isbn].get('train_series_length', 0),
                    'test_series_length': book_results[book_isbn].get('test_series_length', 0),
                    'model_path': book_results[book_isbn].get('model_path', '')
                }

                train_models_from_consolidated_data._book_run_data.append(book_run_data)
                logger.info(f"📊 Stored run data for {book_isbn} - will create individual run after pipeline")

            except Exception as storage_error:
                logger.warning(f"⚠️ Failed to store run data for {book_isbn}: {storage_error}")
                logger.info(f"📊 Model training was successful, individual run creation is optional")

    # Get retraining statistics for monitoring
    retraining_stats = retraining_engine.get_retraining_stats()

    # Log final summary with optimization results
    logger.info(f"🎉 Training pipeline completed!")
    logger.info(f"   Total books: {total_books}")
    logger.info(f"   Newly trained: {successful_models - reused_models}")
    logger.info(f"   Reused models: {reused_models}")
    logger.info(f"   Failed: {failed_models}")
    logger.info(f"   Configuration: {config.environment} mode")

    if retraining_stats['total_decisions'] > 0:
        logger.info(f"   Retraining efficiency: {reused_models}/{total_books} models reused ({reused_models/total_books*100:.1f}%)")

    # Return enhanced results with optimization information
    return {
        'total_books': total_books,
        'successful_models': successful_models,
        'failed_models': failed_models,
        'reused_models': reused_models,
        'newly_trained_models': successful_models - reused_models,
        'book_results': book_results,
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'configuration': config.to_dict(),
        'retraining_stats': retraining_stats,
        'optimization_efficiency': {
            'reuse_rate': reused_models / total_books if total_books > 0 else 0,
            'success_rate': successful_models / total_books if total_books > 0 else 0,
            'avg_trials_per_book': config.n_trials,
            'early_stopping_enabled': True,
            'patience': config.patience,
            'min_improvement': config.min_improvement
        }
    }


@step(
    enable_cache=False,  # Disable cache to see fresh training run
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def train_individual_arima_models_step(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    selected_isbns: List[str],
    output_dir: str,
    config: ARIMATrainingConfig = None
) -> Annotated[Dict[str, Any], ArtifactConfig(name="arima_training_results")]:
    """
    Train individual SARIMA models for each selected book using consolidated artifacts with smart retraining.

    Enhanced with configuration-driven optimization and model reuse logic.
    Note: Deprecated n_trials parameter has been removed. Use config parameter instead.
    """
    import mlflow

    logger.info(f"Starting individual ARIMA training for {len(selected_isbns)} books")
    logger.info(f"Using consolidated artifacts: train_data shape {train_data.shape}, test_data shape {test_data.shape}")

    # Configure remote MLflow tracking server with error handling
    try:
        mlflow_tracking_uri = "https://mlflow-tracking-server-1076639696283.europe-west2.run.app"
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Set MLflow experiment name for this pipeline run
        experiment_name = "book_sales_arima_modeling_v2"
        mlflow.set_experiment(experiment_name)
        logger.info(f"🧪 MLflow configured with remote server: {mlflow_tracking_uri}")
        logger.info(f"🧪 MLflow experiment set to: {experiment_name}")

    except Exception as mlflow_error:
        logger.info("Continuing without MLflow tracking")

    try:
        # Create ARIMA output directory - use more robust path handling for containers
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            arima_output_dir = os.path.join(project_root, 'outputs', 'models', 'arima')
        except Exception as path_error:
            logger.warning(f"Path calculation failed: {path_error}, using fallback")
            # Fallback for container environments
            arima_output_dir = os.path.join(output_dir, 'arima_models')

        os.makedirs(arima_output_dir, exist_ok=True)
        logger.info(f"✅ Created output directory: {arima_output_dir}")

        logger.info(f"Training ARIMA models for ISBNs: {selected_isbns}")
        logger.info(f"Output directory: {arima_output_dir}")

        # Log enhanced pipeline-level parameters using ZenML's experiment tracker
        try:
            mlflow.log_params({
                "pipeline_type": "arima_training_optimized",
                "total_books": len(selected_isbns),
                "n_trials": config.n_trials if config else 10,
                "books": ",".join(selected_isbns),
                "output_directory": arima_output_dir,
                "config_environment": config.environment if config else "unknown",
                "config_force_retrain": config.force_retrain if config else True,
                "config_patience": config.patience if config else 3,
                "config_min_improvement": config.min_improvement if config else 0.5,
                "smart_retraining_enabled": config is not None
            })
            logger.info("✅ Logged pipeline parameters to MLflow")
        except Exception as param_error:
            logger.warning(f"⚠️ Failed to log MLflow parameters: {param_error}")
            logger.info("Continuing without parameter logging")

        # Initialize configuration if not provided
        if config is None:
            config = get_arima_config()
            logger.info(f"Using default configuration for step: {config.environment}")

        # Train individual models using consolidated artifacts with smart retraining
        training_results = train_models_from_consolidated_data(
            train_data=train_data,
            test_data=test_data,
            book_isbns=selected_isbns,
            output_dir=arima_output_dir,
            config=config
        )

        # Log enhanced pipeline-level summary metrics using ZenML's experiment tracker
        pipeline_success_rate = (training_results.get('successful_models', 0) /
                               training_results.get('total_books', 1) * 100)
        reuse_rate = training_results.get('optimization_efficiency', {}).get('reuse_rate', 0) * 100

        mlflow.log_metrics({
            "pipeline_success_rate": pipeline_success_rate,
            "total_books": training_results.get('total_books', 0),
            "successful_models": training_results.get('successful_models', 0),
            "failed_models": training_results.get('failed_models', 0),
            "reused_models": training_results.get('reused_models', 0),
            "newly_trained_models": training_results.get('newly_trained_models', 0),
            "model_reuse_rate_percent": reuse_rate,
            "config_n_trials": config.n_trials,
            "config_patience": config.patience
        })

        # Add enhanced pipeline-level tags to distinguish parent run from individual book runs
        mlflow.set_tags({
            "run_type": "pipeline_summary",
            "architecture": "hybrid_tracking_optimized",
            "individual_runs_created": "true",
            "books_processed": ",".join(selected_isbns),
            "scalable_approach": "parent_child_runs",
            "smart_retraining": "enabled" if config else "disabled",
            "config_environment": config.environment if config else "unknown",
            "optimization_version": "v2"
        })

        logger.info(f"📊 Logged pipeline summary to MLflow: {pipeline_success_rate:.1f}% success rate")

        # Create individual MLflow runs for each book (post-pipeline to avoid ZenML conflicts)
        if hasattr(train_models_from_consolidated_data, '_book_run_data'):
            logger.info(f"🔄 Creating {len(train_models_from_consolidated_data._book_run_data)} individual book runs...")

            # Keep parent run active for proper nested run creation
            logger.info("🔗 Creating nested runs under parent pipeline run")

            for book_data in train_models_from_consolidated_data._book_run_data:
                try:
                    import time
                    clean_title = book_data['book_title'].replace(' ', '_').replace(',', '').replace("'", '').replace('.', '')
                    book_run_name = f"book_{book_data['book_isbn']}_{clean_title[:15]}_{time.strftime('%H%M%S')}"

                    # Create individual nested run under parent pipeline run
                    with mlflow.start_run(run_name=book_run_name, nested=True) as book_run:
                        logger.info(f"📖 Created individual run for {book_data['book_isbn']}: {book_run.info.run_id}")

                        try:
                            # Log book-specific parameters (ensure no duplicates)
                            params_to_log = {
                                "isbn": book_data['book_isbn'],
                                "title": book_data['book_title'],
                                "model_type": "SARIMA",
                                "arima_p": book_data['best_params']['p'],  # Use prefixes to avoid conflicts
                                "arima_d": book_data['best_params']['d'],
                                "arima_q": book_data['best_params']['q'],
                                "seasonal_P": book_data['best_params']['P'],
                                "seasonal_D": book_data['best_params']['D'],
                                "seasonal_Q": book_data['best_params']['Q'],
                                "seasonal_period": 52,
                                "optimization_method": "optuna",
                                "train_length": book_data['train_series_length'],
                                "test_length": book_data['test_series_length']
                            }
                            mlflow.log_params(params_to_log)
                        except Exception as param_error:
                            logger.warning(f"⚠️ Failed to log parameters for {book_data['book_isbn']}: {param_error}")

                        # Log evaluation metrics
                        mlflow.log_metrics({
                            "mae": book_data['evaluation_metrics'].get('mae', 0),
                            "rmse": book_data['evaluation_metrics'].get('rmse', 0),
                            "mape": book_data['evaluation_metrics'].get('mape', 0),
                            "optuna_best_value": book_data['optimization_results'].get('best_value', 0),
                            "optuna_trials": book_data['optimization_results'].get('n_trials', 0)
                        })

                        # Add tags for easy filtering
                        mlflow.set_tags({
                            "run_type": "individual_book",
                            "isbn": book_data['book_isbn'],
                            "model_architecture": "SARIMA",
                            "optimization_engine": "optuna",
                            "created_by": "post_pipeline_sequential"
                        })

                        # Register the model to this individual run for production deployment
                        try:
                            if 'model_path' in book_data and book_data.get('model_path'):
                                model_path = book_data['model_path']
                                model_name = f"arima_book_{book_data['book_isbn']}"

                                # Log the model as an artifact in this run first, then register
                                import mlflow.statsmodels

                                # Load the saved model and log it to this individual run
                                try:
                                    saved_model = mlflow.statsmodels.load_model(model_path)

                                    # Log the model to this run without problematic input examples
                                    # Time series models work better without custom input examples
                                    logged_model = mlflow.statsmodels.log_model(
                                        statsmodels_model=saved_model,
                                        artifact_path="model"
                                    )

                                    # Use the run-based URI for registration (this creates the proper link)
                                    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

                                except Exception as load_error:
                                    logger.warning(f"Could not load model for logging: {load_error}")
                                    # Fallback to file-based registration
                                    model_uri = f"file://{os.path.abspath(model_path)}"

                                # Register model with retry logic - now it will be properly linked to this run
                                @retry_mlflow_operation(max_retries=3, delay=2)
                                def register_model_with_retry():
                                    return mlflow.register_model(
                                        model_uri=model_uri,
                                        name=model_name,
                                        tags={
                                            "book_isbn": book_data['book_isbn'],
                                            "book_title": book_data['book_title'],
                                            "run_type": "individual_book",
                                            "model_type": "SARIMA"
                                        }
                                    )

                                registered_model = register_model_with_retry()

                                mlflow.log_param("registered_model_version", registered_model.version)
                                logger.info(f"📝 Registered model '{model_name}' version {registered_model.version} to individual run")

                        except Exception as model_reg_error:
                            logger.warning(f"⚠️ Model registration to individual run failed for {book_data['book_isbn']}: {model_reg_error}")

                        logger.info(f"✅ Successfully logged individual run for {book_data['book_isbn']}")

                except Exception as book_run_error:
                    logger.warning(f"⚠️ Failed to create individual run for {book_data['book_isbn']}: {book_run_error}")

            # Clean up old models after all individual runs are complete
            cleanup_old_mlflow_models(max_models_per_book=2)

            # Clean up the stored data
            book_run_count = len(train_models_from_consolidated_data._book_run_data)
            delattr(train_models_from_consolidated_data, '_book_run_data')
            logger.info(f"🎉 Completed individual runs for {book_run_count} books")

        # Extract success metrics
        total_books = training_results.get('total_books', 0)
        successful_models = training_results.get('successful_models', 0)
        failed_models = training_results.get('failed_models', 0)

        logger.info(f"ARIMA training completed: {successful_models}/{total_books} models successful")

        # Add ZenML metadata using our helper functions
        metadata_dict = create_step_metadata({
            "selected_isbns": selected_isbns,
            "total_books": total_books,
            "successful_models": successful_models,
            "failed_models": failed_models,
            "success_rate": f"{(successful_models/total_books*100):.1f}%" if total_books > 0 else "0%",
            "n_trials": config.n_trials if config else 10,
            "output_directory": arima_output_dir,
            "training_timestamp": pd.Timestamp.now().isoformat(),
            "early_stopping_enabled": config.patience > 0 if config else True,
            "patience": config.patience if config else 3,
            "min_improvement": config.min_improvement if config else 0.5,
            "min_trials": config.min_trials if config else 10,
            "config_environment": config.environment if config else "unknown",
            "smart_retraining_enabled": config is not None
        })

        # Add individual book performance if available
        book_results = training_results.get('book_results', {})
        for isbn, book_result in book_results.items():
            if 'evaluation_metrics' in book_result:
                metrics = book_result['evaluation_metrics']
                metadata_dict[f"{isbn}_mae"] = f"{metrics.get('mae', 0):.4f}"
                metadata_dict[f"{isbn}_rmse"] = f"{metrics.get('rmse', 0):.4f}"
                metadata_dict[f"{isbn}_mape"] = f"{metrics.get('mape', 0):.4f}"

            if 'best_params' in book_result:
                params = book_result['best_params']
                order = f"({params.get('p', 0)},{params.get('d', 0)},{params.get('q', 0)})"
                seasonal = f"({params.get('P', 0)},{params.get('D', 0)},{params.get('Q', 0)},52)"
                metadata_dict[f"{isbn}_model_params"] = f"SARIMA{order}{seasonal}"

        _add_step_metadata("arima_training_results", metadata_dict)

        logger.info(f"Individual ARIMA training step completed successfully")
        return training_results

    except Exception as e:
        logger.error(f"Failed to train individual ARIMA models: {e}")

        # Return error results
        error_results = {
            "total_books": len(selected_isbns) if selected_isbns else 0,
            "successful_models": 0,
            "failed_models": len(selected_isbns) if selected_isbns else 0,
            "error": str(e),
            "book_results": {},
            "training_timestamp": pd.Timestamp.now().isoformat()
        }

        error_metadata = create_step_metadata({
            "error": str(e),
            "selected_isbns": selected_isbns if selected_isbns else [],
            "training_timestamp": pd.Timestamp.now().isoformat()
        })

        _add_step_metadata("arima_training_results", error_metadata)

        return error_results
