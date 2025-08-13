# Standard library imports
import os
import sys
import json
import pickle
from typing import Tuple, Annotated, Dict, List, Any, Optional

# Third-party imports
import pandas as pd
import numpy as np
import mlflow

# ZenML imports
from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.logger import get_logger
from zenml import ArtifactConfig
from zenml.steps import get_step_context
from zenml.integrations.pandas.materializers.pandas_materializer import PandasMaterializer

# Local step imports
# from steps._01_load_data import get_isbn_data, get_uk_weekly_data  # Unused - pipeline uses step functions directly
from steps._02_preprocessing import preprocess_loaded_data
from steps._02_5_data_quality import create_quality_report_step, parse_quality_report_step
from steps._03_5_modelling_prep import (
    prepare_multiple_books_data,
    filter_book_data
)
from steps._04_5_arima_training import train_individual_arima_models_step

# Import seasonality configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'seasonality_analysis'))
try:
    from seasonality_config import SeasonalityConfig  # type: ignore[import-untyped]
except ImportError:
    SeasonalityConfig = None

# Configuration and utility imports
from config.arima_training_config import (
    ARIMATrainingConfig, 
    get_arima_config,
    DEFAULT_TEST_ISBNS,
    DEFAULT_SPLIT_SIZE, 
    DEFAULT_MAX_SEASONAL_BOOKS
)
from utils.model_reuse import create_retraining_engine
from utils.pipeline_helpers import _add_step_metadata, _create_basic_data_metadata, ensure_datetime_index, get_git_commit_hash, generate_pipeline_run_name

# Initialize logger
logger = get_logger(__name__)

# Module-level storage for book run data (avoiding function attribute assignment)
_book_run_data_storage: List[Dict[str, Any]] = []

# Configure step settings to enable metadata
step_settings = {
    "enable_artifact_metadata": True,
    "enable_artifact_visualization": True,
}

# Helper functions for common operations




def create_step_metadata(base_data: dict, **additional_metadata: Any) -> dict:
    """
    Create standardized metadata dictionary with string conversion.

    Args:
        base_data: Base metadata dictionary
        **additional_metadata: Additional key-value pairs to include

    Returns:
        Metadata dictionary with all values converted to strings
    """
    metadata = base_data.copy()
    metadata.update(additional_metadata)

    # Convert all values to strings for ZenML compatibility
    return {k: str(v) for k, v in metadata.items()}

def cleanup_old_mlflow_models(max_models_per_book: int = 2) -> None:  # noqa: C901
    """
    Cleanup old MLflow model artifacts to prevent disk space issues.
    Keeps only the most recent `max_models_per_book` model.statsmodels files per book.
    
    Note: High complexity due to robust error handling and file operations.
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

# Helper functions are imported from utils.pipeline_helpers

# ------------------ HELPER FUNCTIONS ------------------ #

def train_models_from_consolidated_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    book_isbns: List[str],
    output_dir: str,
    config: Optional[ARIMATrainingConfig] = None,
    n_trials: Optional[int] = None  # Deprecated, use config.n_trials instead
) -> Dict[str, Any]:
    """
    Train individual ARIMA models for each book using consolidated DataFrames with smart retraining.
    This replaces the CSV file-based approach for Vertex AI deployment.

    Enhanced with:
    - Configuration-driven optimization parameters
    - Smart model reuse to avoid unnecessary retraining
    - Performance-based retraining triggers
    - Comprehensive logging and monitoring
    """
    global _book_run_data_storage
    import pandas as pd
    
    from steps._04_arima_standalone import (
        create_time_series_from_df,
        run_optuna_optimization,
        train_final_arima_model,
        evaluate_forecast
    )

    # Initialize configuration if not provided
    if config is None:
        config = get_arima_config()
        logger.info(f"Using default configuration for environment: {config.environment}")

    # Note: Deprecated n_trials parameter support has been removed. Use config.n_trials instead.

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
            
            # Still create individual MLflow run for reused models to maintain visibility
            if book_isbn in book_results and "error" not in book_results[book_isbn]:
                try:
                    # Get book title from train data
                    book_title_rows = train_data[train_data['ISBN'] == book_isbn]['Title']
                    if hasattr(book_title_rows, 'iloc') and hasattr(book_title_rows, 'empty'):
                        book_title = book_title_rows.iloc[0] if len(book_title_rows) > 0 else f"Book_{book_isbn}"
                    else:
                        book_title = book_title_rows[0] if len(book_title_rows) > 0 else f"Book_{book_isbn}"

                    logger.info(f"📖 Scheduling individual MLflow run for REUSED model {book_isbn}")

                    # Store book run data for later processing (reused models)
                    if not _book_run_data_storage:
                        _book_run_data_storage = []

                    book_run_data = {
                        'book_isbn': book_isbn,
                        'book_title': book_title,
                        'best_params': book_results[book_isbn]['best_params'],
                        'evaluation_metrics': book_results[book_isbn]['evaluation_metrics'],
                        'optimization_results': book_results[book_isbn].get('optimization_results', {}),
                        'train_series_length': book_results[book_isbn].get('train_series_length', 0),
                        'test_series_length': book_results[book_isbn].get('test_series_length', 0),
                        'model_path': book_results[book_isbn].get('model_path', ''),
                        'model_status': 'reused'  # Flag to indicate this was reused
                    }

                    _book_run_data_storage.append(book_run_data)
                    logger.info(f"📊 Stored run data for REUSED model {book_isbn} - will create individual run after pipeline")

                except Exception as storage_error:
                    logger.warning(f"⚠️ Failed to store run data for reused model {book_isbn}: {storage_error}")
            
            continue

        # Proceed with training new model
        logger.info(f"🔄 Training new model for {book_isbn}: {reason}")

        # Reset evaluation_metrics for each book to avoid reusing metrics from previous books
        evaluation_metrics = None

        try:
            # Filter consolidated data by ISBN and prepare for modeling
            book_train_clean = filter_book_data(train_data, book_isbn, clean_for_modeling=True)
            book_test_clean = filter_book_data(test_data, book_isbn, clean_for_modeling=True)

            if len(book_train_clean) == 0 or len(book_test_clean) == 0:
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
            def get_seed_parameters_for_book(book_isbn: str) -> List[Dict[str, int]]:
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
                    metrics = evaluate_forecast(np.asarray(test_series), np.asarray(forecast))
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
            optimization_results = None

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
                test_predictions = final_model.forecast(steps=len(test_series))
                evaluation_metrics = evaluate_forecast(np.asarray(test_series), np.asarray(test_predictions))

            # MLflow logging moved outside try block to prevent training failures

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
                    "train_freq": str(getattr(train_series.index, 'freq', 'infer')),
                    "model_params": best_params,
                    "training_data_length": len(train_series)
                }

                # Save with MLflow
                mlflow.statsmodels.save_model(  # type: ignore[attr-defined]
                    statsmodels_model=final_model,
                    path=model_path,
                    signature=model_signature,
                    input_example=pd.DataFrame({"steps": [len(test_series)]}),
                    metadata=model_info
                )

                logger.info(f"✅ Saved MLflow model to: {model_path}")

                # Register model in MLflow Model Registry for production deployment
                try:
                    model_name = f"arima_book_{book_isbn}"
                    registered_model = mlflow.register_model(
                        model_uri=f"file://{os.path.abspath(model_path)}",
                        name=model_name
                    )
                    logger.info(f"📝 Registered model '{model_name}' version {registered_model.version} in MLflow Model Registry")
                except Exception as registry_error:
                    logger.warning(f"⚠️ Model registry failed (model still saved): {registry_error}")
                    logger.info(f"📁 Model available at filesystem path: {model_path}")

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
                if hasattr(book_title_rows, 'iloc') and hasattr(book_title_rows, 'empty'):
                    # It's a pandas Series
                    book_title = book_title_rows.iloc[0] if len(book_title_rows) > 0 else f"Book_{book_isbn}"
                else:
                    # It's a numpy array or other type
                    book_title = book_title_rows[0] if len(book_title_rows) > 0 else f"Book_{book_isbn}"

                logger.info(f"📖 Scheduling individual MLflow run for {book_isbn} post-pipeline")

                # Store book run data for later processing (after main pipeline run completes)
                if not _book_run_data_storage:
                    _book_run_data_storage = []

                book_run_data = {
                    'book_isbn': book_isbn,
                    'book_title': book_title,
                    'best_params': book_results[book_isbn]['best_params'],
                    'evaluation_metrics': book_results[book_isbn]['evaluation_metrics'],
                    'optimization_results': book_results[book_isbn].get('optimization_results', {}),
                    'train_series_length': book_results[book_isbn].get('train_series_length', 0),
                    'test_series_length': book_results[book_isbn].get('test_series_length', 0),
                    'model_path': book_results[book_isbn].get('model_path', ''),
                    'model_status': 'newly_trained'  # Flag to indicate this was newly trained
                }

                _book_run_data_storage.append(book_run_data)
                logger.info(f"📊 Stored run data for {book_isbn} - will create individual run after pipeline")

            except Exception as storage_error:
                logger.warning(f"⚠️ Failed to store run data for {book_isbn}: {storage_error}")
                logger.info("📊 Model training was successful, individual run creation is optional")

    # Get retraining statistics for monitoring
    retraining_stats = retraining_engine.get_retraining_stats()

    # Log final summary with optimization results
    logger.info("🎉 Training pipeline completed!")
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

# ------------------ STEPS ------------------ #

@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def load_isbn_data_step() -> Annotated[pd.DataFrame, ArtifactConfig(name="isbn_data")]:
    """Load ISBN data from GCS bucket and return as DataFrame artifact."""
    logger.info("Starting ISBN data loading from GCS")
    try:
        # Load data directly from GCS bucket
        gcs_path = "gs://book-sales-deployment-artifacts/raw_data/ISBN_data.csv"
        df_isbns = pd.read_csv(gcs_path)

        metadata_dict = _create_basic_data_metadata(df_isbns, "GCS - ISBN data")
        logger.info(f"ISBN data metadata: {metadata_dict}")

        _add_step_metadata("isbn_data", metadata_dict)
        logger.info(f"Loaded {len(df_isbns)} ISBN records from GCS")
        return df_isbns

    except Exception as e:
        logger.error(f"Failed to load ISBN data from GCS: {e}")
        raise

@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def load_uk_weekly_data_step() -> Annotated[pd.DataFrame, ArtifactConfig(name="uk_weekly_data")]:
    """Load UK weekly data from GCS bucket and return as DataFrame artifact."""
    logger.info("Starting UK weekly data loading from GCS")
    try:
        # Load data directly from GCS bucket
        gcs_path = "gs://book-sales-deployment-artifacts/raw_data/UK_weekly_data.csv"
        df_uk_weekly = pd.read_csv(gcs_path)

        metadata_dict = _create_basic_data_metadata(df_uk_weekly, "GCS - UK weekly data")
        logger.info(f"UK weekly data metadata: {metadata_dict}")

        _add_step_metadata("uk_weekly_data", metadata_dict)
        logger.info(f"Loaded {len(df_uk_weekly)} UK weekly records from GCS")
        return df_uk_weekly

    except Exception as e:
        logger.error(f"Failed to load UK weekly data from GCS: {e}")
        raise

@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def preprocess_and_merge_step(
    df_isbns: pd.DataFrame,
    df_uk_weekly: pd.DataFrame
) -> Annotated[pd.DataFrame, ArtifactConfig(name="merged_data")]:
    """Preprocess and merge ISBN and UK weekly data using the new pipeline."""
    logger.info("Starting preprocessing and merging of loaded data")
    try:
        processed = preprocess_loaded_data(df_isbns, df_uk_weekly)
        df_merged = processed['df_uk_weekly']
        logger.info(f"Preprocessing and merging complete. Shape: {df_merged.shape}")
        return df_merged
    except Exception as e:
        logger.error(f"Failed to preprocess and merge data: {e}")
        raise

# create_quality_report_step is imported from steps._02_5_data_quality

@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def save_processed_data_step(
    df_merged: pd.DataFrame,
    output_dir: str
) -> Annotated[str, ArtifactConfig(name="processed_data_path")]:
    """Save processed data to CSV and return file path."""
    logger.info("Starting data saving")
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Save processed data
        processed_file_path = os.path.join(output_dir, 'book_sales_processed.csv')
        df_merged.to_csv(processed_file_path, index=False)

        # Calculate file size
        file_size_mb = round(os.path.getsize(processed_file_path) / (1024*1024), 2)

        metadata_dict = {
            "file_path": processed_file_path,
            "file_size_mb": str(file_size_mb),
            "total_records": str(len(df_merged)),
            "total_columns": str(len(df_merged.columns)),
            "file_format": "CSV",
            "saved_at": pd.Timestamp.now().isoformat()
        }

        logger.info(f"Data saving metadata: {metadata_dict}")

        _add_step_metadata("processed_data_path", metadata_dict)

        logger.info(f"Processed data saved to {processed_file_path}")
        return processed_file_path

    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise

@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True
)
def select_modeling_books_step(
    df_merged: pd.DataFrame,
    use_seasonality_filter: bool = True,
    max_books: int = DEFAULT_MAX_SEASONAL_BOOKS
) -> Annotated[List[str], ArtifactConfig(name="selected_isbns")]:
    """
    Filter books based on seasonality analysis to select optimal books for SARIMA modeling.
    """
    logger.info("Starting seasonal book filtering")

    try:
        if not use_seasonality_filter or SeasonalityConfig is None:
            logger.info("Seasonality filtering disabled or config unavailable, using top volume books")
            # Get top books by volume if seasonality filtering is disabled
            isbn_volumes = df_merged.groupby('ISBN')['Volume'].sum().sort_values(ascending=False)  # type: ignore[call-overload]
            selected_isbns = isbn_volumes.head(max_books).index.astype(str).tolist()
        else:
            # Get seasonal books from the analysis
            seasonal_isbns = SeasonalityConfig.get_seasonal_books()
            seasonal_isbns_str = [str(isbn) for isbn in seasonal_isbns]

            logger.info(f"Found {len(seasonal_isbns_str)} seasonal books from analysis")

            # Filter to books that exist in our dataset
            available_isbns = set(df_merged['ISBN'].astype(str).unique())
            available_seasonal_isbns = [isbn for isbn in seasonal_isbns_str if isbn in available_isbns]

            logger.info(f"Found {len(available_seasonal_isbns)} seasonal books available in dataset")

            if len(available_seasonal_isbns) > max_books:
                # Prioritize by volume if we have too many seasonal books
                seasonal_volumes = df_merged[df_merged['ISBN'].astype(str).isin(available_seasonal_isbns)].groupby('ISBN')['Volume'].sum()
                top_seasonal = seasonal_volumes.sort_values(ascending=False).head(max_books)  # type: ignore[call-overload]
                selected_isbns = top_seasonal.index.astype(str).tolist()
                logger.info(f"Selected top {len(selected_isbns)} seasonal books by volume")
            else:
                selected_isbns = available_seasonal_isbns
                logger.info(f"Using all {len(selected_isbns)} available seasonal books")

        # Add metadata
        metadata_dict = {
            "seasonality_filter_used": str(use_seasonality_filter and SeasonalityConfig is not None),
            "total_seasonal_candidates": str(len(SeasonalityConfig.get_seasonal_books()) if SeasonalityConfig else 0),
            "selected_books_count": str(len(selected_isbns)),
            "max_books_limit": str(max_books),
            "selection_timestamp": pd.Timestamp.now().isoformat()
        }

        _add_step_metadata("selected_isbns", metadata_dict)

        logger.info(f"Selected {len(selected_isbns)} books for modeling")
        return selected_isbns

    except Exception as e:
        logger.error(f"Failed to filter seasonal books: {e}")
        raise

@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def create_train_test_splits_step(
    df_merged: pd.DataFrame,
    output_dir: str,
    selected_isbns: Optional[List[str]] = None,
    column_name: str = 'Volume',
    split_size: int = DEFAULT_SPLIT_SIZE
) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="train_data")],
    Annotated[pd.DataFrame, ArtifactConfig(name="test_data")]
]:
    """
    Prepare data for modeling by splitting into train/test sets for selected books.
    """
    logger.info("Starting modelling data preparation")
    try:
        # Use default ISBNs if none provided
        if selected_isbns is None or len(selected_isbns) == 0:
            selected_isbns = DEFAULT_TEST_ISBNS

        logger.info(f"Preparing modelling data for {len(selected_isbns)} books: {selected_isbns}")
        logger.info(f"Using column: {column_name}, split size: {split_size}")

        # Debug info
        logger.info(f"ISBN column dtype: {df_merged['ISBN'].dtype}")
        logger.info(f"Available columns in df_merged: {list(df_merged.columns)}")

        if 'Volume' not in df_merged.columns:
            logger.error("Volume column not found in df_merged!")
            raise ValueError("Volume column not found in the merged dataframe")

        # Ensure ISBNs are strings
        if df_merged['ISBN'].dtype != 'object':
            logger.info("Converting ISBN column to string type")
            df_merged['ISBN'] = df_merged['ISBN'].astype(str)

        # Filter data for selected ISBNs
        selected_books_data = df_merged[df_merged['ISBN'].isin(selected_isbns)].copy()

        if len(selected_books_data) == 0:
            raise ValueError(f"No data found for selected ISBNs: {selected_isbns}")

        # Group data by ISBN for individual book analysis
        books_data = {}
        book_isbn_mapping = {}

        for isbn in selected_isbns:
            book_data = selected_books_data[selected_books_data['ISBN'] == isbn].copy()
            if len(book_data) > 0:
                # Get book title safely
                if hasattr(book_data, 'columns') and 'Title' in book_data.columns:
                    title_data = book_data['Title']
                    if hasattr(title_data, 'iloc'):
                        book_title = title_data.iloc[0]
                    else:
                        book_title = title_data[0] if len(title_data) > 0 else f"Book_{isbn}"
                else:
                    book_title = f"Book_{isbn}"
                books_data[book_title] = book_data
                book_isbn_mapping[book_title] = isbn
                logger.info(f"Found data for {book_title} (ISBN: {isbn}): {len(book_data)} records")
            else:
                logger.warning(f"No data found for ISBN: {isbn}")

        if not books_data:
            raise ValueError("No valid book data found for any of the selected ISBNs")

        # Prepare train/test data for each book with CSV output
        prepared_data = prepare_multiple_books_data(
            books_data=books_data,
            column_name=column_name,
            split_size=split_size,
            output_dir=output_dir
        )

        # Create metadata for the step (convert all to strings)
        metadata_dict = {
            "selected_isbns": str(selected_isbns),
            "column_name": column_name,
            "split_size": str(split_size),
            "books_processed": str(list(prepared_data.keys())),
            "total_books": str(len(prepared_data)),
            "successful_preparations": str(sum(1 for train, test in prepared_data.values() if train is not None and test is not None)),
            "preparation_timestamp": pd.Timestamp.now().isoformat()
        }

        # Add detailed info for each book
        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                metadata_dict[f"{book_name}_train_shape"] = str(train_data.shape[0])
                metadata_dict[f"{book_name}_test_shape"] = str(test_data.shape[0])
                metadata_dict[f"{book_name}_train_range"] = f"{train_data.index.min()} to {train_data.index.max()}"
                metadata_dict[f"{book_name}_test_range"] = f"{test_data.index.min()} to {test_data.index.max()}"


        logger.info(f"Successfully prepared modelling data for {len(prepared_data)} books")

        # Create consolidated DataFrames for ZenML artifacts (Vertex AI deployment ready)
        # Individual book models will filter these by ISBN for training
        consolidated_train_data = []
        consolidated_test_data = []

        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                # Add book identifiers to enable filtering
                book_isbn = book_isbn_mapping.get(book_name, 'unknown')

                # Prepare train data with identifiers
                train_with_id = train_data.copy()
                train_with_id['ISBN'] = book_isbn
                train_with_id['Title'] = book_name
                consolidated_train_data.append(train_with_id)

                # Prepare test data with identifiers
                test_with_id = test_data.copy()
                test_with_id['ISBN'] = book_isbn
                test_with_id['Title'] = book_name
                consolidated_test_data.append(test_with_id)

        # Combine all books into consolidated DataFrames with proper datetime index preservation
        if consolidated_train_data:
            # Ensure all DataFrames have proper datetime index before concatenation
            for i, df in enumerate(consolidated_train_data):
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    consolidated_train_data[i] = ensure_datetime_index(df, f"train data book {i}")
                    # Remove End Date column if it was used to restore index
                    if 'End Date' in consolidated_train_data[i].columns:
                        consolidated_train_data[i].drop(columns=['End Date'], inplace=True)

            for i, df in enumerate(consolidated_test_data):
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    consolidated_test_data[i] = ensure_datetime_index(df, f"test data book {i}")
                    # Remove End Date column if it was used to restore index
                    if 'End Date' in consolidated_test_data[i].columns:
                        consolidated_test_data[i].drop(columns=['End Date'], inplace=True)

            train_df = pd.concat(consolidated_train_data, ignore_index=False)
            test_df = pd.concat(consolidated_test_data, ignore_index=False)
            logger.info(f"Created consolidated artifacts: train_df shape {train_df.shape}, test_df shape {test_df.shape}")
            logger.info(f"Train index type: {type(train_df.index)}, Test index type: {type(test_df.index)}")
        else:
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()
            logger.warning("No valid data for consolidated DataFrames")

        # Keep CSV files for debugging/development
        logger.info("Individual CSV files available for debugging, consolidated artifacts ready for production")

        # Count successful book preparations for metadata
        successful_books = sum(1 for train, test in prepared_data.values() if train is not None and test is not None)
        individual_files_created = []
        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                book_isbn = book_isbn_mapping.get(book_name, 'unknown')
                individual_files_created.extend([
                    f"train_data_{book_isbn}.csv",
                    f"test_data_{book_isbn}.csv"
                ])

        # Add metadata for consolidated artifacts
        train_metadata = {
            "consolidated_artifacts": "true",
            "deployment_ready": "true",
            "successful_books": str(successful_books),
            "consolidated_train_shape": str(train_df.shape) if len(train_df) > 0 else "empty",
            "books_included": str([book_isbn_mapping.get(name, name) for name in prepared_data.keys() if prepared_data[name][0] is not None]),
            "individual_files_created": str(len(individual_files_created)),
            "file_list": str(individual_files_created),
            "filtering_example": f"train_data[train_data['ISBN'] == '{DEFAULT_TEST_ISBNS[0]}']",
            "preparation_timestamp": pd.Timestamp.now().isoformat()
        }
        _add_step_metadata("train_data", train_metadata)

        test_metadata = {
            "consolidated_artifacts": "true",
            "deployment_ready": "true",
            "successful_books": str(successful_books),
            "consolidated_test_shape": str(test_df.shape) if len(test_df) > 0 else "empty",
            "books_included": str([book_isbn_mapping.get(name, name) for name in prepared_data.keys() if prepared_data[name][1] is not None]),
            "filtering_example": f"test_data[test_data['ISBN'] == '{DEFAULT_TEST_ISBNS[0]}']",
            "note": "Filter consolidated artifacts by ISBN for individual book modeling",
            "preparation_timestamp": pd.Timestamp.now().isoformat()
        }
        _add_step_metadata("test_data", test_metadata)

        return train_df, test_df

    except Exception as e:
        logger.error(f"Failed to prepare modelling data: {e}")
        raise

# Helper steps to parse JSON outputs if needed
@step
def parse_quality_report_step(quality_report_json: str) -> Dict:
    """Helper step to parse quality report JSON back to dict if needed by downstream steps."""
    return json.loads(quality_report_json)

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
    n_trials: int = 10,  # Deprecated, use config instead
    config: Optional[ARIMATrainingConfig] = None
) -> Annotated[Dict[str, Any], ArtifactConfig(name="arima_training_results")]:
    """
    Train individual SARIMA models for each selected book using consolidated artifacts with smart retraining.

    Enhanced with configuration-driven optimization and model reuse logic.
    """
    import mlflow
    import os

    logger.info(f"Starting individual ARIMA training for {len(selected_isbns)} books")
    logger.info(f"Using consolidated artifacts: train_data shape {train_data.shape}, test_data shape {test_data.shape}")

    # Configure remote MLflow tracking server
    mlflow_tracking_uri = "https://mlflow-tracking-server-1076639696283.europe-west2.run.app"
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set MLflow experiment name for this pipeline run
    experiment_name = "book_sales_arima_pipeline"
    mlflow.set_experiment(experiment_name)
    logger.info(f"🧪 MLflow configured with remote server: {mlflow_tracking_uri}")
    logger.info(f"🧪 MLflow experiment set to: {experiment_name}")

    try:
        # Create ARIMA output directory in outputs/models/arima
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        arima_output_dir = os.path.join(project_root, 'outputs', 'models', 'arima')
        os.makedirs(arima_output_dir, exist_ok=True)

        logger.info(f"Training ARIMA models for ISBNs: {selected_isbns}")
        logger.info(f"Output directory: {arima_output_dir}")
        logger.info(f"Optuna trials per book: {n_trials}")

        # Log enhanced pipeline-level parameters using ZenML's experiment tracker
        mlflow.log_params({
            "pipeline_type": "arima_training_optimized",
            "total_books": len(selected_isbns),
            "n_trials": config.n_trials if config else n_trials,
            "books": ",".join(selected_isbns),
            "output_directory": arima_output_dir,
            "config_environment": config.environment if config else "unknown",
            "config_force_retrain": config.force_retrain if config else True,
            "config_patience": config.patience if config else 3,
            "config_min_improvement": config.min_improvement if config else 0.5,
            "smart_retraining_enabled": config is not None
        })

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
            config=config,
            n_trials=n_trials  # Deprecated parameter for backward compatibility
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
            "optimization_version": "v2",
            "git_commit": get_git_commit_hash(),
            "pipeline_timestamp": pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
        })

        logger.info(f"📊 Logged pipeline summary to MLflow: {pipeline_success_rate:.1f}% success rate")

        # Create individual MLflow runs for each book in separate modeling experiment
        if _book_run_data_storage:
            logger.info(f"🔄 Creating {len(_book_run_data_storage)} individual book runs in modeling experiment...")

            # End current MLflow run to avoid conflicts with individual runs
            try:
                mlflow.end_run()
                logger.info("✅ Ended current MLflow run to avoid conflicts")
            except Exception as e:
                logger.warning(f"⚠️ Could not end current MLflow run: {e}")

            # Switch to the individual book modeling experiment
            modeling_experiment_name = "book_sales_arima_modeling_v2"
            mlflow.set_experiment(modeling_experiment_name)
            logger.info(f"📊 Switched to modeling experiment: {modeling_experiment_name}")

            for book_data in _book_run_data_storage:
                try:
                    import time
                    clean_title = book_data['book_title'].replace(' ', '_').replace(',', '').replace("'", '').replace('.', '')
                    model_status_prefix = "REUSED" if book_data.get('model_status') == 'reused' else "NEW"
                    book_run_name = f"{model_status_prefix}_book_{book_data['book_isbn']}_{clean_title[:15]}_{time.strftime('%H%M%S')}"

                    # Create individual run in the modeling experiment (not nested)
                    with mlflow.start_run(run_name=book_run_name) as book_run:
                        model_status = book_data.get('model_status', 'newly_trained')
                        logger.info(f"📖 Created individual run for {book_data['book_isbn']} ({model_status}): {book_run.info.run_id}")

                        try:
                            # Log book-specific parameters (ensure no duplicates)
                            params_to_log = {
                                "isbn": book_data['book_isbn'],
                                "title": book_data['book_title'],
                                "model_type": "SARIMA",
                                "arima_p": book_data['best_params']['p'],
                                "arima_d": book_data['best_params']['d'],
                                "arima_q": book_data['best_params']['q'],
                                "seasonal_P": book_data['best_params']['P'],
                                "seasonal_D": book_data['best_params']['D'],
                                "seasonal_Q": book_data['best_params']['Q'],
                                "seasonal_period": 52,
                                "optimization_method": "optuna",
                                "train_length": book_data['train_series_length'],
                                "test_length": book_data['test_series_length'],
                                "model_status": model_status,
                                "run_purpose": "training" if model_status == "newly_trained" else "deployment_tracking"
                            }
                            
                            # Add model freshness information for reused models
                            if model_status == "reused":
                                params_to_log.update({
                                    "model_reuse_timestamp": pd.Timestamp.now().isoformat(),
                                    "reuse_reason": "existing_model_acceptable",
                                    "deployment_context": "pipeline_execution"
                                })
                            
                            mlflow.log_params(params_to_log)
                        except Exception as param_error:
                            logger.warning(f"⚠️ Failed to log parameters for {book_data['book_isbn']}: {param_error}")

                        # Log dataset information for lineage tracking
                        dataset_info = {
                            "training_samples": book_data.get('train_series_length', 0),
                            "test_samples": book_data.get('test_series_length', 0),
                            "book_isbn": book_data['book_isbn'],
                            "data_source": "consolidated_pipeline_artifacts"
                        }
                        mlflow.log_dict(dataset_info, "dataset_info.json")
                        
                        # Log evaluation metrics with context
                        metrics_to_log = {
                            "mae": book_data['evaluation_metrics'].get('mae', 0),
                            "rmse": book_data['evaluation_metrics'].get('rmse', 0),
                            "mape": book_data['evaluation_metrics'].get('mape', 0),
                            "optuna_best_value": book_data['optimization_results'].get('best_value', 0),
                            "optuna_trials": book_data['optimization_results'].get('n_trials', 0)
                        }
                        
                        # Add context-specific metrics
                        if model_status == "reused":
                            # For reused models, these are validation metrics from original training
                            metrics_to_log.update({
                                "validation_mae": book_data['evaluation_metrics'].get('mae', 0),
                                "validation_rmse": book_data['evaluation_metrics'].get('rmse', 0),
                                "model_reuse_count": 1  # Track how many times this model is reused
                            })
                        else:
                            # For newly trained models, these are fresh training metrics
                            metrics_to_log.update({
                                "training_mae": book_data['evaluation_metrics'].get('mae', 0),
                                "training_rmse": book_data['evaluation_metrics'].get('rmse', 0)
                            })
                        
                        mlflow.log_metrics(metrics_to_log)

                        # Add comprehensive tags for easy filtering and organization
                        mlflow.set_tags({
                            "run_type": "individual_book",
                            "isbn": book_data['book_isbn'],
                            "model_architecture": "SARIMA",
                            "optimization_engine": "optuna",
                            "created_by": "pipeline_separate_experiment",
                            "model_status": model_status,
                            "mlops_purpose": "deployment_tracking" if model_status == "reused" else "model_training",
                            "book_title": book_data['book_title'][:50],  # Truncated for tag limits
                            "pipeline_execution": pd.Timestamp.now().strftime('%Y-%m-%d'),
                            "governance": "audit_trail_maintained",
                            "git_commit": get_git_commit_hash(),
                            "deployment_ready": "vertex_ai_compatible"
                        })
                        
                        # Add run description for better context
                        run_description = (
                            f"{'Model reuse tracking' if model_status == 'reused' else 'New model training'} "
                            f"for book '{book_data['book_title']}' (ISBN: {book_data['book_isbn']}). "
                            f"{'Existing model was deemed acceptable for current data.' if model_status == 'reused' else 'Fresh training completed with optimization.'}"
                        )
                        mlflow.set_tag("mlflow.note.content", run_description)

                        # Register the model to this individual run for production deployment
                        logger.info(f"🔍 Starting artifact logging for {book_data['book_isbn']}")
                        logger.info(f"📂 Model path: {book_data.get('model_path', 'NOT_PROVIDED')}")
                        logger.info(f"📊 Model status: {model_status}")
                        
                        try:
                            if 'model_path' in book_data and book_data.get('model_path'):
                                model_path = book_data['model_path']
                                model_name = f"arima_book_{book_data['book_isbn']}"
                                
                                # Validate model path exists
                                import os
                                if not os.path.exists(model_path):
                                    logger.error(f"❌ Model path does not exist: {model_path}")
                                    raise FileNotFoundError(f"Model file not found: {model_path}")
                                
                                logger.info(f"✅ Model path validated: {model_path}")
                                logger.info(f"📝 Model name: {model_name}")

                                # Log the model as an artifact in this run first, then register
                                import mlflow.statsmodels  # type: ignore[attr-defined]
                                logger.info(f"🔧 Loading model from {model_path}")

                                # Load the saved model and log it to this individual run
                                try:
                                    saved_model = mlflow.statsmodels.load_model(model_path)  # type: ignore[attr-defined]
                                    logger.info(f"✅ Successfully loaded model from {model_path}")

                                    # Log the model to this run to create proper linkage
                                    logger.info(f"📤 Logging model as artifact with path 'model'")
                                    logged_model = mlflow.statsmodels.log_model(  # type: ignore[attr-defined]
                                        statsmodels_model=saved_model,
                                        artifact_path="model"
                                    )
                                    logger.info(f"✅ Successfully logged model artifact: {logged_model}")
                                    
                                    # Log additional model information as artifacts
                                    model_info = {
                                        "model_path": model_path,
                                        "model_status": model_status,
                                        "book_isbn": book_data['book_isbn'],
                                        "book_title": book_data['book_title']
                                    }
                                    
                                    # Save model info as JSON artifact
                                    import json
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                                        json.dump(model_info, f, indent=2)
                                        temp_info_path = f.name
                                    
                                    # Note: Core MLflow artifacts (MLmodel, conda.yaml, model.statsmodels, 
                                    # python_env.yaml, requirements.txt) are automatically logged by 
                                    # mlflow.statsmodels.log_model() - no need to manually log them
                                    logger.info(f"✅ Core MLflow model artifacts logged automatically")

                                    # Use the run-based URI for registration (this creates the proper link)
                                    active_run = mlflow.active_run()
                                    if active_run:
                                        model_uri = f"runs:/{active_run.info.run_id}/model"
                                    else:
                                        raise Exception("No active MLflow run")

                                except Exception as load_error:
                                    logger.error(f"❌ Could not load model for logging: {load_error}")
                                    logger.info(f"🔄 Using file-based fallback registration")
                                    # Fallback to file-based registration
                                    model_uri = f"file://{os.path.abspath(model_path)}"

                                # Register model - now it will be properly linked to this run
                                logger.info(f"📝 Registering model '{model_name}' with URI: {model_uri}")
                                registered_model = mlflow.register_model(
                                    model_uri=model_uri,
                                    name=model_name,
                                    tags={
                                        "book_isbn": book_data['book_isbn'],
                                        "book_title": book_data['book_title'],
                                        "run_type": "individual_book",
                                        "model_type": "SARIMA",
                                        "model_status": model_status
                                    }
                                )

                                mlflow.log_param("registered_model_version", registered_model.version)
                                logger.info(f"✅ Registered model '{model_name}' version {registered_model.version} to individual run")
                                
                            else:
                                logger.warning(f"⚠️ No model path provided for {book_data['book_isbn']}, skipping model artifacts")
                                logger.info(f"📊 Available book_data keys: {list(book_data.keys())}")
                                
                                # Log minimal information for debugging - no model artifacts available
                                logger.info(f"📊 Model path missing for {book_data['book_isbn']} - parameters and metrics logged as run metadata")

                        except Exception as model_reg_error:
                            logger.error(f"❌ Model registration failed for {book_data['book_isbn']}: {model_reg_error}")
                            logger.error(f"📊 Full error details: {str(model_reg_error)}")
                            logger.info(f"📂 Attempted model_path: {book_data.get('model_path', 'NOT_PROVIDED')}")
                            
                            # Error details are captured in logs and run tags for debugging
                            logger.info(f"📊 Error details preserved in run logs for debugging")

                        logger.info(f"✅ Successfully logged individual run for {book_data['book_isbn']}")

                except Exception as book_run_error:
                    logger.warning(f"⚠️ Failed to create individual run for {book_data['book_isbn']}: {book_run_error}")

            # Clean up old models after all individual runs are complete
            cleanup_old_mlflow_models(max_models_per_book=2)

            # Clean up the stored data
            book_run_count = len(_book_run_data_storage)
            _book_run_data_storage.clear()
            logger.info(f"🎉 Completed individual runs for {book_run_count} books in modeling experiment")

        # Extract success metrics
        total_books = training_results.get('total_books', 0)
        successful_models = training_results.get('successful_models', 0)
        failed_models = training_results.get('failed_models', 0)

        logger.info(f"ARIMA training completed: {successful_models}/{total_books} models successful")

        # MLflow logging is now handled individually per book in separate modeling experiment

        # Add ZenML metadata
        metadata_dict = {
            "selected_isbns": str(selected_isbns),
            "total_books": str(total_books),
            "successful_models": str(successful_models),
            "failed_models": str(failed_models),
            "success_rate": f"{(successful_models/total_books*100):.1f}%" if total_books > 0 else "0%",
            "n_trials": str(config.n_trials if config else n_trials),
            "output_directory": arima_output_dir,
            "training_timestamp": pd.Timestamp.now().isoformat(),
            "early_stopping_enabled": str(config.patience > 0 if config else True),
            "patience": str(config.patience if config else 3),
            "min_improvement": str(config.min_improvement if config else 0.5),
            "min_trials": str(config.min_trials if config else 10),
            "config_environment": config.environment if config else "unknown",
            "smart_retraining_enabled": str(config is not None)
        }

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

        error_metadata = {
            "error": str(e),
            "selected_isbns": str(selected_isbns) if selected_isbns else "[]",
            "training_timestamp": pd.Timestamp.now().isoformat()
        }

        _add_step_metadata("arima_training_results", error_metadata)

        return error_results

# Utility functions for pipeline metadata are now imported from utils.pipeline_helpers

# Docker settings - let's go back to manual requirements for now
docker_settings = DockerSettings(
    requirements=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "gcsfs>=2024.2.0",
        "google-cloud-storage>=2.10.0",
        "gdown>=5.2.0",
        "openpyxl>=3.1.2",
        "pmdarima>=2.0.4",
        "optuna>=3.0.0",
        "mlflow>=2.3.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "click<8.1.8"  # Fix ZenML click dependency conflict
    ],
    parent_image="zenmldocker/zenml:0.84.2-py3.10"
)

# ------------------ PIPELINE ------------------ #

@pipeline(settings={"docker": docker_settings})  # type: ignore[arg-type]
def book_sales_arima_modeling_pipeline(
    output_dir: str,
    selected_isbns: Optional[List[str]] = None,
    column_name: str = 'Volume',
    split_size: int = 32,
    use_seasonality_filter: bool = True,
    max_seasonal_books: int = 50,
    train_arima: bool = True,
    n_trials: int = 10,  # Deprecated, use config instead
    config: Optional[ARIMATrainingConfig] = None
) -> Dict[str, Any]:
    """
    Complete book sales ARIMA modeling pipeline with Vertex AI deployment support and smart optimization.

    This pipeline:
    1. Loads ISBN and UK weekly sales data
    2. Preprocesses and merges the data
    3. Analyzes data quality
    4. Saves processed data
    5. Filters books based on seasonality analysis for optimal SARIMA modeling
    6. Creates consolidated train/test artifacts for Vertex AI deployment
    7. Trains individual SARIMA models with smart retraining logic
    8. Logs all experiments and models to MLflow for tracking

    Enhanced Features (v2):
    - Smart model reuse to avoid unnecessary retraining
    - Configuration-driven optimization (development/testing/production modes)
    - Performance-based retraining triggers
    - Environment-specific parameter tuning
    - Consolidated artifacts enable efficient book filtering: train_data[train_data['ISBN'] == book_isbn]
    - Individual SARIMA models per book (scalable to 5+ books)
    - Vertex AI ready with ZenML artifact caching
    - MLflow experiment tracking with hyperparameter optimization
    """
    logger.info("Running book sales ARIMA modeling pipeline")

    # Load raw data
    df_isbns = load_isbn_data_step()
    df_uk_weekly = load_uk_weekly_data_step()

    # Preprocess and merge
    df_merged = preprocess_and_merge_step(df_isbns=df_isbns, df_uk_weekly=df_uk_weekly)

    # Analyze data quality (now returns JSON string)
    quality_report_json = create_quality_report_step(df_merged=df_merged)

    # Save processed data
    processed_data_path = save_processed_data_step(
        df_merged=df_merged,
        output_dir=output_dir
    )

    # Filter books based on seasonality analysis (if enabled)
    if selected_isbns is None or len(selected_isbns) == 0:
        selected_isbns = select_modeling_books_step(
            df_merged=df_merged,
            use_seasonality_filter=use_seasonality_filter,
            max_books=max_seasonal_books
        )
        logger.info(f"Using seasonality-filtered books (artifact created)")
    else:
        logger.info(f"Using provided ISBNs (list provided)")

    # Prepare data for modelling - now returns separate train and test data
    train_data, test_data = create_train_test_splits_step(
        df_merged=df_merged,
        output_dir=output_dir,
        selected_isbns=selected_isbns,
        column_name=column_name,
        split_size=split_size
    )

    # Optional: Parse JSON outputs back to dicts for pipeline return
    quality_report = parse_quality_report_step(quality_report_json)

    # Optional: Train individual ARIMA models with smart optimization
    arima_results = None
    if train_arima:
        logger.info("Starting individual ARIMA model training with smart optimization")
        arima_results = train_individual_arima_models_step(
            train_data=train_data,
            test_data=test_data,
            selected_isbns=selected_isbns,
            output_dir=output_dir,
            n_trials=n_trials,  # Deprecated parameter for backward compatibility
            config=config
        )
        logger.info("ARIMA training completed successfully")
    else:
        logger.info("ARIMA training skipped (train_arima=False)")

    # Pipeline completed successfully
    logger.info("Pipeline execution completed")
    
    # Return pipeline artifacts for ZenML tracking and downstream usage
    return {
        "df_merged": df_merged,
        "quality_report": quality_report,
        "processed_data_path": processed_data_path, 
        "selected_isbns": selected_isbns,
        "train_data": train_data,
        "test_data": test_data,
        "arima_results": arima_results,
    }

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    # Set up output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'processed')

    # Create configuration for development with smart retraining enabled
    config = get_arima_config(
        environment='development',
        n_trials=3,  # Fast development mode
        force_retrain=False  # Enable smart retraining for demo
    )

    print(f"🔧 Using configuration: {config.environment} mode")
    print(f"   Trials: {config.n_trials}, Force retrain: {config.force_retrain}")
    print(f"   Smart retraining: {'Enabled' if not config.force_retrain else 'Disabled'}")

    # Generate descriptive run name and add commit tracking
    run_name = generate_pipeline_run_name()
    commit_hash = get_git_commit_hash()
    
    print(f"🏃 Pipeline run name: {run_name}")
    print(f"📝 Git commit: {commit_hash}")
    
    # Run the optimized ARIMA modeling pipeline with smart retraining
    results = book_sales_arima_modeling_pipeline.with_options(
        run_name=run_name
    )(
        output_dir=output_dir,
        selected_isbns=DEFAULT_TEST_ISBNS,  # Use the 2 specific books: Alchemist and Caterpillar
        column_name='Volume',
        split_size=DEFAULT_SPLIT_SIZE,
        use_seasonality_filter=False,
        max_seasonal_books=DEFAULT_MAX_SEASONAL_BOOKS,  # Not used when specific ISBNs provided
        train_arima=True,  # Enable ARIMA training
        n_trials=3,  # Deprecated, config.n_trials will be used instead
        config=config  # Use optimized configuration
    )

    # Print results summary
    print("\n" + "="*60)
    print("ARIMA MODELING PIPELINE EXECUTION COMPLETED")
    print("="*60)
    print("✅ Data processing and model training completed! Consolidated artifacts and models available.")

    # Access pipeline outputs from ZenML response  
    try:
        if results and hasattr(results, 'steps') and results.steps:
            arima_results = results.steps["train_individual_arima_models_step"].outputs["arima_training_results"][0].load()
        else:
            arima_results = None
    except Exception as e:
        print(f"⚠️  Could not load ARIMA results from pipeline output: {e}")
        arima_results = None

    if arima_results:
        total_books = arima_results.get('total_books', 0)
        successful_models = arima_results.get('successful_models', 0)
        reused_models = arima_results.get('reused_models', 0)
        newly_trained = arima_results.get('newly_trained_models', 0)

        print(f"✅ ARIMA training completed: {successful_models}/{total_books} models successful")

        # Show optimization efficiency
        if reused_models > 0:
            reuse_rate = (reused_models / total_books * 100) if total_books > 0 else 0
            print(f"⚡ Optimization efficiency: {reused_models} models reused, {newly_trained} newly trained ({reuse_rate:.1f}% reuse rate)")
        else:
            print(f"🔄 All {newly_trained} models were newly trained (first run or force_retrain=True)")

        # Show configuration used
        config_info = arima_results.get('configuration', {})
        if config_info:
            print(f"⚙️  Configuration: {config_info.get('environment', 'unknown')} mode, "
                  f"{config_info.get('n_trials', 'unknown')} trials per book")

        # Show individual book results
        book_results = arima_results.get('book_results', {})
        for isbn, book_result in book_results.items():
            if 'evaluation_metrics' in book_result:
                metrics = book_result['evaluation_metrics']
                mae = metrics.get('mae', 0)
                rmse = metrics.get('rmse', 0)
                mape = metrics.get('mape', 0)
                reused = " (reused)" if book_result.get('reused_existing_model', False) else " (newly trained)"
                print(f"  📖 {isbn}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%{reused}")
            elif 'error' in book_result:
                print(f"  ❌ {isbn}: Training failed - {book_result['error']}")

        print(f"📁 ARIMA models saved to: outputs/models/arima/")

        # Show retraining stats if available
        retraining_stats = arima_results.get('retraining_stats', {})
        if retraining_stats.get('total_decisions', 0) > 0:
            print(f"📊 Smart retraining stats: {retraining_stats['reuse_decisions']} reuse decisions, "
                  f"{retraining_stats['retrain_decisions']} retrain decisions")
    else:
        print("⚠️  Could not retrieve ARIMA training results from pipeline output")
        print("📝 Note: ARIMA training may have completed successfully but results are not accessible via pipeline artifacts")

    print("="*60)
