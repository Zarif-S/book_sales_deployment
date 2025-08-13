# Standard library imports
import os
import sys
from typing import Tuple, Dict, List, Any, Optional

# Third-party imports
import pandas as pd

# ZenML imports
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.logger import get_logger

# Step imports
from steps.load_data_steps import load_isbn_data_step, load_uk_weekly_data_step
from steps.preprocessing_steps import preprocess_and_merge_step, save_processed_data_step
from steps._02_5_data_quality import create_quality_report_step
from steps.modeling_prep_steps import select_modeling_books_step, create_train_test_splits_step, parse_quality_report_step
from steps._04_5_arima_training import train_individual_arima_models_step

# Configuration and utility imports
from config.arima_training_config import (
    ARIMATrainingConfig, 
    get_arima_config,
    DEFAULT_TEST_ISBNS,
    DEFAULT_SPLIT_SIZE, 
    DEFAULT_MAX_SEASONAL_BOOKS
)
from utils.pipeline_utils import get_git_commit_hash, generate_pipeline_run_name

# Initialize logger
logger = get_logger(__name__)

# Docker settings for containerization
docker_settings = DockerSettings(
    requirements=[
        "gdown",
        "scikit-learn",
        "statsmodels",
        "optuna",
        "mlflow",
        "google-cloud-storage"
    ]
)


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