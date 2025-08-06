import os
import pandas as pd
import numpy as np
import json
from typing import Tuple, Annotated, Dict, List, Any
import pickle

from zenml import step, pipeline
from zenml.logger import get_logger
from zenml import ArtifactConfig
from zenml.steps import get_step_context
from zenml.integrations.pandas.materializers.pandas_materializer import PandasMaterializer

# Import your existing modules
from steps._01_load_data import (
    get_isbn_data,
    get_uk_weekly_data
)
from steps._02_preprocessing import preprocess_loaded_data
from steps._03_5_modelling_prep import prepare_data_after_2012, prepare_multiple_books_data
from steps._04_arima_standalone import train_multiple_books_arima

# Import seasonality configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'seasonality_analysis'))
try:
    from seasonality_config import SeasonalityConfig
except ImportError:
    SeasonalityConfig = None
    logger.warning("SeasonalityConfig not found. Seasonality filtering will be disabled.")

logger = get_logger(__name__)

# Constants
DEFAULT_TEST_ISBNS = [
    '9780722532935',  # The Alchemist
    '9780241003008',  # Very Hungry Caterpillar, The
]
DEFAULT_SPLIT_SIZE = 32
DEFAULT_MAX_SEASONAL_BOOKS = 50

# Configure step settings to enable metadata
step_settings = {
    "enable_artifact_metadata": True,
    "enable_artifact_visualization": True,
}

# Helper functions for common operations
def _add_step_metadata(output_name: str, metadata_dict: dict) -> None:
    """Helper function to add metadata to step outputs with error handling."""
    try:
        context = get_step_context()
        context.add_output_metadata(
            output_name=output_name,
            metadata=metadata_dict
        )
        logger.info(f"Successfully added {output_name} metadata")
    except Exception as e:
        logger.error(f"Failed to add {output_name} metadata: {e}")

def _create_basic_data_metadata(df: pd.DataFrame, source: str) -> dict:
    """Helper function to create basic metadata for DataFrames."""
    return {
        "total_records": str(len(df)),
        "columns": str(list(df.columns) if hasattr(df, 'columns') else []),
        "data_shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "source": source,
        "missing_values": str(df.isna().sum().to_dict())
    }

# ------------------ STEPS ------------------ #

@step(
    enable_cache=False,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def load_isbn_data_step() -> Annotated[pd.DataFrame, ArtifactConfig(name="isbn_data")]:
    """Load ISBN data from Google Sheets and return as DataFrame artifact."""
    logger.info("Starting ISBN data loading")
    try:
        df_isbns = get_isbn_data()
        metadata_dict = _create_basic_data_metadata(df_isbns, "Google Sheets - ISBN data")
        logger.info(f"ISBN data metadata: {metadata_dict}")

        _add_step_metadata("isbn_data", metadata_dict)
        logger.info(f"Loaded {len(df_isbns)} ISBN records")
        return df_isbns

    except Exception as e:
        logger.error(f"Failed to load ISBN data: {e}")
        raise

@step(
    enable_cache=False,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def load_uk_weekly_data_step() -> Annotated[pd.DataFrame, ArtifactConfig(name="uk_weekly_data")]:
    """Load UK weekly data from Google Sheets and return as DataFrame artifact."""
    logger.info("Starting UK weekly data loading")
    try:
        df_uk_weekly = get_uk_weekly_data()
        metadata_dict = _create_basic_data_metadata(df_uk_weekly, "Google Sheets - UK weekly data")
        logger.info(f"UK weekly data metadata: {metadata_dict}")

        _add_step_metadata("uk_weekly_data", metadata_dict)
        logger.info(f"Loaded {len(df_uk_weekly)} UK weekly records")
        return df_uk_weekly

    except Exception as e:
        logger.error(f"Failed to load UK weekly data: {e}")
        raise

@step(
    enable_cache=False,
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

@step(
    enable_cache=False,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def create_quality_report_step(df_merged: pd.DataFrame) -> Annotated[str, ArtifactConfig(name="data_quality_report")]:
    """Analyze data quality and return quality metrics as JSON string."""
    logger.info("Starting data quality analysis")
    try:
        # Calculate quality metrics
        total_records = len(df_merged)
        missing_values = df_merged.isna().sum()
        missing_percentage = (missing_values / total_records * 100).round(2)

        # Unique values analysis
        unique_counts = {}
        for col in df_merged.columns:
            unique_counts[col] = int(df_merged[col].nunique())  # Convert to int for JSON serialization

        # Data types
        data_types = df_merged.dtypes.astype(str).to_dict()

        # Basic statistics for numeric columns
        numeric_stats = {}
        numeric_cols = df_merged.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            numeric_stats[col] = {
                'mean': float(df_merged[col].mean()),
                'std': float(df_merged[col].std()),
                'min': float(df_merged[col].min()),
                'max': float(df_merged[col].max()),
                'median': float(df_merged[col].median())
            }

        quality_report = {
            "total_records": total_records,
            "total_columns": len(df_merged.columns),
            "missing_values_count": missing_values.to_dict(),
            "missing_values_percentage": missing_percentage.to_dict(),
            "unique_values_per_column": unique_counts,
            "data_types": data_types,
            "numeric_statistics": numeric_stats,
            "quality_score": round((1 - missing_values.sum() / (total_records * len(df_merged.columns))) * 100, 2)
        }

        # Convert to JSON string for ZenML compatibility
        quality_report_json = json.dumps(quality_report, indent=2, default=str)

        logger.info(f"Data quality analysis completed. Quality score: {quality_report['quality_score']}%")

        quality_metadata = {
            "quality_score": str(quality_report['quality_score']),
            "total_records": str(quality_report['total_records']),
            "total_columns": str(quality_report['total_columns'])
        }
        _add_step_metadata("data_quality_report", quality_metadata)

        return quality_report_json

    except Exception as e:
        logger.error(f"Failed to analyze data quality: {e}")
        raise

@step(
    enable_cache=False,
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
    enable_cache=False,
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
            isbn_volumes = df_merged.groupby('ISBN')['Volume'].sum().sort_values(ascending=False)
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
                top_seasonal = seasonal_volumes.sort_values(ascending=False).head(max_books)
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
    enable_cache=False,  # Disable cache to force re-execution with our fix
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def create_train_test_splits_step(
    df_merged: pd.DataFrame,
    output_dir: str,
    selected_isbns: List[str] = None,
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

        if selected_books_data.empty:
            raise ValueError(f"No data found for selected ISBNs: {selected_isbns}")

        # Group data by ISBN for individual book analysis
        books_data = {}
        book_isbn_mapping = {}

        for isbn in selected_isbns:
            book_data = selected_books_data[selected_books_data['ISBN'] == isbn].copy()
            if not book_data.empty:
                book_title = book_data['Title'].iloc[0] if 'Title' in book_data.columns else f"Book_{isbn}"
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

        _add_step_metadata("modelling_data", metadata_dict)

        logger.info(f"Successfully prepared modelling data for {len(prepared_data)} books")

        # For individual book modeling, we rely on the individual CSV files created by prepare_multiple_books_data()
        # These are already saved as train_data_{isbn}.csv and test_data_{isbn}.csv
        
        # Create empty DataFrames for ZenML artifact compatibility
        # Individual book models should use the CSV files directly
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        logger.info("Individual book files created - combined DataFrames intentionally empty for individual modeling")
        
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

        # Add metadata for individual book files
        train_metadata = {
            "individual_modeling_approach": "true",
            "successful_books": str(successful_books),
            "individual_files_created": str(len(individual_files_created)),
            "file_list": str(individual_files_created),
            "preparation_timestamp": pd.Timestamp.now().isoformat()
        }
        _add_step_metadata("train_data", train_metadata)

        test_metadata = {
            "individual_modeling_approach": "true", 
            "successful_books": str(successful_books),
            "note": "Use individual CSV files for training separate models per book",
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
    enable_cache=False,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def train_individual_arima_models_step(
    selected_isbns: List[str],
    output_dir: str,
    n_trials: int = 50
) -> Annotated[Dict[str, Any], ArtifactConfig(name="arima_training_results")]:
    """
    Train individual SARIMA models for each selected book using existing functions.
    """
    logger.info(f"Starting individual ARIMA training for {len(selected_isbns)} books")
    
    try:
        # Create ARIMA output directory
        arima_output_dir = os.path.join(output_dir, 'arima_models')
        os.makedirs(arima_output_dir, exist_ok=True)
        
        logger.info(f"Training ARIMA models for ISBNs: {selected_isbns}")
        logger.info(f"Output directory: {arima_output_dir}")
        logger.info(f"Optuna trials per book: {n_trials}")
        
        # Use the existing train_multiple_books_arima function
        training_results = train_multiple_books_arima(
            book_isbns=selected_isbns,
            output_dir=arima_output_dir,
            n_trials=n_trials
        )
        
        # Extract success metrics
        total_books = training_results.get('total_books', 0)
        successful_models = training_results.get('successful_models', 0)
        failed_models = training_results.get('failed_models', 0)
        
        logger.info(f"ARIMA training completed: {successful_models}/{total_books} models successful")
        
        # Add ZenML metadata
        metadata_dict = {
            "selected_isbns": str(selected_isbns),
            "total_books": str(total_books),
            "successful_models": str(successful_models),
            "failed_models": str(failed_models),
            "success_rate": f"{(successful_models/total_books*100):.1f}%" if total_books > 0 else "0%",
            "n_trials_per_book": str(n_trials),
            "output_directory": arima_output_dir,
            "training_timestamp": pd.Timestamp.now().isoformat(),
            "early_stopping_enabled": "true",
            "patience": "5",
            "min_improvement": "0.1",
            "min_trials": "15"
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

# ------------------ PIPELINE ------------------ #

@pipeline
def book_sales_data_preparation_pipeline(
    output_dir: str,
    selected_isbns: List[str] = None,
    column_name: str = 'Volume',
    split_size: int = 32,
    use_seasonality_filter: bool = True,
    max_seasonal_books: int = 50,
    train_arima: bool = True,
    arima_n_trials: int = 50
) -> Dict:
    """
    Complete book sales data processing and ARIMA training pipeline.

    This pipeline:
    1. Loads ISBN and UK weekly sales data
    2. Preprocesses and merges the data
    3. Analyzes data quality
    4. Saves processed data
    5. Filters books based on seasonality analysis for optimal SARIMA modeling
    6. Prepares train/test data for modeling and saves to data/processed
    7. Trains individual SARIMA models for each selected book (optional)
    """
    logger.info("Running book sales data preparation pipeline")

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

    # Optional: Train individual ARIMA models
    arima_results = None
    if train_arima:
        logger.info("Starting individual ARIMA model training")
        arima_results = train_individual_arima_models_step(
            selected_isbns=selected_isbns,
            output_dir=output_dir,
            n_trials=arima_n_trials
        )
        logger.info("ARIMA training completed successfully")
    else:
        logger.info("ARIMA training skipped (train_arima=False)")

    return {
        "df_merged": df_merged,
        "quality_report": quality_report,  # Parsed dict
        "processed_data_path": processed_data_path,
        "selected_isbns": selected_isbns,  # Selected ISBNs
        "train_data": train_data,  # Training data artifact
        "test_data": test_data,    # Test data artifact
        "arima_results": arima_results,  # ARIMA training results
    }

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    # Set up output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'processed')

    # Run the complete data preparation and ARIMA training pipeline with specific books
    results = book_sales_data_preparation_pipeline(
        output_dir=output_dir,
        selected_isbns=DEFAULT_TEST_ISBNS,  # Use the 2 specific books: Alchemist and Caterpillar
        column_name='Volume',
        split_size=DEFAULT_SPLIT_SIZE,
        use_seasonality_filter=False,
        max_seasonal_books=DEFAULT_MAX_SEASONAL_BOOKS,  # Not used when specific ISBNs provided
        train_arima=True,  # Enable ARIMA training
        arima_n_trials=50  # Number of Optuna trials per book
    )

    # Print results summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*60)
    print("✅ Data preparation completed! Train and test data saved to data/processed directory.")
    
    if results.get('arima_results'):
        arima_results = results['arima_results']
        total_books = arima_results.get('total_books', 0)
        successful_models = arima_results.get('successful_models', 0)
        
        print(f"✅ ARIMA training completed: {successful_models}/{total_books} models trained successfully")
        
        # Show individual book results
        book_results = arima_results.get('book_results', {})
        for isbn, book_result in book_results.items():
            if 'evaluation_metrics' in book_result:
                metrics = book_result['evaluation_metrics']
                mae = metrics.get('mae', 0)
                rmse = metrics.get('rmse', 0)
                mape = metrics.get('mape', 0)
                print(f"  📖 {isbn}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
            elif 'error' in book_result:
                print(f"  ❌ {isbn}: Training failed - {book_result['error']}")
        
        print(f"📁 ARIMA models saved to: {output_dir}/arima_models/")
    else:
        print("⚠️  ARIMA training was skipped or failed")
    
    print("="*60)
