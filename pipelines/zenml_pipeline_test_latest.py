import os
import pandas as pd
from typing import Tuple, Annotated, Dict

import json
import pickle
from steps._04_model_arima import tune_arima, fit_final_arima

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
from steps._04_arima_zenml_mlflow_optuna import train_arima_optuna_step

logger = get_logger(__name__)

# Configure step settings to enable metadata
step_settings = {
    "enable_artifact_metadata": True,
    "enable_artifact_visualization": True,
}

# ------------------ STEPS ------------------ #

@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def load_isbn_data_step() -> Annotated[pd.DataFrame, ArtifactConfig(name="isbn_data")]:
    """Load ISBN data from Google Sheets and return as DataFrame artifact."""
    logger.info("Starting ISBN data loading")
    
    try:
        df_isbns = get_isbn_data()
        
        # Prepare metadata
        metadata_dict = {
            "total_records": len(df_isbns),
            "columns": list(df_isbns.columns) if hasattr(df_isbns, 'columns') else [],
            "data_shape": f"{df_isbns.shape[0]} rows x {df_isbns.shape[1]} columns",
            "source": "Google Sheets - ISBN data",
            "missing_values": df_isbns.isna().sum().to_dict()
        }
        
        logger.info(f"ISBN data metadata: {metadata_dict}")
        
        try:
            context = get_step_context()
            context.add_output_metadata(
                output_name="isbn_data",
                metadata=metadata_dict
            )
            logger.info("Successfully added ISBN data metadata")
        except Exception as e:
            logger.error(f"Failed to add ISBN data metadata: {e}")
        
        logger.info(f"Loaded {len(df_isbns)} ISBN records")
        return df_isbns
        
    except Exception as e:
        logger.error(f"Failed to load ISBN data: {e}")
        raise

@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def load_uk_weekly_data_step() -> Annotated[pd.DataFrame, ArtifactConfig(name="uk_weekly_data")]:
    """Load UK weekly data from Google Sheets and return as DataFrame artifact."""
    logger.info("Starting UK weekly data loading")
    
    try:
        df_uk_weekly = get_uk_weekly_data()
        
        # Prepare metadata
        metadata_dict = {
            "total_records": len(df_uk_weekly),
            "columns": list(df_uk_weekly.columns) if hasattr(df_uk_weekly, 'columns') else [],
            "data_shape": f"{df_uk_weekly.shape[0]} rows x {df_uk_weekly.shape[1]} columns",
            "source": "Google Sheets - UK weekly data",
            "missing_values": df_uk_weekly.isna().sum().to_dict()
        }
        
        logger.info(f"UK weekly data metadata: {metadata_dict}")
        
        try:
            context = get_step_context()
            context.add_output_metadata(
                output_name="uk_weekly_data",
                metadata=metadata_dict
            )
            logger.info("Successfully added UK weekly data metadata")
        except Exception as e:
            logger.error(f"Failed to add UK weekly data metadata: {e}")
        
        logger.info(f"Loaded {len(df_uk_weekly)} UK weekly records")
        return df_uk_weekly
        
    except Exception as e:
        logger.error(f"Failed to load UK weekly data: {e}")
        raise

@step(
    # enable_cache=False,
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
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def analyze_data_quality_step(df_merged: pd.DataFrame) -> Annotated[Dict, ArtifactConfig(name="data_quality_report")]:
    """Analyze data quality and return quality metrics."""
    logger.info("Starting data quality analysis")
    
    try:
        # Calculate quality metrics
        total_records = len(df_merged)
        missing_values = df_merged.isna().sum()
        missing_percentage = (missing_values / total_records * 100).round(2)
        
        # Unique values analysis
        unique_counts = {}
        for col in df_merged.columns:
            unique_counts[col] = df_merged[col].nunique()
        
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
        
        logger.info(f"Data quality report: {quality_report}")
        
        try:
            context = get_step_context()
            context.add_output_metadata(
                output_name="data_quality_report",
                metadata=quality_report
            )
            logger.info("Successfully added data quality metadata")
        except Exception as e:
            logger.error(f"Failed to add data quality metadata: {e}")
        
        logger.info(f"Data quality analysis completed. Quality score: {quality_report['quality_score']}%")
        return quality_report
        
    except Exception as e:
        logger.error(f"Failed to analyze data quality: {e}")
        raise

@step(
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
            "file_size_mb": file_size_mb,
            "total_records": len(df_merged),
            "total_columns": len(df_merged.columns),
            "file_format": "CSV",
            "saved_at": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Data saving metadata: {metadata_dict}")
        
        try:
            context = get_step_context()
            context.add_output_metadata(
                output_name="processed_data_path",
                metadata=metadata_dict
            )
            logger.info("Successfully added data saving metadata")
        except Exception as e:
            logger.error(f"Failed to add data saving metadata: {e}")
        
        logger.info(f"Processed data saved to {processed_file_path}")
        return processed_file_path
        
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise

@step
def prepare_book_data_step(df_merged: pd.DataFrame) -> Annotated[Dict[str, pd.DataFrame], ArtifactConfig(name="book_data")]:
    """Prepares book data using the already processed DatetimeIndex."""
    logger.info("Preparing per-book DataFrame data using existing index.")

    # Original ISBNs defined in preprocessing (as strings - best practice for ISBNs)
    original_isbns = ['9780722532935', '9780241003008']  # The Alchemist, The Very Hungry Caterpillar
    original_titles = {
        '9780722532935': 'The Alchemist',
        '9780241003008': 'The Very Hungry Caterpillar'
    }
    
    # Convert ISBN column to string to ensure consistent comparison
    df_merged['ISBN'] = df_merged['ISBN'].astype(str)
    
    # Check what ISBNs are actually available
    available_isbns = df_merged['ISBN'].unique()
    logger.info(f"Available ISBNs in data: {len(available_isbns)}")
    
    # Try to use original ISBNs first
    selected_isbns = []
    book_titles = {}
    
    for isbn in original_isbns:
        if isbn in available_isbns:
            selected_isbns.append(isbn)
            book_titles[isbn] = original_titles[isbn]
            logger.info(f"Found original ISBN {isbn}: {original_titles[isbn]}")
        else:
            logger.warning(f"Original ISBN {isbn} ({original_titles[isbn]}) not found in data")
    
    # If no original ISBNs found, use available ones
    if not selected_isbns:
        logger.warning("No original ISBNs found in data. Using first 2 available ISBNs instead.")
        selected_isbns = available_isbns[:2]
        
        # Get titles for available ISBNs
        for isbn in selected_isbns:
            title = df_merged[df_merged['ISBN'] == isbn]['Title'].iloc[0]
            book_titles[isbn] = title
            logger.info(f"Using available ISBN {isbn}: {title}")
    
    logger.info(f"Final selected ISBNs: {selected_isbns}")
    
    # Filter for selected books
    df_selected = df_merged[df_merged['ISBN'].isin(selected_isbns)].copy()
    
    if df_selected.empty:
        raise ValueError("No data found for selected books. Check ISBN values.")
    
    logger.info(f"Filtered data for {len(selected_isbns)} books: {df_selected.shape}")
    
    # Trust the preprocessed data - the DataFrame's index is already the correct date
    # Add the date index as a column so we can group by it later
    df_selected['date'] = df_selected.index
    
    logger.info(f"Using existing datetime index from preprocessing")
    logger.info(f"Date range: {df_selected['date'].min()} to {df_selected['date'].max()}")
    
    # Create DataFrame for each book
    book_dataframes = {}
    
    for isbn in selected_isbns:
        book_data = df_selected[df_selected['ISBN'] == isbn].copy()
        
        if book_data.empty:
            logger.warning(f"No data found for ISBN {isbn}")
            continue
            
        # Remove any negative volume values (returns/refunds)
        book_data = book_data[book_data['Volume'] >= 0]
        
        # Sort by date
        book_data = book_data.sort_values('date')
        
        book_name = book_titles.get(isbn, f"Book_{isbn}")
        book_dataframes[book_name] = book_data
        
        logger.info(f"Prepared DataFrame for {book_name}: {len(book_data)} data points")
        logger.info(f"Date range: {book_data['date'].min()} to {book_data['date'].max()}")
    
    if not book_dataframes:
        raise ValueError("No valid DataFrames created for any selected books")
    
    logger.info(f"Successfully prepared DataFrames for {len(book_dataframes)} books")
    return book_dataframes

@step
def tune_and_fit_all_arima_models_step(
    book_dataframes: Dict[str, pd.DataFrame]
) -> Annotated[Dict, ArtifactConfig(name="all_book_results")]:
    """
    Tune and fit ARIMA models for all books provided in the input dictionary.
    """
    logger.info("Starting ARIMA tuning and fitting for all books.")
    
    book_results = {}
    
    for book_name, df_book in book_dataframes.items():
        try:
            logger.info(f"--- Processing: {book_name} ---")
            
            # 1. Prepare time series data - use the preprocessed data directly
            # The data is already weekly and properly formatted from preprocessing
            time_series = df_book.groupby('date')['Volume'].sum().sort_index()
            
            if time_series.empty:
                logger.warning(f"Skipping {book_name} due to empty time series.")
                continue

            logger.info(f"Created time series for {book_name}: {len(time_series)} points")
            logger.info(f"Time series date range: {time_series.index.min()} to {time_series.index.max()}")

            # 2. Tune hyperparameters
            best_params, best_score = tune_arima(time_series, seasonal_period=52, n_trials=30)
            best_params['score'] = best_score
            logger.info(f"Best ARIMA params for {book_name}: {best_params}")

            # 3. Fit the final model
            model, residuals = fit_final_arima(time_series, best_params, seasonal_period=52)

            # 4. Save model and residuals
            model_path = f"final_arima_model_{book_name.replace(' ', '_')}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            residuals_path = f"arima_residuals_{book_name.replace(' ', '_')}.csv"
            pd.DataFrame({"residuals": residuals}).to_csv(residuals_path, index=True)
            logger.info(f"Saved model and residuals for {book_name}")
            
            # 5. Store results for this book
            book_results[book_name] = {
                "model_path": model_path,
                "residuals_path": residuals_path,
                "best_params": best_params,
                "dataframe_shape": df_book.shape,
                "time_series_length": len(time_series)
            }
        except Exception as e:
            logger.error(f"Failed to process model for {book_name}: {e}")

    logger.info(f"Completed ARIMA modeling for {len(book_results)} books")
    return book_results

@step
def load_processed_data_step(data_path: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="loaded_processed_data")]:
    """Load processed data from CSV file."""
    logger.info(f"Loading processed data from {data_path}")
    
    try:
        df_loaded = pd.read_csv(data_path)
        logger.info(f"Loaded processed data with shape: {df_loaded.shape}")
        return df_loaded
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        raise

# ------------------ PIPELINE ------------------ #

@pipeline
def book_sales_pipeline(output_dir: str) -> Dict:
    """Book sales data processing pipeline (data processing only)."""
    logger.info("Running pipeline: book_sales_pipeline (data processing only)")
    
    # Load raw data
    df_isbns = load_isbn_data_step()
    df_uk_weekly = load_uk_weekly_data_step()
    
    # Preprocess and merge
    df_merged = preprocess_and_merge_step(df_isbns=df_isbns, df_uk_weekly=df_uk_weekly)
    
    # Analyze data quality
    quality_report = analyze_data_quality_step(df_merged=df_merged)
    
    # Save processed data
    processed_data_path = save_processed_data_step(
        df_merged=df_merged, 
        output_dir=output_dir
    )
    
    return {
        "df_merged": df_merged,
        "quality_report": quality_report,
        "processed_data_path": processed_data_path,
    }

@pipeline(
    enable_cache=False
)
def book_sales_model_pipeline(data_path: str) -> Dict:
    """Book sales ARIMA modeling pipeline for multiple books."""
    logger.info("Running pipeline: book_sales_model_pipeline")
    
    # Load the processed data
    df_merged = load_processed_data_step(data_path=data_path)
    
    # Prepare DataFrame data for each book
    book_dataframes = prepare_book_data_step(df_merged=df_merged)
    
    # Tune, fit, and save models for all books in one step
    all_results = tune_and_fit_all_arima_models_step(book_dataframes=book_dataframes)
    
    return all_results

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    # Set up output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'processed')

    # 1. Run data processing pipeline
    data_run = book_sales_pipeline(output_dir=output_dir)
    print("Data pipeline run complete!")

    # 2. Get the processed data directly from the pipeline return
    # The pipeline returns a dictionary with the processed data
    pipeline_outputs = data_run.model_dump()
    print(f"Pipeline outputs keys: {list(pipeline_outputs.keys())}")
    
    # Since we can't easily access the step outputs, let's run the preprocessing step again
    # to get the data we need for the model pipeline
    print("Re-running preprocessing to get data for model pipeline...")
    
    # Import the preprocessing function directly
    from steps._01_load_data import get_isbn_data, get_uk_weekly_data
    from steps._02_preprocessing import preprocess_loaded_data
    
    # Load and preprocess data
    df_isbns = get_isbn_data()
    df_uk_weekly = get_uk_weekly_data()
    processed = preprocess_loaded_data(df_isbns, df_uk_weekly)
    processed_data = processed['df_uk_weekly']
    
    print(f"Retrieved processed data with shape: {processed_data.shape}")
    
    # 3. Run model training pipeline using the retrieved DataFrame
    print("Starting model training pipeline...")
    model_run = book_sales_model_pipeline(data_path=os.path.join(output_dir, 'book_sales_processed.csv'))
    print("Model pipeline run complete!")
    
    # 4. Access model results (optional)
    try:
        model_results = model_run.model_dump()
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime_to_str(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_str(item) for item in obj]
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            else:
                return obj
        
        serializable_results = convert_datetime_to_str(model_results)
        
        print("Final Model Results:")
        print(json.dumps(serializable_results, indent=2))
    except Exception as e:
        print(f"Could not access model results: {e}")
        # Try to get basic info about the run
        try:
            print(f"Model pipeline completed successfully!")
            print(f"Check ZenML dashboard for detailed results.")
        except:
            pass
    
    print("\nAll done! Check ZenML dashboard for details.")
    # The dashboard URL is printed when the pipeline starts

