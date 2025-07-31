import os
import pandas as pd
from typing import Tuple, Annotated, Dict

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
from steps._04_arima_zenml_mlflow_optuna.py import train_arima_optuna_step

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

    # NEW STEP
    arima_results = train_arima_optuna_step(df_merged=df_merged)
    
    return {
        "df_merged": df_merged,
        "quality_report": quality_report,
        "processed_data_path": processed_data_path,
        "arima_results": arima_results
    }

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    # Set up output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'processed')

    # Run the pipeline
    book_sales_pipeline(output_dir=output_dir)
    print("Pipeline run submitted! Check the ZenML dashboard for outputs.")

    # If you want to inspect outputs programmatically, 
    # let me know and I can show you how to do that using ZenML's artifact store 
    # or by running your logic outside the pipeline system.

