"""
Data Preprocessing Steps

This module contains ZenML step definitions for data preprocessing and merging operations.
"""

import os
import pandas as pd
from typing import Annotated
from zenml import step, ArtifactConfig
from zenml.materializers.pandas_materializer import PandasMaterializer
from zenml.logger import get_logger
from utils.zenml_helpers import _add_step_metadata
from scripts.preprocessing import preprocess_loaded_data

# Initialize logger
logger = get_logger(__name__)


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