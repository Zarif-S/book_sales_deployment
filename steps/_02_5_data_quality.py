"""
Data Quality Analysis Step

This module contains the data quality analysis step that evaluates
the processed data and provides comprehensive quality metrics.
"""

import json
import pandas as pd
from typing import Annotated
from zenml import step
from zenml.logger import get_logger
from zenml import ArtifactConfig
from utils.zenml_helpers import _add_step_metadata

# Initialize logger
logger = get_logger(__name__)


@step(
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
)
def create_quality_report_step(df_merged: pd.DataFrame) -> Annotated[str, ArtifactConfig(name="data_quality_report")]:
    """
    Analyze data quality and return quality metrics as JSON string.
    
    Args:
        df_merged: Merged DataFrame to analyze
        
    Returns:
        JSON string containing comprehensive data quality metrics
    """
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


@step
def parse_quality_report_step(quality_report_json: str) -> dict:
    """
    Helper step to parse quality report JSON back to dict if needed by downstream steps.
    
    Args:
        quality_report_json: JSON string from quality report step
        
    Returns:
        Parsed quality report as dictionary
    """
    return json.loads(quality_report_json)