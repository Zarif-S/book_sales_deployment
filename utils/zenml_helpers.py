"""
ZenML Helper Utilities

This module contains utility functions specifically for ZenML operations,
including metadata management, step utilities, and common data operations
used across the pipeline steps.
"""

import pandas as pd
from typing import Dict, Any
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def _add_step_metadata(output_name: str, metadata_dict: dict) -> None:
    """
    Helper function to add metadata to step outputs with error handling.
    
    Args:
        output_name: Name of the step output to add metadata to
        metadata_dict: Dictionary of metadata to add
    """
    try:
        from zenml.steps import get_step_context
        context = get_step_context()
        context.add_output_metadata(
            output_name=output_name,
            metadata=metadata_dict
        )
        logger.info(f"Successfully added {output_name} metadata")
    except ImportError:
        logger.warning(f"ZenML step context not available, skipping metadata for {output_name}")
    except Exception as e:
        logger.warning(f"Could not add {output_name} metadata: {e}")  # Changed from error to warning


def create_step_metadata(base_data: dict, **additional_metadata) -> dict:
    """
    Create standardized metadata dictionary with string conversion.
    
    Args:
        base_data: Base metadata dictionary
        **additional_metadata: Additional key-value pairs to include
        
    Returns:
        Metadata dictionary with all values converted to strings for ZenML compatibility
    """
    metadata = base_data.copy()
    metadata.update(additional_metadata)
    
    # Convert all values to strings for ZenML compatibility
    return {k: str(v) for k, v in metadata.items()}


def _create_basic_data_metadata(df: pd.DataFrame, source: str) -> dict:
    """
    Helper function to create basic metadata for DataFrames.
    
    Args:
        df: DataFrame to analyze
        source: Source description for the data
        
    Returns:
        Dictionary with basic DataFrame metadata
    """
    return {
        "total_records": str(len(df)),
        "columns": str(list(df.columns) if hasattr(df, 'columns') else []),
        "data_shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "source": source,
        "missing_values": str(df.isna().sum().to_dict())
    }


def ensure_datetime_index(df: pd.DataFrame, book_identifier: str = "") -> pd.DataFrame:
    """
    Ensure DataFrame has proper datetime index, restore from 'End Date' if needed.
    
    Args:
        df: DataFrame to process
        book_identifier: Optional identifier for logging (e.g., ISBN or book name)
    
    Returns:
        DataFrame with datetime index
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        identifier_msg = f" for {book_identifier}" if book_identifier else ""
        logger.warning(f"Data{identifier_msg} missing datetime index, attempting to restore")
        if 'End Date' in df.columns:
            df = df.set_index(pd.to_datetime(df['End Date']))
            logger.info(f"Restored datetime index{identifier_msg}")
        else:
            logger.warning(f"No 'End Date' column found{identifier_msg}")
    return df