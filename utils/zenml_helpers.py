"""
ZenML Helper Utilities

This module contains ZenML-specific utility functions for metadata management
and step operations.
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