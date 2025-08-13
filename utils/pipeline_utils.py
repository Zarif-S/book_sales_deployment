"""
General Pipeline Utilities

This module contains general-purpose utility functions for pipeline operations
that are not specific to any particular framework.
"""

import os
import subprocess
import pandas as pd
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility tracking"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass
    return "unknown"


def generate_pipeline_run_name() -> str:
    """Generate descriptive pipeline run name"""
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d-%H_%M_%S_%f')
    return f"book_sales_arima_modeling_pipeline-{timestamp}"


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