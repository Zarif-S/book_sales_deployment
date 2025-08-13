"""
Data Loading Steps

This module contains ZenML step definitions for loading data from various sources.
"""

import pandas as pd
from typing import Annotated
from zenml import step, ArtifactConfig
from zenml.materializers.pandas_materializer import PandasMaterializer
from zenml.logger import get_logger
from utils.zenml_helpers import _add_step_metadata, _create_basic_data_metadata

# Initialize logger
logger = get_logger(__name__)


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