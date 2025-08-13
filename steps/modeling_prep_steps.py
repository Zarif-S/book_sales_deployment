"""
Modeling Preparation Steps

This module contains ZenML step definitions for preparing data for modeling,
including book selection and train/test split operations.
"""

import os
import sys
import json
import pandas as pd
from typing import Tuple, Annotated, Dict, List, Optional
from zenml import step, ArtifactConfig
from zenml.materializers.pandas_materializer import PandasMaterializer
from zenml.logger import get_logger
from utils.zenml_helpers import _add_step_metadata
from utils.pipeline_utils import ensure_datetime_index
from scripts.modelling_prep import prepare_multiple_books_data

# Import seasonality configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'seasonality_analysis'))
try:
    from seasonality_config import SeasonalityConfig  # type: ignore[import-untyped]
except ImportError:
    SeasonalityConfig = None

# Configuration imports
from config.arima_training_config import (
    DEFAULT_TEST_ISBNS,
    DEFAULT_SPLIT_SIZE,
    DEFAULT_MAX_SEASONAL_BOOKS
)

# Initialize logger
logger = get_logger(__name__)


@step(
    enable_cache=True,
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
            isbn_volumes = df_merged.groupby('ISBN')['Volume'].sum().sort_values(ascending=False)  # type: ignore[call-overload]
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
                top_seasonal = seasonal_volumes.sort_values(ascending=False).head(max_books)  # type: ignore[call-overload]
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
    enable_cache=True,
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def create_train_test_splits_step(
    df_merged: pd.DataFrame,
    output_dir: str,
    selected_isbns: Optional[List[str]] = None,
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

        if len(selected_books_data) == 0:
            raise ValueError(f"No data found for selected ISBNs: {selected_isbns}")

        # Group data by ISBN for individual book analysis
        books_data = {}
        book_isbn_mapping = {}

        for isbn in selected_isbns:
            book_data = selected_books_data[selected_books_data['ISBN'] == isbn].copy()
            if len(book_data) > 0:
                # Get book title safely
                if hasattr(book_data, 'columns') and 'Title' in book_data.columns:
                    title_data = book_data['Title']
                    if hasattr(title_data, 'iloc'):
                        book_title = title_data.iloc[0]
                    else:
                        book_title = title_data[0] if len(title_data) > 0 else f"Book_{isbn}"
                else:
                    book_title = f"Book_{isbn}"
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


        logger.info(f"Successfully prepared modelling data for {len(prepared_data)} books")

        # Create consolidated DataFrames for ZenML artifacts (Vertex AI deployment ready)
        # Individual book models will filter these by ISBN for training
        consolidated_train_data = []
        consolidated_test_data = []

        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                # Add book identifiers to enable filtering
                book_isbn = book_isbn_mapping.get(book_name, 'unknown')

                # Prepare train data with identifiers
                train_with_id = train_data.copy()
                train_with_id['ISBN'] = book_isbn
                train_with_id['Title'] = book_name
                consolidated_train_data.append(train_with_id)

                # Prepare test data with identifiers
                test_with_id = test_data.copy()
                test_with_id['ISBN'] = book_isbn
                test_with_id['Title'] = book_name
                consolidated_test_data.append(test_with_id)

        # Combine all books into consolidated DataFrames with proper datetime index preservation
        if consolidated_train_data:
            # Ensure all DataFrames have proper datetime index before concatenation
            for i, df in enumerate(consolidated_train_data):
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    consolidated_train_data[i] = ensure_datetime_index(df, f"train data book {i}")
                    # Remove End Date column if it was used to restore index
                    if 'End Date' in consolidated_train_data[i].columns:
                        consolidated_train_data[i].drop(columns=['End Date'], inplace=True)

            for i, df in enumerate(consolidated_test_data):
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    consolidated_test_data[i] = ensure_datetime_index(df, f"test data book {i}")
                    # Remove End Date column if it was used to restore index
                    if 'End Date' in consolidated_test_data[i].columns:
                        consolidated_test_data[i].drop(columns=['End Date'], inplace=True)

            train_df = pd.concat(consolidated_train_data, ignore_index=False)
            test_df = pd.concat(consolidated_test_data, ignore_index=False)
            logger.info(f"Created consolidated artifacts: train_df shape {train_df.shape}, test_df shape {test_df.shape}")
            logger.info(f"Train index type: {type(train_df.index)}, Test index type: {type(test_df.index)}")
        else:
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()
            logger.warning("No valid data for consolidated DataFrames")

        # Keep CSV files for debugging/development
        logger.info("Individual CSV files available for debugging, consolidated artifacts ready for production")

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

        # Add metadata for consolidated artifacts
        train_metadata = {
            "consolidated_artifacts": "true",
            "deployment_ready": "true",
            "successful_books": str(successful_books),
            "consolidated_train_shape": str(train_df.shape) if len(train_df) > 0 else "empty",
            "books_included": str([book_isbn_mapping.get(name, name) for name in prepared_data.keys() if prepared_data[name][0] is not None]),
            "individual_files_created": str(len(individual_files_created)),
            "file_list": str(individual_files_created),
            "filtering_example": f"train_data[train_data['ISBN'] == '{DEFAULT_TEST_ISBNS[0]}']",
            "preparation_timestamp": pd.Timestamp.now().isoformat()
        }
        _add_step_metadata("train_data", train_metadata)

        test_metadata = {
            "consolidated_artifacts": "true",
            "deployment_ready": "true",
            "successful_books": str(successful_books),
            "consolidated_test_shape": str(test_df.shape) if len(test_df) > 0 else "empty",
            "books_included": str([book_isbn_mapping.get(name, name) for name in prepared_data.keys() if prepared_data[name][1] is not None]),
            "filtering_example": f"test_data[test_data['ISBN'] == '{DEFAULT_TEST_ISBNS[0]}']",
            "note": "Filter consolidated artifacts by ISBN for individual book modeling",
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