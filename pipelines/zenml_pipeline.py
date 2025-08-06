import os
import pandas as pd
import numpy as np
import json
from typing import Tuple, Annotated, Dict, List, Any

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

logger = get_logger(__name__)

# Configure step settings to enable metadata
step_settings = {
    "enable_artifact_metadata": True,
    "enable_artifact_visualization": True,
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
        # Prepare metadata
        metadata_dict = {
            "total_records": str(len(df_isbns)),
            "columns": str(list(df_isbns.columns) if hasattr(df_isbns, 'columns') else []),
            "data_shape": f"{df_isbns.shape[0]} rows x {df_isbns.shape[1]} columns",
            "source": "Google Sheets - ISBN data",
            "missing_values": str(df_isbns.isna().sum().to_dict())
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
        # Prepare metadata (convert all to strings for ZenML compatibility)
        metadata_dict = {
            "total_records": str(len(df_uk_weekly)),
            "columns": str(list(df_uk_weekly.columns) if hasattr(df_uk_weekly, 'columns') else []),
            "data_shape": f"{df_uk_weekly.shape[0]} rows x {df_uk_weekly.shape[1]} columns",
            "source": "Google Sheets - UK weekly data",
            "missing_values": str(df_uk_weekly.isna().sum().to_dict())
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
def analyze_data_quality_step(df_merged: pd.DataFrame) -> Annotated[str, ArtifactConfig(name="data_quality_report")]:
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

        try:
            context = get_step_context()
            context.add_output_metadata(
                output_name="data_quality_report",
                metadata={
                    "quality_score": str(quality_report['quality_score']),
                    "total_records": str(quality_report['total_records']),
                    "total_columns": str(quality_report['total_columns'])
                }
            )
            logger.info("Successfully added data quality metadata")
        except Exception as e:
            logger.error(f"Failed to add data quality metadata: {e}")

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

@step(
    enable_cache=False,  # Disable cache to force re-execution with our fix
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def prepare_modelling_data_step(
    df_merged: pd.DataFrame,
    output_dir: str,
    selected_isbns: List[str] = None,
    column_name: str = 'Volume',
    split_size: int = 32
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
        if selected_isbns is None:
            selected_isbns = [
                '9780722532935',  # The Alchemist
            ]

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

        try:
            context = get_step_context()
            context.add_output_metadata(
                output_name="modelling_data",
                metadata=metadata_dict
            )
            logger.info("Successfully added modelling data metadata")
        except Exception as e:
            logger.error(f"Failed to add modelling data metadata: {e}")

        logger.info(f"Successfully prepared modelling data for {len(prepared_data)} books")

        # Create separate train and test DataFrames
        train_data_list = []
        test_data_list = []

        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                book_isbn = book_isbn_mapping.get(book_name, 'unknown')

                # Add book identifier columns to train data
                train_copy = train_data.copy()
                train_copy['data_type'] = 'train'
                train_data_list.append(train_copy)

                # Add book identifier columns to test data
                test_copy = test_data.copy()
                test_copy['data_type'] = 'test'
                test_data_list.append(test_copy)

        # Combine all books' train data into one DataFrame
        train_df = pd.concat(train_data_list, ignore_index=False) if train_data_list else pd.DataFrame()

        # Combine all books' test data into one DataFrame
        test_df = pd.concat(test_data_list, ignore_index=False) if test_data_list else pd.DataFrame()

        logger.info(f"Created train DataFrame with shape: {train_df.shape}")
        logger.info(f"Created test DataFrame with shape: {test_df.shape}")

        # Save combined train and test data to CSV in data/processed directory
        if not train_df.empty:
            train_csv_path = os.path.join(output_dir, 'combined_train_data.csv')
            train_df.to_csv(train_csv_path, index=True)
            logger.info(f"Saved combined training data to: {train_csv_path}")

        if not test_df.empty:
            test_csv_path = os.path.join(output_dir, 'combined_test_data.csv')
            test_df.to_csv(test_csv_path, index=True)
            logger.info(f"Saved combined test data to: {test_csv_path}")

        # Add metadata for train data
        try:
            context = get_step_context()
            train_metadata = {
                "train_shape": str(train_df.shape),
                "train_books": str(train_df['book_name'].nunique() if not train_df.empty else 0),
                "train_csv_path": train_csv_path if not train_df.empty else "N/A",
                "preparation_timestamp": pd.Timestamp.now().isoformat()
            }
            context.add_output_metadata(
                output_name="train_data",
                metadata=train_metadata
            )

            test_metadata = {
                "test_shape": str(test_df.shape),
                "test_books": str(test_df['book_name'].nunique() if not test_df.empty else 0),
                "test_csv_path": test_csv_path if not test_df.empty else "N/A",
                "preparation_timestamp": pd.Timestamp.now().isoformat()
            }
            context.add_output_metadata(
                output_name="test_data",
                metadata=test_metadata
            )
            logger.info("Successfully added train/test data metadata")
        except Exception as e:
            logger.error(f"Failed to add train/test data metadata: {e}")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Failed to prepare modelling data: {e}")
        raise

# Helper steps to parse JSON outputs if needed
@step
def parse_quality_report_step(quality_report_json: str) -> Dict:
    """Helper step to parse quality report JSON back to dict if needed by downstream steps."""
    return json.loads(quality_report_json)

# ------------------ PIPELINE ------------------ #

@pipeline
def book_sales_data_preparation_pipeline(
    output_dir: str,
    selected_isbns: List[str] = None,
    column_name: str = 'Volume',
    split_size: int = 32
) -> Dict:
    """
    Complete book sales data processing and preparation pipeline.

    This pipeline:
    1. Loads ISBN and UK weekly sales data
    2. Preprocesses and merges the data
    3. Analyzes data quality
    4. Saves processed data
    5. Prepares train/test data for modeling and saves to data/processed
    """
    logger.info("Running book sales data preparation pipeline")

    # Load raw data
    df_isbns = load_isbn_data_step()
    df_uk_weekly = load_uk_weekly_data_step()

    # Preprocess and merge
    df_merged = preprocess_and_merge_step(df_isbns=df_isbns, df_uk_weekly=df_uk_weekly)

    # Analyze data quality (now returns JSON string)
    quality_report_json = analyze_data_quality_step(df_merged=df_merged)

    # Save processed data
    processed_data_path = save_processed_data_step(
        df_merged=df_merged,
        output_dir=output_dir
    )

    # Prepare data for modelling - now returns separate train and test data
    train_data, test_data = prepare_modelling_data_step(
        df_merged=df_merged,
        output_dir=output_dir,
        selected_isbns=selected_isbns,
        column_name=column_name,
        split_size=split_size
    )

    # Optional: Parse JSON outputs back to dicts for pipeline return
    quality_report = parse_quality_report_step(quality_report_json)

    return {
        "df_merged": df_merged,
        "quality_report": quality_report,  # Parsed dict
        "quality_report_json": quality_report_json,  # Original JSON string
        "processed_data_path": processed_data_path,
        "train_data": train_data,  # Training data artifact
        "test_data": test_data,    # Test data artifact
    }

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    # Set up output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'processed')

    # Define default parameters for modeling
    default_selected_isbns = [
        '9780722532935',  # The Alchemist
    ]

    # Run the complete data preparation pipeline
    results = book_sales_data_preparation_pipeline(
        output_dir=output_dir,
        selected_isbns=default_selected_isbns,
        column_name='Volume',
        split_size=32
    )

    print("Data preparation pipeline completed! Train and test data saved to data/processed directory.")