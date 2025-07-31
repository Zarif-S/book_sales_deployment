import os
import pandas as pd
from typing import Tuple, Annotated, Dict, List

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

@step(
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def prepare_modelling_data_step(
    df_merged: pd.DataFrame,
    selected_isbns: List[str] = None,
    column_name: str = 'Volume',
    split_size: int = 32
) -> Annotated[pd.DataFrame, ArtifactConfig(name="modelling_data")]:
    """
    Prepare data for ARIMA modeling by splitting into train/test sets for selected books.
    
    Args:
        df_merged: Merged and processed DataFrame
        selected_isbns: List of ISBNs to prepare data for (defaults to common books)
        column_name: Column to use for time series analysis (default: 'Volume')
        split_size: Number of entries to include in test set (default: 32 weeks)
    
    Returns:
        Dictionary containing train/test data for each book
    """
    logger.info("Starting modelling data preparation")
    
    try:
        # Use default ISBNs if none provided
        if selected_isbns is None:
            selected_isbns = [
                '9780722532935',  # The Alchemist
                '9780241003008'   # The Very Hungry Caterpillar
            ]
        
        logger.info(f"Preparing modelling data for {len(selected_isbns)} books: {selected_isbns}")
        logger.info(f"Using column: {column_name}, split size: {split_size}")
        
        # Debug: Check data types and available ISBNs
        logger.info(f"ISBN column dtype: {df_merged['ISBN'].dtype}")
        logger.info(f"Sample ISBNs in data: {df_merged['ISBN'].head().tolist()}")
        logger.info(f"Total unique ISBNs in data: {df_merged['ISBN'].nunique()}")
        
        # Ensure ISBNs are strings throughout the pipeline
        # This is important for consistency and to avoid type issues downstream
        if df_merged['ISBN'].dtype != 'object':
            logger.info("Converting ISBN column to string type")
            df_merged['ISBN'] = df_merged['ISBN'].astype(str)
        
        # Keep selected ISBNs as strings (they should already be strings)
        logger.info(f"Using ISBNs as strings: {selected_isbns}")
        
        # Filter data for selected ISBNs
        selected_books_data = df_merged[df_merged['ISBN'].isin(selected_isbns)].copy()
        
        if selected_books_data.empty:
            raise ValueError(f"No data found for selected ISBNs: {selected_isbns}")
        
        # Group data by ISBN for individual book analysis
        books_data = {}
        book_isbn_mapping = {}  # Create mapping from book title to ISBN
        for isbn in selected_isbns:
            book_data = selected_books_data[selected_books_data['ISBN'] == isbn].copy()
            if not book_data.empty:
                # Get book title for better identification
                book_title = book_data['Title'].iloc[0] if 'Title' in book_data.columns else f"Book_{isbn}"
                books_data[book_title] = book_data
                book_isbn_mapping[book_title] = isbn  # Store the mapping
                logger.info(f"Found data for {book_title} (ISBN: {isbn}): {len(book_data)} records")
            else:
                logger.warning(f"No data found for ISBN: {isbn}")
        
        if not books_data:
            raise ValueError("No valid book data found for any of the selected ISBNs")
        
        # Prepare train/test data for each book
        prepared_data = prepare_multiple_books_data(
            books_data=books_data,
            column_name=column_name,
            split_size=split_size
        )
        
        # Create metadata for the step
        metadata_dict = {
            "selected_isbns": selected_isbns,
            "column_name": column_name,
            "split_size": split_size,
            "books_processed": list(prepared_data.keys()),
            "total_books": len(prepared_data),
            "successful_preparations": sum(1 for train, test in prepared_data.values() if train is not None and test is not None),
            "preparation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Add detailed info for each book
        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                metadata_dict[f"{book_name}_train_shape"] = train_data.shape[0]
                metadata_dict[f"{book_name}_test_shape"] = test_data.shape[0]
                metadata_dict[f"{book_name}_train_range"] = f"{train_data.index.min()} to {train_data.index.max()}"
                metadata_dict[f"{book_name}_test_range"] = f"{test_data.index.min()} to {test_data.index.max()}"
        
        logger.info(f"Modelling data preparation metadata: {metadata_dict}")
        
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
        
        # Create a DataFrame for visualization that contains the train/test data
        visualization_data = []
        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                # Get the correct ISBN for this book (already a string)
                book_isbn = book_isbn_mapping.get(book_name, 'unknown')
                
                # Add train data
                for date, value in train_data.items():
                    visualization_data.append({
                        'book_name': book_name,
                        'date': date,
                        'value': value,
                        'data_type': 'train',
                        'isbn': book_isbn
                    })
                
                # Add test data
                for date, value in test_data.items():
                    visualization_data.append({
                        'book_name': book_name,
                        'date': date,
                        'value': value,
                        'data_type': 'test',
                        'isbn': book_isbn
                    })
        
        # Create DataFrame for visualization
        viz_df = pd.DataFrame(visualization_data)
        if not viz_df.empty:
            viz_df['date'] = pd.to_datetime(viz_df['date'])
            viz_df = viz_df.sort_values(['book_name', 'date'])
        
        logger.info(f"Created visualization DataFrame with {len(viz_df)} rows")
        return viz_df
        
    except Exception as e:
        logger.error(f"Failed to prepare modelling data: {e}")
        raise

# ------------------ PIPELINE ------------------ #

@pipeline
def book_sales_pipeline_with_modelling_prep(
    output_dir: str,
    selected_isbns: List[str] = None,
    column_name: str = 'Volume',
    split_size: int = 32
) -> Dict:
    """
    Book sales data processing pipeline with modelling preparation.
    
    This pipeline extends the basic processing pipeline by adding a step
    to prepare data specifically for ARIMA modeling.
    """
    logger.info("Running pipeline: book_sales_pipeline_with_modelling_prep")
    
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
    
    # Prepare data for modelling
    modelling_data = prepare_modelling_data_step(
        df_merged=df_merged,
        selected_isbns=selected_isbns,
        column_name=column_name,
        split_size=split_size
    )
    
    return {
        "df_merged": df_merged,
        "quality_report": quality_report,
        "processed_data_path": processed_data_path,
        "modelling_data": modelling_data,
    }

# ------------------ MAIN ------------------ #

if __name__ == "__main__":
    # Set up output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'processed')

    # Define default parameters for modelling preparation
    default_selected_isbns = [
        '9780722532935',  # The Alchemist
        '9780241003008'   # The Very Hungry Caterpillar
    ]

    # Run the pipeline
    book_sales_pipeline_with_modelling_prep(
        output_dir=output_dir,
        selected_isbns=default_selected_isbns,
        column_name='Volume',
        split_size=32
    )
    print("Pipeline with modelling preparation run submitted! Check the ZenML dashboard for outputs.") 