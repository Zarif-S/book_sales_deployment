"""
Data preprocessing module for book sales analysis.

This module handles the preprocessing of weekly book sales data including:
- Data cleaning and type conversion
- Resampling to fill missing weeks
- Filtering data by date ranges
- Selecting specific books for analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters."""
    # Default ISBNs for analysis
    DEFAULT_SELECTED_ISBNS: List[str] = None
    
    # Default dates
    DEFAULT_CUTOFF_DATE: str = '2024-07-01'
    DEFAULT_START_DATE: str = '2012-01-01'
    
    # Required columns
    REQUIRED_COLUMNS: List[str] = None
    
    # Column mappings for resampling
    NUMERIC_COLUMNS: List[str] = None
    CATEGORICAL_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.DEFAULT_SELECTED_ISBNS is None:
            self.DEFAULT_SELECTED_ISBNS = [
                '9780722532935',  # The Alchemist
                '9780241003008',  # The Very Hungry Caterpillar
                '9780140500875'   # The Very Hungry Caterpillar (different edition)
            ]
        
        if self.REQUIRED_COLUMNS is None:
            self.REQUIRED_COLUMNS = ['ISBN', 'End Date', 'Volume', 'Value', 'Title']
        
        if self.NUMERIC_COLUMNS is None:
            self.NUMERIC_COLUMNS = ['Value', 'ASP', 'RRP', 'Volume']
        
        if self.CATEGORICAL_COLUMNS is None:
            self.CATEGORICAL_COLUMNS = [
                'Title', 'Author', 'Binding', 'Imprint', 
                'Publisher Group', 'Product Class', 'Source'
            ]

# Global config instance
config = PreprocessingConfig()

def get_isbn_to_title_mapping() -> Dict[str, str]:
    """
    Get mapping of ISBNs to book titles for the default selected books.
    
    Returns:
        Dictionary mapping ISBNs to titles
    """
    return {
        '9780722532935': 'Alchemist, The',
        '9780241003008': 'Very Hungry Caterpillar, The',
        '9780140500875': 'Very Hungry Caterpillar, The'
    }

def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        The directory path
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def get_project_directories() -> Dict[str, str]:
    """
    Get standardized project directory paths.
    
    Returns:
        Dictionary with directory paths
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'project_root': project_root,
        'raw_data': os.path.join(project_root, 'data', 'raw'),
        'processed_data': os.path.join(project_root, 'data', 'processed')
    }

def validate_required_columns(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If required columns are missing
    """
    if required_columns is None:
        required_columns = config.REQUIRED_COLUMNS
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def validate_date_format(date_string: str) -> bool:
    """
    Validate that a date string is in YYYY-MM-DD format.
    
    Args:
        date_string: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def load_csv_data_from_raw() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load CSV data directly from the data/raw directory.
    
    Returns:
        Tuple of (df_isbns, df_uk_weekly) or (None, None) if files not found
    """
    logger.info("Loading CSV data from data/raw directory...")
    
    directories = get_project_directories()
    raw_data_dir = directories['raw_data']
    
    def load_csv_file(file_path: str, description: str) -> Optional[pd.DataFrame]:
        """Helper function to load a single CSV file."""
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {description}: {df.shape}")
            return df
        else:
            logger.warning(f"{description} CSV file not found: {file_path}")
            return None
    
    # Load ISBN data
    isbn_csv_path = os.path.join(raw_data_dir, 'ISBN_data.csv')
    df_isbns = load_csv_file(isbn_csv_path, "ISBN data")
    
    # Load UK weekly data
    uk_csv_path = os.path.join(raw_data_dir, 'UK_weekly_data.csv')
    df_uk_weekly = load_csv_file(uk_csv_path, "UK weekly data")
    
    return df_isbns, df_uk_weekly

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types for proper processing with validation.
    
    Args:
        df: Raw sales data DataFrame
        
    Returns:
        DataFrame with converted data types
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Converting data types...")
    
    # Validate required columns exist
    validate_required_columns(df)
    
    df_converted = df.copy()
    
    # Convert ISBN to string
    df_converted['ISBN'] = df_converted['ISBN'].astype(str)
    
    # Convert date to datetime if not already
    if 'End Date' in df_converted.columns:
        try:
            df_converted['End Date'] = pd.to_datetime(df_converted['End Date'])
            logger.info(f"Converted 'End Date' to datetime. Date range: {df_converted['End Date'].min()} to {df_converted['End Date'].max()}")
        except Exception as e:
            logger.error(f"Failed to convert 'End Date' to datetime: {e}")
            raise
    
    # Convert numeric columns
    for col in config.NUMERIC_COLUMNS:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
    
    return df_converted

def prepare_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for time series analysis by setting date as index and sorting.
    
    Args:
        df: DataFrame with 'End Date' column or already having 'End Date' as index
        
    Returns:
        DataFrame with 'End Date' as index, sorted chronologically
    """
    logger.info("Preparing time series data...")
    
    df_ts = df.copy()
    
    # Check if 'End Date' is already the index
    if df_ts.index.name == 'End Date':
        logger.info("'End Date' is already the index")
    else:
        # Set 'End Date' as index
        if 'End Date' not in df_ts.columns:
            raise ValueError("'End Date' column not found in DataFrame")
        df_ts.set_index('End Date', inplace=True)
    
    # Sort by index
    df_ts.sort_index(inplace=True)
    logger.info(f"Time series data prepared. Date range: {df_ts.index.min()} to {df_ts.index.max()}")
    
    return df_ts

def create_resampling_aggregation_dict() -> Dict[str, str]:
    """
    Create aggregation dictionary for resampling operations.
    
    Returns:
        Dictionary mapping column names to aggregation functions
    """
    agg_dict = {}
    
    # Numeric columns - use mean
    for col in config.NUMERIC_COLUMNS:
        agg_dict[col] = 'mean'
    
    # Categorical columns - use first
    for col in config.CATEGORICAL_COLUMNS:
        agg_dict[col] = 'first'
    
    return agg_dict

def resample_group_data(group: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a single group by ISBN with proper aggregation.
    
    Args:
        group: DataFrame group for a single ISBN
        
    Returns:
        Resampled DataFrame
    """
    agg_dict = create_resampling_aggregation_dict()
    
    # Only include columns that exist in the group
    available_agg_dict = {col: func for col, func in agg_dict.items() if col in group.columns}
    
    if not available_agg_dict:
        logger.warning("No columns available for aggregation")
        return group.resample('W-SAT').first()
    
    return group.resample('W-SAT').agg(available_agg_dict)

def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing weeks with 0 sales to ensure continuous time series.
    
    Args:
        df: DataFrame with 'End Date' as index
        
    Returns:
        DataFrame with missing weeks filled with 0
    """
    logger.info("Filling missing weeks with 0 sales...")
    
    if 'ISBN' not in df.columns:
        raise ValueError("'ISBN' column not found in DataFrame")
    
    completed_dfs = []
    total_groups = df['ISBN'].nunique()
    
    for i, (isbn, group) in enumerate(df.groupby('ISBN')):
        if i % 100 == 0:  # Progress logging
            logger.info(f"Processing ISBN group {i+1}/{total_groups}")
        
        # Fill missing weeks with 0 for numeric columns, forward fill for others
        group_filled = group.asfreq('W-SAT')
        
        # Fill numeric columns with 0
        for col in config.NUMERIC_COLUMNS:
            if col in group_filled.columns:
                group_filled[col] = group_filled[col].fillna(0)
        
        # Forward fill categorical columns
        for col in config.CATEGORICAL_COLUMNS:
            if col in group_filled.columns:
                group_filled[col] = group_filled[col].fillna(method='ffill')
        
        # Ensure ISBN is preserved
        group_filled['ISBN'] = isbn
        completed_dfs.append(group_filled)
    
    df_filled = pd.concat(completed_dfs)
    logger.info(f"Filled missing weeks. Final shape: {df_filled.shape}")
    return df_filled

def resample_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample data to weekly frequency with proper aggregation.
    
    Args:
        df: DataFrame with filled missing weeks
        
    Returns:
        Resampled DataFrame
    """
    logger.info("Resampling data to weekly frequency...")
    
    # Apply resampling to each ISBN group
    weekly_resampled = df.groupby('ISBN').apply(resample_group_data).reset_index()
    
    logger.info(f"Resampled data shape: {weekly_resampled.shape}")
    return weekly_resampled

def filter_data_by_date(df: pd.DataFrame, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Filter data by date range with validation.
    
    Args:
        df: DataFrame with datetime index
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (optional)
        
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If date format is invalid
    """
    logger.info(f"Filtering data from {start_date} to {end_date or 'end'}")
    
    # Validate date formats
    if not validate_date_format(start_date):
        raise ValueError(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD")
    
    if end_date and not validate_date_format(end_date):
        raise ValueError(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD")
    
    # Apply date filter
    if end_date:
        mask = (df.index >= start_date) & (df.index <= end_date)
    else:
        mask = df.index >= start_date
    
    filtered_df = df[mask]
    logger.info(f"Filtered data shape: {filtered_df.shape}")
    return filtered_df

def get_isbns_beyond_date(df: pd.DataFrame, cutoff_date: str = None) -> List[str]:
    """
    Get ISBNs that have sales data beyond a specific date.
    
    Args:
        df: DataFrame with sales data
        cutoff_date: Date cutoff in 'YYYY-MM-DD' format (defaults to config value)
        
    Returns:
        List of ISBNs with data beyond the cutoff date
        
    Raises:
        ValueError: If date format is invalid
    """
    if cutoff_date is None:
        cutoff_date = config.DEFAULT_CUTOFF_DATE
    
    logger.info(f"Finding ISBNs with data beyond {cutoff_date}")
    
    # Validate date format
    if not validate_date_format(cutoff_date):
        raise ValueError(f"Invalid cutoff_date format: {cutoff_date}. Expected YYYY-MM-DD")
    
    filtered_data = df[df.index >= cutoff_date]
    unique_isbns = filtered_data['ISBN'].unique().tolist()
    
    logger.info(f"Found {len(unique_isbns)} ISBNs with data beyond {cutoff_date}")
    return unique_isbns

def get_book_summary(df: pd.DataFrame, book_titles: List[str]) -> pd.DataFrame:
    """
    Get summary statistics for specific books.
    
    Args:
        df: DataFrame with sales data
        book_titles: List of book titles to filter by
        
    Returns:
        DataFrame with book summary statistics
    """
    logger.info(f"Getting summary for {len(book_titles)} books")
    
    if 'Title' not in df.columns:
        raise ValueError("'Title' column not found in DataFrame")
    
    # Filter for books with the specified titles, case-insensitive
    pattern = '|'.join(book_titles)
    filtered_books = df[df['Title'].str.contains(pattern, case=False, na=False)]
    
    if filtered_books.empty:
        logger.warning("No books found matching the specified titles")
        return pd.DataFrame()
    
    # Define aggregation based on available columns
    agg_dict = {
        'ISBN': 'size',  # Count the number of rows (occurrences)
    }
    
    # Add numeric aggregations if columns exist
    if 'Volume' in filtered_books.columns:
        agg_dict['Volume_Sum'] = ('Volume', 'sum')
    if 'Value' in filtered_books.columns:
        agg_dict['Value_Sum'] = ('Value', 'sum')
    if 'ASP' in filtered_books.columns:
        agg_dict['ASP_Avg'] = ('ASP', 'mean')
    
    # Group by available columns
    group_cols = ['ISBN', 'Title']
    if 'Binding' in filtered_books.columns:
        group_cols.append('Binding')
    if 'RRP' in filtered_books.columns:
        group_cols.append('RRP')
    
    book_summary = filtered_books.groupby(group_cols).agg(**agg_dict).reset_index()
    book_summary.rename(columns={'ISBN': 'Count'}, inplace=True)
    
    logger.info(f"Generated summary for {len(book_summary)} unique book variations")
    return book_summary

def select_specific_books(df: pd.DataFrame, isbn_list: Optional[List[str]] = None, 
                         start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Select specific books by ISBN for detailed analysis.
    
    Args:
        df: DataFrame with sales data
        isbn_list: List of ISBNs to select (defaults to config value)
        start_date: Start date for filtering in 'YYYY-MM-DD' format (defaults to config value)
        
    Returns:
        Filtered DataFrame with selected books
        
    Raises:
        ValueError: If date format is invalid
    """
    if isbn_list is None:
        isbn_list = config.DEFAULT_SELECTED_ISBNS
    
    if start_date is None:
        start_date = config.DEFAULT_START_DATE
    
    logger.info(f"Selecting {len(isbn_list)} books from {start_date}")
    
    # Validate date format
    if not validate_date_format(start_date):
        raise ValueError(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD")
    
    if 'ISBN' not in df.columns:
        raise ValueError("'ISBN' column not found in DataFrame")
    
    filtered_data = df[
        (df['ISBN'].isin(isbn_list)) & 
        (df.index >= start_date)
    ]
    
    logger.info(f"Selected books data shape: {filtered_data.shape}")
    return filtered_data

def aggregate_yearly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly data to yearly data.
    
    Args:
        df: DataFrame with weekly sales data
        
    Returns:
        DataFrame with yearly aggregated data
    """
    logger.info("Aggregating data to yearly frequency...")
    
    if 'Volume' not in df.columns:
        raise ValueError("'Volume' column not found in DataFrame")
    
    yearly_data = df.groupby([df.index.year, 'ISBN'])['Volume'].sum().reset_index()
    yearly_data.rename(columns={'End Date': 'Year'}, inplace=True)
    
    logger.info(f"Yearly aggregated data shape: {yearly_data.shape}")
    return yearly_data

def save_processed_data(df: pd.DataFrame, filename: str, processed_dir: Optional[str] = None) -> str:
    """
    Save processed DataFrame to the processed data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file to save (without extension)
        processed_dir: Directory to save to (optional, defaults to data/processed)
        
    Returns:
        Path to the saved file
    """
    if processed_dir is None:
        directories = get_project_directories()
        processed_dir = directories['processed_data']
    
    # Ensure the processed directory exists
    ensure_directory_exists(processed_dir)
    
    # Save as CSV
    csv_path = os.path.join(processed_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=True if df.index.name else False)
    logger.info(f"Saved processed data to: {csv_path}")
    
    return csv_path

def load_processed_data(filename: str, processed_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load processed DataFrame from the processed data directory.
    
    Args:
        filename: Name of the file to load (without extension)
        processed_dir: Directory to load from (optional, defaults to data/processed)
        
    Returns:
        Loaded DataFrame or None if file not found
    """
    if processed_dir is None:
        directories = get_project_directories()
        processed_dir = directories['processed_data']
    
    # Construct file path
    csv_path = os.path.join(processed_dir, f"{filename}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded processed data from: {csv_path}")
        return df
    else:
        logger.warning(f"Processed data file not found: {csv_path}")
        return None

def analyze_missing_values(df: pd.DataFrame, dataset_name: str) -> Dict[str, int]:
    """
    Analyze and log missing values in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary with missing value counts per column
    """
    logger.info(f"Analyzing missing values in {dataset_name}")
    
    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    
    logger.info(f"{dataset_name} shape: {df.shape}")
    logger.info(f"Total missing values: {total_missing}")
    
    if total_missing > 0:
        missing_cols = [col for col, count in missing_values.items() if count > 0]
        logger.info(f"Columns with missing values: {missing_cols}")
        for col, count in missing_values.items():
            if count > 0:
                percentage = (count / len(df)) * 100
                logger.info(f"  {col}: {count} ({percentage:.2f}%)")
    else:
        logger.info(f"No missing values found in {dataset_name}")
    
    return missing_values

def merge_and_fill_author_data(df_uk_weekly: pd.DataFrame, df_isbns: pd.DataFrame) -> pd.DataFrame:
    """
    Merge UK weekly data with ISBN data and fill missing author values.
    
    Args:
        df_uk_weekly: UK weekly sales data
        df_isbns: ISBN reference data
        
    Returns:
        Merged DataFrame with filled author data
    """
    logger.info("Merging UK weekly data with ISBN data and filling missing authors...")
    
    # Validate required columns
    if 'ISBN' not in df_uk_weekly.columns or 'ISBN' not in df_isbns.columns:
        raise ValueError("'ISBN' column not found in one or both DataFrames")
    
    if 'Author' not in df_isbns.columns:
        logger.warning("'Author' column not found in ISBN data")
        return df_uk_weekly
    
    # Merge data
    df_merged = df_uk_weekly.merge(
        df_isbns[['ISBN', 'Author']], 
        on='ISBN', 
        how='left', 
        suffixes=('', '_ISBN')
    )
    
    # Fill missing author values
    if 'Author' in df_merged.columns and 'Author_ISBN' in df_merged.columns:
        missing_authors_before = df_merged['Author'].isnull().sum()
        df_merged['Author'] = df_merged['Author'].fillna(df_merged['Author_ISBN'])
        missing_authors_after = df_merged['Author'].isnull().sum()
        
        filled_count = missing_authors_before - missing_authors_after
        logger.info(f"Filled {filled_count} missing author values")
        
        # Drop the temporary column
        df_merged = df_merged.drop(columns=['Author_ISBN'])
    
    logger.info(f"Merged data shape: {df_merged.shape}")
    return df_merged

def display_data_summary(df: pd.DataFrame) -> Dict[str, Union[int, List[str]]]:
    """
    Display and return summary information about the data.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating data summary...")
    
    summary = {
        'total_records': len(df),
        'unique_isbns': df['ISBN'].nunique() if 'ISBN' in df.columns else 0,
        'unique_titles': df['Title'].nunique() if 'Title' in df.columns else 0,
        'date_range': (df.index.min(), df.index.max()) if hasattr(df.index, 'min') else None,
        'columns': df.columns.tolist()
    }
    
    if 'Title' in df.columns:
        unique_titles = df['Title'].unique()
        summary['sample_titles'] = unique_titles[:10].tolist()
        logger.info(f"Unique titles: {len(unique_titles)}")
        logger.info(f"Sample titles: {summary['sample_titles']}")
    
    logger.info(f"Data summary: {summary}")
    return summary

def create_data_backups(df_isbns: pd.DataFrame, df_uk_weekly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create backups of raw data.
    
    Args:
        df_isbns: ISBN data
        df_uk_weekly: UK weekly data
        
    Returns:
        Tuple of backup DataFrames
    """
    logger.info("Creating data backups...")
    return df_isbns.copy(), df_uk_weekly.copy()

def create_processing_summary(df_filled: pd.DataFrame, selected_books_data: pd.DataFrame, 
                            isbns_beyond_cutoff: List[str], cutoff_date: str) -> Dict:
    """
    Create a summary of the processing pipeline results.
    
    Args:
        df_filled: Processed filled data
        selected_books_data: Selected books data
        isbns_beyond_cutoff: List of ISBNs beyond cutoff date
        cutoff_date: Cutoff date used
        
    Returns:
        Dictionary with processing summary
    """
    return {
        'total_records_processed': len(df_filled),
        'unique_isbns': df_filled['ISBN'].nunique(),
        'date_range': (df_filled.index.min(), df_filled.index.max()),
        'selected_books_records': len(selected_books_data),
        'isbns_beyond_cutoff': len(isbns_beyond_cutoff),
        'cutoff_date_used': cutoff_date,
        'processing_timestamp': datetime.now().isoformat()
    }

def save_processing_summary(summary: Dict, processed_dir: Optional[str] = None) -> str:
    """
    Save processing summary to a text file.
    
    Args:
        summary: Processing summary dictionary
        processed_dir: Directory to save to
        
    Returns:
        Path to saved summary file
    """
    if processed_dir is None:
        directories = get_project_directories()
        processed_dir = directories['processed_data']
    
    ensure_directory_exists(processed_dir)
    summary_path = os.path.join(processed_dir, "processing_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Data Processing Summary\n")
        f.write("=" * 30 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved processing summary to: {summary_path}")
    return summary_path

def preprocess_sales_data(df_raw: pd.DataFrame, 
                         selected_isbns: Optional[List[str]] = None,
                         start_date: Optional[str] = None,
                         cutoff_date: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function that orchestrates the entire data preprocessing pipeline.
    
    Args:
        df_raw: Raw sales data DataFrame
        selected_isbns: List of ISBNs to select for analysis (defaults to config value)
        start_date: Start date for selected books analysis (defaults to config value)
        cutoff_date: Cutoff date for finding ISBNs (defaults to config value)
        save_outputs: Whether to save processed outputs to data/processed directory
        
    Returns:
        Tuple of (processed_data, filtered_data, selected_books_data)
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Use defaults if not provided
    if selected_isbns is None:
        selected_isbns = config.DEFAULT_SELECTED_ISBNS
    if start_date is None:
        start_date = config.DEFAULT_START_DATE
    if cutoff_date is None:
        cutoff_date = config.DEFAULT_CUTOFF_DATE
    
    # Step 1: Convert data types
    df = convert_data_types(df_raw.copy())
    
    # Step 2: Prepare time series data
    df = prepare_time_series_data(df)
    
    # Step 3: Fill missing weeks
    df_filled = fill_missing_weeks(df)
    
    # Step 4: Get ISBNs with data beyond cutoff date
    isbns_beyond_cutoff = get_isbns_beyond_date(df_filled, cutoff_date)
    
    # Step 5: Filter data for selected books
    selected_books_data = select_specific_books(df_filled, selected_isbns, start_date)
    
    # Step 6: Save processed outputs if requested
    if save_outputs:
        logger.info("Saving processed outputs to data/processed directory...")
        
        # Save the filled data (main processed dataset)
        save_processed_data(df_filled, "processed_sales_data_filled")
        
        # Save the selected books data
        save_processed_data(selected_books_data, "selected_books_data")
        
        # Create and save processing summary
        summary = create_processing_summary(df_filled, selected_books_data, isbns_beyond_cutoff, cutoff_date)
        save_processing_summary(summary)
    
    logger.info("Data preprocessing completed successfully")
    logger.info(f"Processed data shape: {df_filled.shape}")
    logger.info(f"Selected books data shape: {selected_books_data.shape}")
    
    return df_filled, df_filled, selected_books_data

def preprocess_loaded_data(df_isbns: pd.DataFrame, df_uk_weekly: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Run the preprocessing pipeline on loaded data with analysis and merging.
    
    Args:
        df_isbns: ISBN reference data
        df_uk_weekly: UK weekly sales data
        
    Returns:
        Dictionary with processed DataFrames
    """
    logger.info("Starting preprocessing of loaded data...")
    
    # Analyze missing values
    analyze_missing_values(df_isbns, "ISBN")
    analyze_missing_values(df_uk_weekly, "UK Weekly")
    
    # Merge and fill author data
    df_uk_weekly_filled = merge_and_fill_author_data(df_uk_weekly, df_isbns)

    # Ensure 'End Date' is datetime and set as index
    if 'End Date' in df_uk_weekly_filled.columns:
        df_uk_weekly_filled = df_uk_weekly_filled.copy()
        df_uk_weekly_filled['End Date'] = pd.to_datetime(df_uk_weekly_filled['End Date'])
        df_uk_weekly_filled = df_uk_weekly_filled.set_index('End Date')
    
    # Final missing value analysis
    logger.info("Analyzing remaining missing values after filling 'Author':")
    analyze_missing_values(df_uk_weekly_filled, "UK Weekly (after author fill)")
    
    # Display data summary
    display_data_summary(df_uk_weekly_filled)
    
    # Create data backups
    df_isbns_raw, df_uk_weekly_raw = create_data_backups(df_isbns, df_uk_weekly)
    
    logger.info("Preprocessing of loaded data completed successfully")
    
    return {
        'df_isbns': df_isbns,
        'df_uk_weekly': df_uk_weekly_filled,
        'df_isbns_raw': df_isbns_raw,
        'df_uk_weekly_raw': df_uk_weekly_raw
    }

def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    logger.info("Generating comprehensive data info...")
    
    info = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Add date range if index is datetime
    if hasattr(df.index, 'min') and pd.api.types.is_datetime64_any_dtype(df.index):
        info['date_range'] = (df.index.min(), df.index.max())
    
    # Add unique counts for key columns
    if 'ISBN' in df.columns:
        info['unique_isbns'] = df['ISBN'].nunique()
    
    if 'Title' in df.columns:
        info['unique_titles'] = df['Title'].nunique()
    
    # Missing values summary
    missing_values = df.isnull().sum()
    info['missing_values'] = missing_values.to_dict()
    info['total_missing'] = missing_values.sum()
    info['missing_percentage'] = (missing_values.sum() / (df.shape[0] * df.shape[1])) * 100
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        info['numeric_columns'] = numeric_cols
        info['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    logger.info(f"Dataset info generated: {df.shape[0]} rows, {df.shape[1]} columns, {info['memory_usage_mb']:.2f} MB")
    
    return info

def validate_preprocessing_inputs(selected_isbns: Optional[List[str]] = None,
                                start_date: Optional[str] = None,
                                cutoff_date: Optional[str] = None) -> Tuple[List[str], str, str]:
    """
    Validate and return preprocessing input parameters with defaults.
    
    Args:
        selected_isbns: List of ISBNs to select
        start_date: Start date string
        cutoff_date: Cutoff date string
        
    Returns:
        Tuple of validated (selected_isbns, start_date, cutoff_date)
        
    Raises:
        ValueError: If any inputs are invalid
    """
    # Use defaults if not provided
    if selected_isbns is None:
        selected_isbns = config.DEFAULT_SELECTED_ISBNS.copy()
    
    if start_date is None:
        start_date = config.DEFAULT_START_DATE
    
    if cutoff_date is None:
        cutoff_date = config.DEFAULT_CUTOFF_DATE
    
    # Validate ISBNs
    if not isinstance(selected_isbns, list) or not selected_isbns:
        raise ValueError("selected_isbns must be a non-empty list")
    
    # Validate date formats
    if not validate_date_format(start_date):
        raise ValueError(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD")
    
    if not validate_date_format(cutoff_date):
        raise ValueError(f"Invalid cutoff_date format: {cutoff_date}. Expected YYYY-MM-DD")
    
    # Validate date logic
    if start_date >= cutoff_date:
        logger.warning(f"start_date ({start_date}) is not before cutoff_date ({cutoff_date})")
    
    return selected_isbns, start_date, cutoff_date

def run_full_preprocessing_pipeline(df_raw: pd.DataFrame,
                                  selected_isbns: Optional[List[str]] = None,
                                  start_date: Optional[str] = None,
                                  cutoff_date: Optional[str] = None,
                                  save_outputs: bool = True,
                                  create_backups: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline with all steps and validations.
    
    Args:
        df_raw: Raw sales data DataFrame
        selected_isbns: List of ISBNs to select for analysis
        start_date: Start date for selected books analysis
        cutoff_date: Cutoff date for finding ISBNs
        save_outputs: Whether to save processed outputs
        create_backups: Whether to create backup copies of input data
        
    Returns:
        Dictionary containing all processed DataFrames and metadata
    """
    logger.info("=" * 50)
    logger.info("STARTING FULL PREPROCESSING PIPELINE")
    logger.info("=" * 50)
    
    # Validate inputs
    selected_isbns, start_date, cutoff_date = validate_preprocessing_inputs(
        selected_isbns, start_date, cutoff_date
    )
    
    logger.info(f"Pipeline parameters:")
    logger.info(f"  Selected ISBNs: {selected_isbns}")
    logger.info(f"  Start date: {start_date}")
    logger.info(f"  Cutoff date: {cutoff_date}")
    logger.info(f"  Save outputs: {save_outputs}")
    
    try:
        # Create backup if requested
        df_raw_backup = df_raw.copy() if create_backups else None
        
        # Run main preprocessing
        df_filled, _, selected_books_data = preprocess_sales_data(
            df_raw=df_raw,
            selected_isbns=selected_isbns,
            start_date=start_date,
            cutoff_date=cutoff_date,
            save_outputs=save_outputs
        )
        
        # Generate comprehensive data info
        raw_info = get_data_info(df_raw)
        processed_info = get_data_info(df_filled)
        selected_info = get_data_info(selected_books_data)
        
        # Prepare results
        results = {
            'processed_data': df_filled,
            'selected_books_data': selected_books_data,
            'raw_data_info': raw_info,
            'processed_data_info': processed_info,
            'pipeline_config': {
                'selected_isbns': selected_isbns,
                'start_date': start_date,
                'cutoff_date': cutoff_date,
                'save_outputs': save_outputs
            }
        }
        
        # Add backup if created
        if create_backups:
            results['raw_data_backup'] = df_raw_backup
        
        logger.info("=" * 50)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Raw data shape: {df_raw.shape}")
        logger.info(f"Processed data shape: {df_filled.shape}")
        logger.info(f"Selected books data shape: {selected_books_data.shape}")
        
        return results
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise

# Convenience functions for common use cases
def quick_preprocess(df_raw: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Quick preprocessing with default settings.
    
    Args:
        df_raw: Raw data DataFrame
        save: Whether to save outputs
        
    Returns:
        Processed DataFrame
    """
    logger.info("Running quick preprocessing with default settings...")
    processed_data, _, _ = preprocess_sales_data(df_raw, save_outputs=save)
    return processed_data

def preprocess_for_books(df_raw: pd.DataFrame, book_isbns: List[str], 
                        from_date: str = None, save: bool = True) -> pd.DataFrame:
    """
    Preprocess data focusing on specific books.
    
    Args:
        df_raw: Raw data DataFrame
        book_isbns: List of ISBNs to focus on
        from_date: Start date for analysis
        save: Whether to save outputs
        
    Returns:
        Selected books DataFrame
    """
    logger.info(f"Preprocessing data for {len(book_isbns)} specific books...")
    _, _, selected_books = preprocess_sales_data(
        df_raw, 
        selected_isbns=book_isbns,
        start_date=from_date,
        save_outputs=save
    )
    return selected_books

if __name__ == "__main__":
    """
    Main execution block for testing the preprocessing functions
    when the script is run directly.
    """
    logger.info("Preprocessing module loaded successfully")
    
    # Load raw data and run preprocessing pipeline
    try:
        df_isbns, df_uk_weekly = load_csv_data_from_raw()
        
        if df_uk_weekly is not None:
            logger.info("Running full preprocessing pipeline...")
            
            # Run the complete pipeline
            results = run_full_preprocessing_pipeline(
                df_raw=df_uk_weekly,
                save_outputs=True,
                create_backups=True
            )
            
            logger.info("Full preprocessing completed successfully!")
            logger.info("Available results:")
            for key in results.keys():
                if isinstance(results[key], pd.DataFrame):
                    logger.info(f"  {key}: {results[key].shape}")
                else:
                    logger.info(f"  {key}: {type(results[key])}")
            
            # If ISBN data is available, also run the loaded data preprocessing
            if df_isbns is not None:
                logger.info("\nRunning loaded data preprocessing (with ISBN merge)...")
                loaded_results = preprocess_loaded_data(df_isbns, df_uk_weekly)
                logger.info("Loaded data preprocessing completed!")
        else:
            logger.error("No UK weekly data found. Please ensure UK_weekly_data.csv exists in data/raw/")
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")