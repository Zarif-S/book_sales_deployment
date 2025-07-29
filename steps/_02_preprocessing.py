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
from typing import Dict, List, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data_from_raw():
    """
    Load CSV data directly from the data/raw directory.
    
    Returns:
        Tuple of (df_isbns, df_uk_weekly) or (None, None) if files not found
    """
    logger.info("Loading CSV data from data/raw directory...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    
    # Load ISBN data
    isbn_csv_path = os.path.join(raw_data_dir, 'ISBN_data.csv')
    if os.path.exists(isbn_csv_path):
        df_isbns = pd.read_csv(isbn_csv_path)
        logger.info(f"Loaded ISBN data: {df_isbns.shape}")
    else:
        logger.warning(f"ISBN CSV file not found: {isbn_csv_path}")
        df_isbns = None
    
    # Load UK weekly data
    uk_csv_path = os.path.join(raw_data_dir, 'UK_weekly_data.csv')
    if os.path.exists(uk_csv_path):
        df_uk_weekly = pd.read_csv(uk_csv_path)
        logger.info(f"Loaded UK weekly data: {df_uk_weekly.shape}")
    else:
        logger.warning(f"UK weekly CSV file not found: {uk_csv_path}")
        df_uk_weekly = None
    
    return df_isbns, df_uk_weekly


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types for proper processing.
    
    Args:
        df: Raw sales data DataFrame
        
    Returns:
        DataFrame with converted data types
    """
    logger.info("Converting data types...")
    
    # Convert ISBN to string
    df['ISBN'] = df['ISBN'].astype(str)
    
    # Convert date to datetime if not already
    if 'End Date' in df.columns:
        df['End Date'] = pd.to_datetime(df['End Date'])
    
    return df


def prepare_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for time series analysis by setting date as index and sorting.
    
    Args:
        df: DataFrame with 'End Date' column or already having 'End Date' as index
        
    Returns:
        DataFrame with 'End Date' as index, sorted chronologically
    """
    logger.info("Preparing time series data...")
    
    # Check if 'End Date' is already the index
    if df.index.name == 'End Date':
        logger.info("'End Date' is already the index, skipping index setting...")
    else:
        # Set 'End Date' as index and sort by it
        df.set_index('End Date', inplace=True)
    
    # Sort by index
    df.sort_index(inplace=True)
    
    return df


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing weeks with 0 sales to ensure continuous time series.
    
    Args:
        df: DataFrame with 'End Date' as index
        
    Returns:
        DataFrame with missing weeks filled with 0
    """
    logger.info("Filling missing weeks with 0 sales...")
    
    completed_dfs = []
    for isbn, group in df.groupby('ISBN'):
        # Fill missing weeks with 0
        group = group.asfreq('W-SAT', fill_value=0)
        group['ISBN'] = isbn  # Add ISBN back to group
        completed_dfs.append(group)
    
    df_filled = pd.concat(completed_dfs)
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
    
    def resample_group(group):
        """Resample a single group by ISBN."""
        resampled = group.resample('W-SAT').agg({
            'Value': 'mean',        # Weekly average of Value
            'ASP': 'mean',          # Weekly average of ASP
            'RRP': 'mean',          # Weekly average of RRP
            'Volume': 'mean',       # Weekly average of Volume
            'Title': 'first',       # Keep first occurrence of Title
            'Author': 'first',      # Keep first occurrence of Author
            'Binding': 'first',     # Keep first occurrence of Binding
            'Imprint': 'first',     # Keep first occurrence of Imprint
            'Publisher Group': 'first',  # Keep first occurrence of Publisher Group
            'Product Class': 'first',    # Keep first occurrence of Product Class
            'Source': 'first',           # Keep first occurrence of Source
        })
        return resampled
    
    # Apply the function to each group
    weekly_resampled = df.groupby('ISBN').apply(resample_group).reset_index()
    return weekly_resampled


def filter_data_by_date(df: pd.DataFrame, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Filter data by date range.
    
    Args:
        df: DataFrame with datetime index
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (optional)
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering data from {start_date} to {end_date or 'end'}")
    
    if end_date:
        mask = (df.index >= start_date) & (df.index <= end_date)
    else:
        mask = df.index >= start_date
    
    return df[mask]


def get_isbns_beyond_date(df: pd.DataFrame, cutoff_date: str) -> List[str]:
    """
    Get ISBNs that have sales data beyond a specific date.
    
    Args:
        df: DataFrame with sales data
        cutoff_date: Date cutoff in 'YYYY-MM-DD' format
        
    Returns:
        List of ISBNs with data beyond the cutoff date
    """
    logger.info(f"Finding ISBNs with data beyond {cutoff_date}")
    
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
    logger.info(f"Getting summary for books: {book_titles}")
    
    # Filter for books with the specified titles, case-insensitive
    pattern = '|'.join(book_titles)
    filtered_books = df[df['Title'].str.contains(pattern, case=False, na=False)]
    
    # Get unique books without duplicates
    unique_books = filtered_books[['ISBN', 'Title', 'Binding', 'RRP']].drop_duplicates()
    
    # Group by ISBN, Title, Binding, and RRP, then summarize data
    book_summary = filtered_books.groupby(['ISBN', 'Title', 'Binding', 'RRP']).agg(
        Count=('ISBN', 'size'),       # Count the number of rows (occurrences)
        Volume_Sum=('Volume', 'sum'),  # Sum of Volume
        Value_Sum=('Value', 'sum'),    # Sum of Value
        ASP_Avg=('ASP', 'mean')        # Average of ASP
    ).reset_index()
    
    return book_summary


def select_specific_books(df: pd.DataFrame, isbn_list: List[str], start_date: str) -> pd.DataFrame:
    """
    Select specific books by ISBN for detailed analysis.
    
    Args:
        df: DataFrame with sales data
        isbn_list: List of ISBNs to select
        start_date: Start date for filtering in 'YYYY-MM-DD' format
        
    Returns:
        Filtered DataFrame with selected books
    """
    logger.info(f"Selecting books with ISBNs: {isbn_list} from {start_date}")
    
    filtered_data = df[
        (df['ISBN'].isin(isbn_list)) & 
        (df.index >= start_date)
    ]
    
    return filtered_data


def get_isbn_to_title_mapping() -> Dict[str, str]:
    """
    Get mapping of ISBNs to book titles for the selected books.
    
    Returns:
        Dictionary mapping ISBNs to titles
    """
    return {
        '9780722532935': 'Alchemist, The',
        '9780241003008': 'Very Hungry Caterpillar, The',
        '9780140500875': 'Very Hungry Caterpillar, The'
    }


def aggregate_yearly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly data to yearly data.
    
    Args:
        df: DataFrame with weekly sales data
        
    Returns:
        DataFrame with yearly aggregated data
    """
    logger.info("Aggregating data to yearly frequency...")
    
    yearly_data = df.groupby([df.index.year, 'ISBN'])['Volume'].sum().reset_index()
    yearly_data.rename(columns={'End Date': 'Year'}, inplace=True)
    
    return yearly_data


def save_processed_data(df: pd.DataFrame, filename: str, processed_dir: str = None) -> str:
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
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(project_root, 'data', 'processed')
    
    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(processed_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=True if df.index.name else False)
    logger.info(f"Saved processed data to: {csv_path}")
    
    return csv_path


def load_processed_data(filename: str, processed_dir: str = None) -> pd.DataFrame:
    """
    Load processed DataFrame from the processed data directory.
    
    Args:
        filename: Name of the file to load (without extension)
        processed_dir: Directory to load from (optional, defaults to data/processed)
        
    Returns:
        Loaded DataFrame or None if file not found
    """
    if processed_dir is None:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(project_root, 'data', 'processed')
    
    # Construct file path
    csv_path = os.path.join(processed_dir, f"{filename}.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded processed data from: {csv_path}")
        return df
    else:
        logger.warning(f"Processed data file not found: {csv_path}")
        return None


def preprocess_sales_data(df_raw: pd.DataFrame, save_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function that orchestrates the entire data preprocessing pipeline.
    
    Args:
        df_raw: Raw sales data DataFrame
        save_outputs: Whether to save processed outputs to data/processed directory
        
    Returns:
        Tuple of (processed_data, filtered_data, selected_books_data)
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Step 1: Convert data types
    df = convert_data_types(df_raw.copy())
    
    # Step 2: Prepare time series data
    df = prepare_time_series_data(df)
    
    # Step 3: Fill missing weeks
    df_filled = fill_missing_weeks(df)
    
    # Step 4: Get ISBNs with data beyond 2024-07-01
    isbns_beyond_2024 = get_isbns_beyond_date(df_filled, '2024-07-01')
    
    # Step 5: Filter data for selected books (The Alchemist and The Very Hungry Caterpillar)
    selected_isbns = ['9780722532935', '9780241003008', '9780140500875']
    selected_books_data = select_specific_books(df_filled, selected_isbns, '2012-01-01')
    
    # Step 6: Save processed outputs if requested
    if save_outputs:
        logger.info("Saving processed outputs to data/processed directory...")
        
        # Save the filled data (main processed dataset)
        save_processed_data(df_filled, "processed_sales_data_filled")
        
        # Save the selected books data
        save_processed_data(selected_books_data, "selected_books_data")
        
        # Save a summary of the processing
        processing_summary = {
            'total_records_processed': len(df_filled),
            'unique_isbns': df_filled['ISBN'].nunique(),
            'date_range': (df_filled.index.min(), df_filled.index.max()),
            'selected_books_records': len(selected_books_data),
            'isbns_beyond_2024': len(isbns_beyond_2024)
        }
        
        # Save processing summary as a simple text file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(project_root, 'data', 'processed')
        summary_path = os.path.join(processed_dir, "processing_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Data Processing Summary\n")
            f.write("=" * 30 + "\n\n")
            for key, value in processing_summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved processing summary to: {summary_path}")
    
    logger.info("Data preprocessing completed successfully")
    
    return df_filled, df_filled, selected_books_data


def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'shape': df.shape,
        'date_range': (df.index.min(), df.index.max()),
        'unique_isbns': df['ISBN'].nunique(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return info


def analyze_missing_values(df, dataset_name):
    """Analyze and print missing values in a DataFrame."""
    print(f"{dataset_name} Data Info:")
    df.info()
    print(f"\nMissing values in {dataset_name}:")
    missing_values = df.isna().sum()
    print(missing_values)
    if missing_values.sum() > 0:
        print(f"\nNulls present in: {', '.join(missing_values[missing_values > 0].index.tolist())}")
    else:
        print(f"\nNo missing values found in {dataset_name}")
    return missing_values

def merge_and_fill_author_data(df_uk_weekly, df_isbns):
    """Merge UK weekly data with ISBN data and fill missing author values."""
    df_merged = df_uk_weekly.merge(df_isbns[['ISBN', 'Author']], on='ISBN', how='left', suffixes=('', '_ISBN'))
    print("\nSample of merged DataFrame with missing Author values:")
    print(df_merged[df_merged['Author'].isna()].head())
    df_merged['Author'] = df_merged['Author'].fillna(df_merged['Author_ISBN'])
    print("\nSample after filling missing 'Author' values:")
    filled_samples = df_merged[df_merged['Author_ISBN'].notna()]
    if not filled_samples.empty:
        print(filled_samples.head())
    else:
        print("No samples with filled Author values found")
    df_uk_weekly_filled = df_merged.drop(columns=['Author_ISBN'])
    print("\nSample after dropping extra columns:")
    print(df_uk_weekly_filled.head())
    return df_uk_weekly_filled

def display_data_summary(df_uk_weekly):
    """Display summary information about the data."""
    unique_titles = df_uk_weekly['Title'].unique()
    print(f"\nUnique titles in df_UK_weekly: {len(unique_titles)}")
    print("First 10 unique titles:")
    for i, title in enumerate(unique_titles[:10]):
        print(f"{i+1}. {title}")
    if len(unique_titles) > 10:
        print(f"... and {len(unique_titles) - 10} more titles")

def create_data_backups(df_isbns, df_uk_weekly):
    """Create backups of raw data."""
    df_isbns_raw = df_isbns.copy()
    df_uk_weekly_raw = df_uk_weekly.copy()
    return df_isbns_raw, df_uk_weekly_raw

# Optionally, add a new function to perform the full preprocessing pipeline that was previously in get_csv_data/get_merged_data

def preprocess_loaded_data(df_isbns, df_uk_weekly):
    """Run the preprocessing pipeline on loaded data."""
    analyze_missing_values(df_isbns, "ISBN")
    analyze_missing_values(df_uk_weekly, "UK Weekly")
    df_uk_weekly_filled = merge_and_fill_author_data(df_uk_weekly, df_isbns)
    print("\nRemaining missing values after filling 'Author':")
    print(df_uk_weekly_filled.isna().sum())
    display_data_summary(df_uk_weekly)
    df_isbns_raw, df_uk_weekly_raw = create_data_backups(df_isbns, df_uk_weekly)
    return {
        'df_isbns': df_isbns,
        'df_uk_weekly': df_uk_weekly_filled,
        'df_isbns_raw': df_isbns_raw,
        'df_uk_weekly_raw': df_uk_weekly_raw
    }

if __name__ == "__main__":
    # This section would be used for testing the preprocessing functions
    # when the script is run directly
    logger.info("Preprocessing module loaded successfully")
    
    # Load raw data and run preprocessing pipeline
    try:
        df_isbns, df_uk_weekly = load_csv_data_from_raw()
        
        if df_uk_weekly is not None:
            logger.info("Running preprocessing pipeline...")
            processed_data, filtered_data, selected_books = preprocess_sales_data(df_uk_weekly, save_outputs=True)
            logger.info("Preprocessing completed successfully!")
            logger.info(f"Processed data shape: {processed_data.shape}")
            logger.info(f"Selected books data shape: {selected_books.shape}")
        else:
            logger.error("No UK weekly data found. Please ensure UK_weekly_data.csv exists in data/raw/")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")