import pandas as pd
import os
from typing import Tuple

def prepare_data_after_2012(book_data: pd.DataFrame, column_name: str, split_size: int = 32, output_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and testing data after 2012-01-01 based on a given split size.

    Args:
        book_data (pd.DataFrame): The DataFrame containing the book data with a time series index.
        column_name (str): The column to split into train and test data.
        split_size (int): The number of entries (weeks or months) to include in the test set.
        output_dir (str): Directory to save CSV files (optional).

    Returns:
        pd.DataFrame: Training data (all data except the last split_size weeks/months).
        pd.DataFrame: Test data (last split_size weeks/months).
    """
    print(f"Preparing data for column: {column_name}")
    print(f"Input book_data shape: {book_data.shape}")
    print(f"Input book_data columns: {list(book_data.columns)}")
    print(f"Input book_data index: {book_data.index.name}")

    # Check if the column exists
    if column_name not in book_data.columns:
        raise ValueError(f"Column '{column_name}' not found in book_data. Available columns: {list(book_data.columns)}")

    # Check the column data
    print(f"Column '{column_name}' dtype: {book_data[column_name].dtype}")
    print(f"Column '{column_name}' non-null count: {book_data[column_name].count()}")
    print(f"Column '{column_name}' sample values: {book_data[column_name].head().tolist()}")

    # Filter data for dates after 2012-01-01 inclusive
    data_after_2012 = book_data[book_data.index >= '2012-01-01']
    print(f"Data after 2012-01-01 shape: {data_after_2012.shape}")

    # CRITICAL FIX: Sort data chronologically (oldest to newest) to ensure proper train/test split
    data_after_2012 = data_after_2012.sort_index(ascending=True)
    print(f"Date range after sorting: {data_after_2012.index.min()} to {data_after_2012.index.max()}")

    # Ensure there is enough data for splitting
    if len(data_after_2012) < split_size:
        raise ValueError(f"Not enough data available for the test set (at least {split_size} entries required).")

    # Split into train and test data - return full DataFrames instead of just the column
    # Now that data is sorted chronologically, iloc[-split_size:] will get the MOST RECENT data
    train_data_df = data_after_2012.iloc[:-split_size].copy()  # All data except the last split_size entries
    test_data_df = data_after_2012.iloc[-split_size:].copy()   # Last split_size entries of data (most recent)

    # Display the results
    print(f"Training data shape: {train_data_df.shape}")
    print(f"Test data shape: {test_data_df.shape}")
    print(f"Training data range: {train_data_df.index.min()} to {train_data_df.index.max()}")
    print(f"Test data range: {test_data_df.index.min()} to {test_data_df.index.max()}")

    # Save to CSV if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get book identifier from data if available
        book_id = "unknown"
        if 'ISBN' in train_data_df.columns and not train_data_df['ISBN'].empty:
            book_id = str(train_data_df['ISBN'].iloc[0])
        elif 'Title' in train_data_df.columns and not train_data_df['Title'].empty:
            book_id = str(train_data_df['Title'].iloc[0]).replace(' ', '_').replace('/', '_')
        
        train_csv_path = os.path.join(output_dir, f'train_data_{book_id}.csv')
        test_csv_path = os.path.join(output_dir, f'test_data_{book_id}.csv')
        
        train_data_df.to_csv(train_csv_path, index=True)
        test_data_df.to_csv(test_csv_path, index=True)
        
        print(f"Saved training data to: {train_csv_path}")
        print(f"Saved test data to: {test_csv_path}")

    return train_data_df, test_data_df

def prepare_multiple_books_data(books_data: dict, column_name: str = 'Volume', split_size: int = 32, output_dir: str = None) -> dict:
    """
    Prepare training and testing data for multiple books after 2012-01-01.

    Args:
        books_data (dict): Dictionary with book names as keys and DataFrames as values.
        column_name (str): The column to split into train and test data.
        split_size (int): The number of entries to include in the test set.
        output_dir (str): Directory to save CSV files (optional).

    Returns:
        dict: Dictionary with book names as keys and tuples of (train_data, test_data) as values.
    """
    prepared_data = {}

    print(f"Preparing data for {len(books_data)} books using column: {column_name}")

    for book_name, book_data in books_data.items():
        try:
            print(f"Processing book: {book_name}")
            print(f"Book data shape: {book_data.shape}")
            print(f"Book data columns: {list(book_data.columns)}")

            # Check if the required column exists
            if column_name not in book_data.columns:
                print(f"Error: Column '{column_name}' not found in data for {book_name}")
                print(f"Available columns: {list(book_data.columns)}")
                prepared_data[book_name] = (None, None)
                continue

            # Check column data
            print(f"Column '{column_name}' dtype: {book_data[column_name].dtype}")
            print(f"Column '{column_name}' non-null count: {book_data[column_name].count()}")
            print(f"Column '{column_name}' sample values: {book_data[column_name].head().tolist()}")

            train_data, test_data = prepare_data_after_2012(book_data, column_name, split_size, output_dir)
            prepared_data[book_name] = (train_data, test_data)
            print(f"Successfully prepared data for {book_name}")
        except Exception as e:
            print(f"Error preparing data for {book_name}: {e}")
            prepared_data[book_name] = (None, None)

    return prepared_data
