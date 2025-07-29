import pandas as pd
import gdown
import os

def download_google_sheet(file_id, destination_file):
    """Download Google Sheets and load into a DataFrame."""
    download_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
    gdown.download(download_url, destination_file, quiet=False)
    return destination_file

def load_and_concat_sheets(file_name, sheet_names, source_labels):
    """Load and concatenate multiple sheets into a single DataFrame."""
    all_sheets = []
    for sheet_name, source_label in zip(sheet_names, source_labels):
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        df['Source'] = source_label
        all_sheets.append(df)
    return pd.concat(all_sheets, ignore_index=True)

def get_data_config():
    """Return configuration for data sources."""
    return {
        'isbn_file_id': '1OlWWukTpNCXKT21n92yGF1rWHZPwy3a-',
        'uk_file_id': '1uMYiErVo4YI8QVBGzEXixtoxTfgnpgHi',
        'isbn_sheet_names': ["F - Adult Fiction", "S Adult Non-Fiction Specialist", "T Adult Non-Fiction Trade", "Y Children's, YA & Educational"],
        'uk_sheet_names': ["F Adult Fiction", "S Adult Non-Fiction Specialist", "T Adult Non-Fiction Trade", "Y Children's, YA & Educational"],
        'source_labels': ["F - Adult Fiction", "S Adult Non-Fiction Specialist", "T Adult Non-Fiction Trade", "Y Children's, YA & Educational"]
    }

def ensure_raw_data_dir():
    """Ensure the raw data directory exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    return raw_data_dir

def load_isbn_data(config):
    """Load ISBN data from Google Sheets."""
    raw_data_dir = ensure_raw_data_dir()
    isbn_destination_file = os.path.join(raw_data_dir, 'ISBN.xlsx')
    download_google_sheet(config['isbn_file_id'], isbn_destination_file)
    df_isbns = load_and_concat_sheets(isbn_destination_file, config['isbn_sheet_names'], config['source_labels'])
    return df_isbns

def load_uk_weekly_data(config):
    """Load UK weekly data from Google Sheets."""
    raw_data_dir = ensure_raw_data_dir()
    uk_destination_file = os.path.join(raw_data_dir, 'UK_weekly_data.xlsx')
    download_google_sheet(config['uk_file_id'], uk_destination_file)
    df_uk_weekly = load_and_concat_sheets(uk_destination_file, config['uk_sheet_names'], config['source_labels'])
    return df_uk_weekly

def save_raw_data_as_csv(df_isbns, df_uk_weekly):
    """Save raw data as CSV files in the raw data directory."""
    raw_data_dir = ensure_raw_data_dir()
    
    # Save ISBN data as CSV
    isbn_csv_path = os.path.join(raw_data_dir, 'ISBN_data.csv')
    df_isbns.to_csv(isbn_csv_path, index=False)
    print(f"ISBN data saved as CSV: {isbn_csv_path}")
    
    # Save UK weekly data as CSV
    uk_csv_path = os.path.join(raw_data_dir, 'UK_weekly_data.csv')
    df_uk_weekly.to_csv(uk_csv_path, index=False)
    print(f"UK weekly data saved as CSV: {uk_csv_path}")
    
    return isbn_csv_path, uk_csv_path

def analyze_missing_values(df, dataset_name):
    """Analyze and print missing values in a DataFrame."""
    print(f"{dataset_name} Data Info:")
    df.info()
    
    print(f"\nMissing values in {dataset_name}:")
    missing_values = df.isna().sum()
    print(missing_values)
    
    # Print summary of missing values
    if missing_values.sum() > 0:
        print(f"\nNulls present in: {', '.join(missing_values[missing_values > 0].index.tolist())}")
    else:
        print(f"\nNo missing values found in {dataset_name}")
    
    return missing_values

def merge_and_fill_author_data(df_uk_weekly, df_isbns):
    """Merge UK weekly data with ISBN data and fill missing author values."""
    # Step 1: Merge df_UK_weekly with df_ISBNs on 'ISBN' column
    df_merged = df_uk_weekly.merge(df_isbns[['ISBN', 'Author']], on='ISBN', how='left', suffixes=('', '_ISBN'))
    
    print("\nSample of merged DataFrame with missing Author values:")
    print(df_merged[df_merged['Author'].isna()].head())
    
    # Step 2: Fill missing 'Author' values in df_UK_weekly from df_ISBNs where available
    df_merged['Author'] = df_merged['Author'].fillna(df_merged['Author_ISBN'])
    
    print("\nSample after filling missing 'Author' values:")
    filled_samples = df_merged[df_merged['Author_ISBN'].notna()]
    if not filled_samples.empty:
        print(filled_samples.head())
    else:
        print("No samples with filled Author values found")
    
    # Step 3: Drop the extra 'Author_ISBN' column after merge
    df_uk_weekly_filled = df_merged.drop(columns=['Author_ISBN'])
    
    print("\nSample after dropping extra columns:")
    print(df_uk_weekly_filled.head())
    
    return df_uk_weekly_filled

def display_data_summary(df_uk_weekly):
    """Display summary information about the data."""
    # Display unique titles
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

def load_csv_data():
    """Load data from existing CSV files in the raw data directory."""
    raw_data_dir = ensure_raw_data_dir()
    
    # Load ISBN data from CSV
    isbn_csv_path = os.path.join(raw_data_dir, 'ISBN_data.csv')
    if os.path.exists(isbn_csv_path):
        df_isbns = pd.read_csv(isbn_csv_path)
        print(f"Loaded ISBN data from CSV: {isbn_csv_path}")
    else:
        print(f"ISBN CSV file not found: {isbn_csv_path}")
        df_isbns = None
    
    # Load UK weekly data from CSV
    uk_csv_path = os.path.join(raw_data_dir, 'UK_weekly_data.csv')
    if os.path.exists(uk_csv_path):
        df_uk_weekly = pd.read_csv(uk_csv_path)
        print(f"Loaded UK weekly data from CSV: {uk_csv_path}")
    else:
        print(f"UK weekly CSV file not found: {uk_csv_path}")
        df_uk_weekly = None
    
    return df_isbns, df_uk_weekly

def get_csv_data():
    """Main function to load and return data from CSV files."""
    df_isbns, df_uk_weekly = load_csv_data()
    
    if df_isbns is not None and df_uk_weekly is not None:
        # Analyze missing values
        analyze_missing_values(df_isbns, "ISBN")
        analyze_missing_values(df_uk_weekly, "UK Weekly")
        
        # Merge and fill missing data
        df_uk_weekly_filled = merge_and_fill_author_data(df_uk_weekly, df_isbns)
        
        # Check remaining missing values
        print("\nRemaining missing values after filling 'Author':")
        print(df_uk_weekly_filled.isna().sum())
        
        # Display summary
        display_data_summary(df_uk_weekly)
        
        # Create backups
        df_isbns_raw, df_uk_weekly_raw = create_data_backups(df_isbns, df_uk_weekly)
        
        return {
            'df_isbns': df_isbns,
            'df_uk_weekly': df_uk_weekly_filled,
            'df_isbns_raw': df_isbns_raw,
            'df_uk_weekly_raw': df_uk_weekly_raw
        }
    else:
        print("Could not load one or both CSV files. Please ensure they exist in data/raw/")
        return None

def get_isbn_data():
    """Main function to load and return ISBN data."""
    config = get_data_config()
    df_isbns = load_isbn_data(config)
    return df_isbns

def get_uk_weekly_data():
    """Main function to load and return UK weekly data."""
    config = get_data_config()
    df_uk_weekly = load_uk_weekly_data(config)
    return df_uk_weekly

def get_merged_data():
    """Main function to load, merge, and return processed data."""
    config = get_data_config()
    
    # Load raw data
    df_isbns = load_isbn_data(config)
    df_uk_weekly = load_uk_weekly_data(config)
    
    # Save raw data as CSV files
    save_raw_data_as_csv(df_isbns, df_uk_weekly)
    
    # Analyze missing values
    analyze_missing_values(df_isbns, "ISBN")
    analyze_missing_values(df_uk_weekly, "UK Weekly")
    
    # Merge and fill missing data
    df_uk_weekly_filled = merge_and_fill_author_data(df_uk_weekly, df_isbns)
    
    # Check remaining missing values
    print("\nRemaining missing values after filling 'Author':")
    print(df_uk_weekly_filled.isna().sum())
    
    # Display summary
    display_data_summary(df_uk_weekly)
    
    # Create backups
    df_isbns_raw, df_uk_weekly_raw = create_data_backups(df_isbns, df_uk_weekly)
    
    return {
        'df_isbns': df_isbns,
        'df_uk_weekly': df_uk_weekly_filled,
        'df_isbns_raw': df_isbns_raw,
        'df_uk_weekly_raw': df_uk_weekly_raw
    }

if __name__ == '__main__':
    # Load and process all data
    data_dict = get_merged_data()
    
    # Access individual DataFrames
    df_isbns = data_dict['df_isbns']
    df_uk_weekly = data_dict['df_uk_weekly']
    df_isbns_raw = data_dict['df_isbns_raw']
    df_uk_weekly_raw = data_dict['df_uk_weekly_raw']
    
    print("\nData loading completed successfully!")
    print(f"ISBN data shape: {df_isbns.shape}")
    print(f"UK weekly data shape: {df_uk_weekly.shape}")