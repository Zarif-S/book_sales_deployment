import pandas as pd
import gdown
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_file_recent(file_path: str, max_age_hours: int = 240) -> bool:
    """Check if a file exists and is recent enough to avoid re-downloading."""
    if not os.path.exists(file_path):
        return False
    
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
    return file_age < timedelta(hours=max_age_hours)

def download_google_sheet(file_id: str, destination_file: str, force_download: bool = False) -> str:
    """Download Google Sheets with error handling and caching."""
    if not force_download and is_file_recent(destination_file):
        logger.info(f"Using cached file: {destination_file}")
        return destination_file
    
    download_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx'
    
    try:
        logger.info(f"Downloading from Google Sheets: {file_id}")
        gdown.download(download_url, destination_file, quiet=False)
        logger.info(f"Successfully downloaded: {destination_file}")
        return destination_file
    except Exception as e:
        logger.error(f"Failed to download Google Sheet {file_id}: {str(e)}")
        # If download fails but cached file exists, use it
        if os.path.exists(destination_file):
            logger.warning(f"Download failed, using existing file: {destination_file}")
            return destination_file
        else:
            raise Exception(f"Download failed and no cached file available: {str(e)}")

def load_excel_sheet(file_name: str, sheet_name: str) -> pd.DataFrame:
    """Load a single Excel sheet into a DataFrame."""
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        logger.info(f"Loaded sheet '{sheet_name}' with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load sheet '{sheet_name}' from {file_name}: {str(e)}")
        raise

def concatenate_sheets(sheets_data: List[Tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    """Concatenate multiple sheets with source labels into a single DataFrame."""
    all_sheets = []
    for df, source_label in sheets_data:
        df_copy = df.copy()
        df_copy['Source'] = source_label
        all_sheets.append(df_copy)
    
    concatenated_df = pd.concat(all_sheets, ignore_index=True)
    logger.info(f"Concatenated {len(all_sheets)} sheets into DataFrame with {len(concatenated_df)} total rows")
    return concatenated_df

def load_and_concat_sheets(file_name: str, sheet_names: List[str], source_labels: List[str]) -> pd.DataFrame:
    """Load multiple sheets and concatenate them into a single DataFrame."""
    sheets_data = []
    for sheet_name, source_label in zip(sheet_names, source_labels):
        df = load_excel_sheet(file_name, sheet_name)
        sheets_data.append((df, source_label))
    
    return concatenate_sheets(sheets_data)

def get_data_config() -> Dict[str, Union[str, List[str]]]:
    """Return configuration for data sources."""
    return {
        'isbn_file_id': '1OlWWukTpNCXKT21n92yGF1rWHZPwy3a-',
        'uk_file_id': '1uMYiErVo4YI8QVBGzEXixtoxTfgnpgHi',
        'isbn_sheet_names': ["F - Adult Fiction", "S Adult Non-Fiction Specialist", "T Adult Non-Fiction Trade", "Y Children's, YA & Educational"],
        'uk_sheet_names': ["F Adult Fiction", "S Adult Non-Fiction Specialist", "T Adult Non-Fiction Trade", "Y Children's, YA & Educational"],
        'source_labels': ["F - Adult Fiction", "S Adult Non-Fiction Specialist", "T Adult Non-Fiction Trade", "Y Children's, YA & Educational"]
    }

def ensure_raw_data_dir() -> str:
    """Ensure the raw data directory exists and return its path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    return raw_data_dir

def load_data_from_google_sheets(file_id: str, sheet_names: List[str], source_labels: List[str], 
                                destination_filename: str, force_download: bool = False) -> pd.DataFrame:
    """Generic function to load data from Google Sheets."""
    raw_data_dir = ensure_raw_data_dir()
    destination_file = os.path.join(raw_data_dir, destination_filename)
    
    download_google_sheet(file_id, destination_file, force_download)
    return load_and_concat_sheets(destination_file, sheet_names, source_labels)

def save_dataframe_as_csv(df: pd.DataFrame, filename: str, description: str) -> str:
    """Save a DataFrame as CSV and return the file path."""
    raw_data_dir = ensure_raw_data_dir()
    csv_path = os.path.join(raw_data_dir, filename)
    df.to_csv(csv_path, index=False)
    logger.info(f"{description} saved as CSV: {csv_path}")
    return csv_path

def save_raw_data_as_csv(datasets: Dict[str, Tuple[pd.DataFrame, str, str]]) -> Dict[str, str]:
    """Save multiple DataFrames as CSV files."""
    csv_paths = {}
    for key, (df, filename, description) in datasets.items():
        csv_paths[key] = save_dataframe_as_csv(df, filename, description)
    return csv_paths

def load_csv_data_file(csv_path: str, description: str) -> Optional[pd.DataFrame]:
    """Load a single CSV file if it exists."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {description} from CSV: {csv_path}")
        return df
    else:
        logger.warning(f"{description} CSV file not found: {csv_path}")
        return None

def load_csv_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load data from existing CSV files in the raw data directory."""
    raw_data_dir = ensure_raw_data_dir()
    
    # Define CSV file paths
    isbn_csv_path = os.path.join(raw_data_dir, 'ISBN_data.csv')
    uk_csv_path = os.path.join(raw_data_dir, 'UK_weekly_data.csv')
    
    # Load both datasets
    df_isbns = load_csv_data_file(isbn_csv_path, "ISBN data")
    df_uk_weekly = load_csv_data_file(uk_csv_path, "UK weekly data")
    
    return df_isbns, df_uk_weekly

def get_csv_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Main function to load and return data from CSV files."""
    return load_csv_data()

def load_isbn_data(config: Optional[Dict[str, Union[str, List[str]]]] = None, 
                   force_download: bool = False) -> pd.DataFrame:
    """Load ISBN data from Google Sheets."""
    if config is None:
        config = get_data_config()
    
    return load_data_from_google_sheets(
        file_id=config['isbn_file_id'],
        sheet_names=config['isbn_sheet_names'],
        source_labels=config['source_labels'],
        destination_filename='ISBN.xlsx',
        force_download=force_download
    )

def load_uk_weekly_data(config: Optional[Dict[str, Union[str, List[str]]]] = None, 
                        force_download: bool = False) -> pd.DataFrame:
    """Load UK weekly data from Google Sheets."""
    if config is None:
        config = get_data_config()
    
    return load_data_from_google_sheets(
        file_id=config['uk_file_id'],
        sheet_names=config['uk_sheet_names'],
        source_labels=config['source_labels'],
        destination_filename='UK_weekly_data.xlsx',
        force_download=force_download
    )

def get_isbn_data(force_download: bool = False) -> pd.DataFrame:
    """Main function to load and return ISBN data."""
    return load_isbn_data(force_download=force_download)

def get_uk_weekly_data(force_download: bool = False) -> pd.DataFrame:
    """Main function to load and return UK weekly data."""
    return load_uk_weekly_data(force_download=force_download)

def get_merged_data(force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """Main function to load, merge, and return processed data."""
    config = get_data_config()
    
    # Load both datasets
    df_isbns = load_isbn_data(config, force_download)
    df_uk_weekly = load_uk_weekly_data(config, force_download)
    
    # Save as CSV files
    datasets = {
        'isbn': (df_isbns, 'ISBN_data.csv', 'ISBN data'),
        'uk_weekly': (df_uk_weekly, 'UK_weekly_data.csv', 'UK weekly data')
    }
    save_raw_data_as_csv(datasets)
    
    return {
        'df_isbns': df_isbns,
        'df_uk_weekly': df_uk_weekly
    }

if __name__ == '__main__':
    # Load and process all data
    data_dict = get_merged_data()
    
    # Access individual DataFrames
    df_isbns = data_dict['df_isbns']
    df_uk_weekly = data_dict['df_uk_weekly']
    
    print("\nData loading completed successfully!")
    print(f"ISBN data shape: {df_isbns.shape}")
    print(f"UK weekly data shape: {df_uk_weekly.shape}")