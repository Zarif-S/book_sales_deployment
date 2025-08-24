"""
Data Helpers for Streamlit App

Self-contained data loading and processing functions.
Handles loading historical data and formatting for visualization.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Simple data loader for historical sales data."""
    
    def __init__(self, base_path: str = None):
        """Initialize data loader with base path."""
        if base_path is None:
            # Default to project root data directory
            current_dir = Path(__file__).parent.parent.parent
            self.base_path = current_dir / "data" / "processed"
        else:
            self.base_path = Path(base_path)
        
        logger.info(f"DataLoader initialized with base path: {self.base_path}")
    
    def get_available_books(self) -> Dict[str, Dict[str, str]]:
        """Get available books based on data files."""
        available_books = {}
        
        # Known book metadata
        book_metadata = {
            "9780722532935": {
                "title": "The Alchemist",
                "author": "Paulo Coelho"
            },
            "9780241003008": {
                "title": "The Very Hungry Caterpillar", 
                "author": "Eric Carle"
            }
        }
        
        # Check which books have data files
        for isbn, metadata in book_metadata.items():
            train_file = self.base_path / f"train_data_{isbn}.csv"
            test_file = self.base_path / f"test_data_{isbn}.csv"
            
            if train_file.exists():
                available_books[isbn] = {
                    **metadata,
                    "train_file": str(train_file),
                    "test_file": str(test_file) if test_file.exists() else None,
                    "has_data": True
                }
                logger.info(f"Found data for {metadata['title']} ({isbn})")
            else:
                # Still include in list but mark as no data
                available_books[isbn] = {
                    **metadata,
                    "has_data": False
                }
                logger.warning(f"No data file found for {metadata['title']} ({isbn})")
        
        return available_books
    
    def load_historical_data(self, isbn: str = None) -> pd.DataFrame:
        """
        Load historical sales data for a specific book or all books.
        
        Args:
            isbn: Specific ISBN to load (if None, loads all available)
            
        Returns:
            DataFrame with historical sales data
        """
        logger.info(f"Loading historical data for ISBN: {isbn or 'all books'}")
        
        if isbn:
            return self._load_single_book_data(isbn)
        else:
            return self._load_all_books_data()
    
    def _load_single_book_data(self, isbn: str) -> pd.DataFrame:
        """Load data for a single book."""
        train_file = self.base_path / f"train_data_{isbn}.csv"
        test_file = self.base_path / f"test_data_{isbn}.csv"
        
        combined_data = pd.DataFrame()
        
        # Load training data
        if train_file.exists():
            try:
                train_df = pd.read_csv(train_file)
                combined_data = pd.concat([combined_data, train_df], ignore_index=True)
                logger.info(f"Loaded training data: {len(train_df)} rows")
            except Exception as e:
                logger.error(f"Error loading training data for {isbn}: {e}")
        else:
            logger.warning(f"Training file not found: {train_file}")
        
        # Load test data if available
        if test_file.exists():
            try:
                test_df = pd.read_csv(test_file)
                combined_data = pd.concat([combined_data, test_df], ignore_index=True)
                logger.info(f"Loaded test data: {len(test_df)} rows")
            except Exception as e:
                logger.error(f"Error loading test data for {isbn}: {e}")
        
        if combined_data.empty:
            logger.warning(f"No data found for ISBN: {isbn}")
            return self._create_sample_data(isbn)
        
        # Clean and format data
        return self._clean_data(combined_data)
    
    def _load_all_books_data(self) -> pd.DataFrame:
        """Load data for all available books."""
        all_data = pd.DataFrame()
        available_books = self.get_available_books()
        
        for isbn, book_info in available_books.items():
            if book_info.get('has_data', False):
                book_data = self._load_single_book_data(isbn)
                all_data = pd.concat([all_data, book_data], ignore_index=True)
        
        if all_data.empty:
            logger.warning("No historical data found, creating sample data")
            return self._create_sample_data_all()
        
        return self._clean_data(all_data)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data format."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'End Date': 'End_Date',
            'end_date': 'End_Date',
            'date': 'End_Date'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure we have required columns
        required_columns = ['End_Date', 'ISBN', 'Title', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        
        # Convert date column
        if 'End_Date' in df.columns:
            df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce')
        
        # Sort by date
        if 'End_Date' in df.columns:
            df = df.sort_values('End_Date')
        
        # Remove any rows with missing critical data
        if 'Volume' in df.columns:
            df = df.dropna(subset=['Volume'])
        
        logger.info(f"Cleaned data: {len(df)} rows, columns: {list(df.columns)}")
        return df
    
    def _create_sample_data(self, isbn: str) -> pd.DataFrame:
        """Create sample data for development when no real data is available."""
        logger.info(f"Creating sample data for ISBN: {isbn}")
        
        # Book metadata
        book_info = {
            "9780722532935": {"title": "The Alchemist", "author": "Paulo Coelho"},
            "9780241003008": {"title": "The Very Hungry Caterpillar", "author": "Eric Carle"}
        }
        
        info = book_info.get(isbn, {"title": f"Book {isbn}", "author": "Unknown Author"})
        
        # Generate sample data for ~2 years
        date_range = pd.date_range(
            start='2022-01-01', 
            end='2023-12-31', 
            freq='W'
        )
        
        # Create realistic sales patterns
        np.random.seed(hash(isbn) % 2**32)  # Consistent random data for each ISBN
        
        base_sales = 400 if isbn == "9780722532935" else 320
        seasonal_pattern = np.sin(2 * np.pi * np.arange(len(date_range)) / 52.0)  # Annual cycle
        trend = np.linspace(0, 0.1, len(date_range))  # Slight upward trend
        noise = np.random.normal(0, 0.1, len(date_range))
        
        volume = base_sales * (1 + 0.2 * seasonal_pattern + trend + noise)
        volume = np.maximum(volume, 50)  # Minimum sales floor
        
        sample_data = pd.DataFrame({
            'End_Date': date_range,
            'ISBN': isbn,
            'Title': info['title'],
            'Author': info['author'],
            'Volume': volume.round(1),
            'Interval': [f"{d.year}{d.week:02d}" for d in date_range],
            'Value': volume * 7.5,  # Rough price estimate
            'ASP': 7.5,
            'RRP': 9.99,
            'Binding': 'Paperback',
            'Source': 'Sample Data'
        })
        
        logger.info(f"Generated {len(sample_data)} sample records for {info['title']}")
        return sample_data
    
    def _create_sample_data_all(self) -> pd.DataFrame:
        """Create sample data for all books."""
        all_sample = pd.DataFrame()
        
        for isbn in ["9780722532935", "9780241003008"]:
            book_sample = self._create_sample_data(isbn)
            all_sample = pd.concat([all_sample, book_sample], ignore_index=True)
        
        return all_sample
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics about available data."""
        summary = {
            "available_books": {},
            "total_books": 0,
            "date_ranges": {},
            "total_records": 0
        }
        
        available_books = self.get_available_books()
        summary["total_books"] = len(available_books)
        
        for isbn, book_info in available_books.items():
            if book_info.get('has_data', False):
                try:
                    data = self._load_single_book_data(isbn)
                    if not data.empty and 'End_Date' in data.columns:
                        summary["available_books"][isbn] = {
                            "title": book_info['title'],
                            "records": len(data),
                            "date_start": data['End_Date'].min().strftime('%Y-%m-%d') if pd.notnull(data['End_Date'].min()) else 'Unknown',
                            "date_end": data['End_Date'].max().strftime('%Y-%m-%d') if pd.notnull(data['End_Date'].max()) else 'Unknown'
                        }
                        summary["total_records"] += len(data)
                except Exception as e:
                    logger.error(f"Error getting summary for {isbn}: {e}")
        
        return summary
    
    def get_latest_date(self, isbn: str = None) -> Optional[pd.Timestamp]:
        """Get the latest date in the historical data."""
        try:
            data = self.load_historical_data(isbn)
            if not data.empty and 'End_Date' in data.columns:
                return data['End_Date'].max()
        except Exception as e:
            logger.error(f"Error getting latest date: {e}")
        
        return None


# Utility functions
def format_data_for_display(df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    """Format data for nice display in Streamlit."""
    if df.empty:
        return df
    
    display_df = df.copy()
    
    # Select key columns for display
    display_columns = ['End_Date', 'Title', 'Volume', 'Value', 'ASP']
    available_columns = [col for col in display_columns if col in display_df.columns]
    
    if available_columns:
        display_df = display_df[available_columns]
    
    # Format dates
    if 'End_Date' in display_df.columns:
        display_df['Date'] = display_df['End_Date'].dt.strftime('%Y-%m-%d')
        display_df = display_df.drop('End_Date', axis=1)
    
    # Round numeric columns
    numeric_columns = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_columns] = display_df[numeric_columns].round(2)
    
    # Limit rows
    if len(display_df) > max_rows:
        display_df = display_df.tail(max_rows)
    
    return display_df


def get_data_insights(df: pd.DataFrame) -> Dict[str, any]:
    """Generate simple insights about the data."""
    if df.empty:
        return {"error": "No data available"}
    
    insights = {}
    
    if 'Volume' in df.columns:
        insights['avg_weekly_sales'] = df['Volume'].mean()
        insights['min_weekly_sales'] = df['Volume'].min()
        insights['max_weekly_sales'] = df['Volume'].max()
        insights['total_periods'] = len(df)
    
    if 'End_Date' in df.columns:
        date_range = df['End_Date'].max() - df['End_Date'].min()
        insights['data_span_days'] = date_range.days
        insights['data_span_weeks'] = date_range.days // 7
    
    if 'Title' in df.columns:
        insights['books_in_data'] = df['Title'].nunique()
        insights['book_titles'] = df['Title'].unique().tolist()
    
    return insights


# Example usage and testing
if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    
    # Get available books
    available = loader.get_available_books()
    print("Available books:")
    for isbn, info in available.items():
        print(f"  {isbn}: {info['title']} - Data: {info.get('has_data', False)}")
    
    # Load data for The Alchemist
    data = loader.load_historical_data("9780722532935")
    print(f"\nLoaded data shape: {data.shape}")
    if not data.empty:
        print("Columns:", list(data.columns))
        print("Date range:", data['End_Date'].min(), "to", data['End_Date'].max())
    
    # Get summary
    summary = loader.get_data_summary()
    print(f"\nData Summary:")
    print(f"Total books: {summary['total_books']}")
    print(f"Total records: {summary['total_records']}")