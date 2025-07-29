#!/usr/bin/env python3
"""
Test script to demonstrate loading CSV data from data/raw directory and saving processed data.
"""

import sys
import os

# Add the steps directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'steps'))

from _01_load_data import get_csv_data
from _02_preprocessing import (
    load_csv_data_from_raw, 
    preprocess_sales_data, 
    save_processed_data, 
    load_processed_data
)

def test_csv_loading():
    """Test loading CSV data using different methods."""
    print("=== Testing CSV Data Loading ===\n")
    
    # Method 1: Using the data loading module
    print("1. Testing get_csv_data() from _01_load_data:")
    try:
        data_dict = get_csv_data()
        if data_dict:
            print("✓ Successfully loaded data using get_csv_data()")
            print(f"  - ISBN data shape: {data_dict['df_isbns'].shape}")
            print(f"  - UK weekly data shape: {data_dict['df_uk_weekly'].shape}")
        else:
            print("✗ Failed to load data using get_csv_data()")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Using the preprocessing module
    print("2. Testing load_csv_data_from_raw() from _02_preprocessing:")
    try:
        df_isbns, df_uk_weekly = load_csv_data_from_raw()
        if df_isbns is not None and df_uk_weekly is not None:
            print("✓ Successfully loaded data using load_csv_data_from_raw()")
            print(f"  - ISBN data shape: {df_isbns.shape}")
            print(f"  - UK weekly data shape: {df_uk_weekly.shape}")
            
            # Test preprocessing with saving
            print("\n3. Testing preprocessing pipeline with saving:")
            try:
                processed_data, filtered_data, selected_books = preprocess_sales_data(df_uk_weekly, save_outputs=True)
                print("✓ Successfully preprocessed data and saved to data/processed/")
                print(f"  - Processed data shape: {processed_data.shape}")
                print(f"  - Selected books data shape: {selected_books.shape}")
                
                # Test loading processed data
                print("\n4. Testing loading processed data:")
                try:
                    loaded_processed = load_processed_data("processed_sales_data_filled")
                    loaded_selected = load_processed_data("selected_books_data")
                    
                    if loaded_processed is not None and loaded_selected is not None:
                        print("✓ Successfully loaded processed data")
                        print(f"  - Loaded processed data shape: {loaded_processed.shape}")
                        print(f"  - Loaded selected books shape: {loaded_selected.shape}")
                    else:
                        print("✗ Failed to load some processed data files")
                        
                except Exception as e:
                    print(f"✗ Error loading processed data: {e}")
                    
            except Exception as e:
                print(f"✗ Error in preprocessing: {e}")
        else:
            print("✗ Failed to load data using load_csv_data_from_raw()")
    except Exception as e:
        print(f"✗ Error loading data: {e}")

if __name__ == "__main__":
    test_csv_loading() 