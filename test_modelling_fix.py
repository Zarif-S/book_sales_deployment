#!/usr/bin/env python3
"""
Quick test script to verify the modelling data preparation fix.
"""
import pandas as pd
from steps._01_load_data import get_isbn_data, get_uk_weekly_data
from steps._02_preprocessing import preprocess_loaded_data
from steps._03_5_modelling_prep import prepare_multiple_books_data

def test_modelling_data_fix():
    print("Testing modelling data preparation fix...")
    
    # Load and preprocess data
    print("Loading data...")
    df_isbns = get_isbn_data()
    df_uk_weekly = get_uk_weekly_data()
    
    print("Preprocessing data...")
    processed = preprocess_loaded_data(df_isbns, df_uk_weekly)
    df_merged = processed['df_uk_weekly']
    
    # Focus on single book for simpler testing
    selected_isbns = ['9780722532935']  # The Alchemist only
    
    print(f"Filtering for ISBN: {selected_isbns[0]}")
    df_merged['ISBN'] = df_merged['ISBN'].astype(str)
    selected_books_data = df_merged[df_merged['ISBN'].isin(selected_isbns)].copy()
    
    print(f"Selected books data shape: {selected_books_data.shape}")
    
    # Group data by ISBN
    books_data = {}
    for isbn in selected_isbns:
        book_data = selected_books_data[selected_books_data['ISBN'] == isbn].copy()
        if not book_data.empty:
            book_title = book_data['Title'].iloc[0]
            books_data[book_title] = book_data
            print(f"Found {len(book_data)} records for {book_title}")
    
    # Prepare train/test data
    print("Preparing train/test splits...")
    prepared_data = prepare_multiple_books_data(
        books_data=books_data,
        column_name='Volume',
        split_size=32
    )
    
    # Create visualization data (the fixed version)
    print("Creating visualization data...")
    visualization_data = []
    
    for book_name, (train_data, test_data) in prepared_data.items():
        if train_data is not None and test_data is not None:
            print(f"\nProcessing {book_name}:")
            print(f"  Train data: {len(train_data)} records")
            print(f"  Test data: {len(test_data)} records")
            
            # Add training data
            for date, value in train_data.items():
                visualization_data.append({
                    'book_name': book_name,
                    'date': date,
                    'volume': value,
                    'data_type': 'train',
                    'isbn': selected_isbns[0]
                })
            
            # Add test data
            for date, value in test_data.items():
                visualization_data.append({
                    'book_name': book_name,
                    'date': date,
                    'volume': value,
                    'data_type': 'test',
                    'isbn': selected_isbns[0]
                })
    
    # Create DataFrame and check for duplicates
    viz_df = pd.DataFrame(visualization_data)
    if not viz_df.empty:
        viz_df['date'] = pd.to_datetime(viz_df['date'])
        viz_df = viz_df.sort_values(['book_name', 'date'])
        
        print(f"\nVisualization DataFrame shape: {viz_df.shape}")
        
        # Check for duplicates
        before_dedup = len(viz_df)
        duplicates = viz_df.duplicated(subset=['book_name', 'date', 'isbn'], keep=False)
        duplicate_count = duplicates.sum()
        
        print(f"Duplicate entries found: {duplicate_count}")
        
        if duplicate_count > 0:
            print("\nDuplicate dates:")
            print(viz_df[duplicates][['date', 'volume', 'data_type']].head(10))
        else:
            print("✅ No duplicate dates found!")
        
        # Show sample of data around the train/test boundary
        print(f"\nSample data around train/test boundary:")
        print(viz_df.tail(10)[['date', 'volume', 'data_type']])
        
        # Filter for recent dates to check if duplication is fixed
        recent_data = viz_df[viz_df['date'] >= '2023-12-16'].copy()
        print(f"\nRecent data (>= 2023-12-16) shape: {recent_data.shape}")
        
        # Check for duplicates in recent data
        recent_duplicates = recent_data.duplicated(subset=['date'], keep=False)
        recent_duplicate_count = recent_duplicates.sum()
        print(f"Recent duplicate dates: {recent_duplicate_count}")
        
        if recent_duplicate_count > 0:
            print("❌ Still have duplicate dates in recent data")
            print(recent_data[recent_duplicates][['date', 'volume', 'data_type']].head(10))
        else:
            print("✅ No duplicates in recent data!")

if __name__ == "__main__":
    test_modelling_data_fix()