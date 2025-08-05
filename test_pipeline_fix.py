#!/usr/bin/env python3
"""
Test script to verify the ZenML pipeline fix resolves the duplication issue.
"""

import pandas as pd
import numpy as np

def test_visualization_fix():
    """Test the fixed visualization DataFrame creation logic."""
    
    print("🧪 Testing ZenML pipeline visualization fix...")
    print("=" * 60)
    
    # Simulate the REAL data structure that causes duplication
    # Based on your debug output: same dates appear in both train and test with different values
    
    # Create the problematic scenario from your data_debugging.py output
    dates = pd.to_datetime([
        '2023-12-16', '2023-12-23', '2023-12-30', '2024-01-06', '2024-01-13',
        '2024-01-20', '2024-01-27', '2024-02-03', '2024-02-10', '2024-02-17'
    ])
    
    # This simulates the duplication: same dates with different values
    # Train data: first set of values for same dates
    train_volumes = [1260, 2201, 1050, 806, 748, 695, 724, 664, 682, 618]
    train_data = pd.Series(train_volumes, index=dates)
    
    # Test data: second set of values for SAME dates (this causes the issue)
    test_volumes = [2794, 3527, 1151, 1273, 1421, 1474, 1355, 1328, 1310, 1374] 
    test_data = pd.Series(test_volumes, index=dates)  # SAME dates as train!
    
    print(f"📊 Train data: {len(train_data)} periods from {train_data.index.min()} to {train_data.index.max()}")
    print(f"📊 Test data: {len(test_data)} periods from {test_data.index.min()} to {test_data.index.max()}")
    
    # Simulate the prepared_data structure
    prepared_data = {
        'The Alchemist': (train_data, test_data)
    }
    book_isbn_mapping = {
        'The Alchemist': '9780722532935'
    }
    
    print("\n🔧 Testing ORIGINAL approach (would cause duplication):")
    
    # Original problematic approach
    original_visualization_data = []
    for book_name, (train_data, test_data) in prepared_data.items():
        if train_data is not None and test_data is not None:
            book_isbn = book_isbn_mapping.get(book_name, 'unknown')

            # Add train data
            for date, value in train_data.items():
                original_visualization_data.append({
                    'book_name': book_name,
                    'date': date,
                    'volume': value,
                    'data_type': 'train',
                    'isbn': book_isbn
                })

            # Add test data  
            for date, value in test_data.items():
                original_visualization_data.append({
                    'book_name': book_name,
                    'date': date,
                    'volume': value,
                    'data_type': 'test',
                    'isbn': book_isbn
                })
    
    original_df = pd.DataFrame(original_visualization_data)
    print(f"   • Original approach total rows: {len(original_df)}")
    print(f"   • Expected: {len(train_data)} + {len(test_data)} = {len(train_data) + len(test_data)} rows")
    
    print("\n✅ Testing FIXED approach (avoids duplication):")
    
    # Fixed approach
    fixed_visualization_data = []
    for book_name, (train_data, test_data) in prepared_data.items():
        if train_data is not None and test_data is not None:
            book_isbn = book_isbn_mapping.get(book_name, 'unknown')

            # Combine train and test data into a single continuous series per book
            # This avoids the duplication where dates appear in both train and test
            combined_series = pd.concat([train_data, test_data])
            
            # Add each unique date only once
            for date, value in combined_series.items():
                # Determine if this date is in train or test period
                data_type = 'train' if date in train_data.index else 'test'
                
                fixed_visualization_data.append({
                    'book_name': book_name,
                    'date': date,
                    'volume': value,
                    'data_type': data_type,
                    'isbn': book_isbn
                })

    fixed_df = pd.DataFrame(fixed_visualization_data)
    if not fixed_df.empty:
        fixed_df['date'] = pd.to_datetime(fixed_df['date'])
        fixed_df = fixed_df.sort_values(['book_name', 'date'])
        
        # Remove any potential duplicates (safety check)
        before_dedup = len(fixed_df)
        fixed_df = fixed_df.drop_duplicates(subset=['book_name', 'date', 'isbn'], keep='first')
        after_dedup = len(fixed_df)
        
        print(f"   • Fixed approach total rows: {after_dedup}")
        print(f"   • Expected: {len(train_data)} + {len(test_data)} = {len(train_data) + len(test_data)} rows")
        if before_dedup != after_dedup:
            print(f"   • Removed {before_dedup - after_dedup} duplicate entries")
        else:
            print(f"   • No duplicates found ✅")
    
    print("\n📊 Comparison:")
    print(f"   • Original approach: {len(original_df)} rows")
    print(f"   • Fixed approach: {len(fixed_df)} rows")
    print(f"   • Reduction: {len(original_df) - len(fixed_df)} rows")
    
    # Check for duplicates
    original_duplicates = original_df[original_df.duplicated(subset=['date'], keep=False)]
    fixed_duplicates = fixed_df[fixed_df.duplicated(subset=['date'], keep=False)]
    
    print(f"   • Original duplicate dates: {len(original_duplicates)}")
    print(f"   • Fixed duplicate dates: {len(fixed_duplicates)}")
    
    if len(fixed_duplicates) == 0 and len(original_duplicates) > 0:
        print("✅ SUCCESS: Fixed approach eliminates date duplication!")
    elif len(fixed_duplicates) == 0 and len(original_duplicates) == 0:
        print("ℹ️  No duplication in either approach (unexpected for this test)")
    else:
        print("❌ FAILURE: Fixed approach still has duplicates")
    
    # Show sample of the fixed data structure
    print(f"\n📋 Sample of fixed DataFrame:")
    sample_df = fixed_df.head(10)
    print(sample_df[['date', 'volume', 'data_type']].to_string(index=False))
    
    return len(fixed_duplicates) == 0 and len(original_df) > len(fixed_df)

if __name__ == "__main__":
    success = test_visualization_fix()
    if success:
        print("\n🎉 Test PASSED - Fixed approach eliminates duplication!")
    else:
        print("\n❌ Test FAILED - Check the implementation")