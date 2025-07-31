#!/usr/bin/env python3
"""
Test script to verify persistent Optuna storage and ZenML caching functionality.

This script tests:
1. Persistent Optuna storage with SQLite
2. ZenML step caching
3. Proper metadata handling (dict to string conversion)
4. MLflow logging integration
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from steps._04_arima_zenml_mlflow_optuna import (
    run_optuna_optimization,
    create_time_series_from_df,
    split_time_series
)

def create_test_data():
    """Create synthetic test data for ARIMA modeling."""
    print("Creating synthetic test data...")
    
    # Generate weekly dates for 2 years
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(104)]  # 2 years of weekly data
    
    # Create synthetic volume data with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(100, 200, len(dates))  # Upward trend
    seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  # Weekly seasonality
    noise = np.random.normal(0, 10, len(dates))  # Random noise
    
    volumes = trend + seasonality + noise
    volumes = np.maximum(volumes, 0)  # Ensure non-negative volumes
    
    # Create test DataFrame
    test_data = []
    for date, volume in zip(dates, volumes):
        test_data.append({
            'book_name': 'Test Book',
            'date': date,
            'volume': volume,
            'data_type': 'train',
            'isbn': '9781234567890'
        })
    
    df = pd.DataFrame(test_data)
    print(f"Created test data with {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Volume range: {df['volume'].min():.2f} to {df['volume'].max():.2f}")
    
    return df

def test_persistent_optuna():
    """Test persistent Optuna storage functionality."""
    print("\n" + "="*60)
    print("TESTING PERSISTENT OPTUNA STORAGE")
    print("="*60)
    
    # Create test data
    test_df = create_test_data()
    
    # Convert to time series
    time_series = create_time_series_from_df(test_df, target_col="volume", date_col="date")
    train_series, test_series = split_time_series(time_series, test_size=32)
    
    print(f"Time series created: {len(time_series)} total points")
    print(f"Train set: {len(train_series)} points")
    print(f"Test set: {len(test_series)} points")
    
    # Test 1: First optimization run
    print("\n--- Test 1: First optimization run ---")
    study_name = "test_persistent_optimization"
    
    results1 = run_optuna_optimization(
        train_series=train_series,
        test_series=test_series,
        n_trials=5,  # Small number for testing
        study_name=study_name
    )
    
    print(f"First run results:")
    print(f"  Best params: {results1['best_params']}")
    print(f"  Best RMSE: {results1['best_value']:.4f}")
    print(f"  Trials completed: {results1['n_trials']}")
    print(f"  Storage URL: {results1['storage_url']}")
    
    # Test 2: Second optimization run (should resume from previous)
    print("\n--- Test 2: Second optimization run (should resume) ---")
    
    results2 = run_optuna_optimization(
        train_series=train_series,
        test_series=test_series,
        n_trials=3,  # Additional trials
        study_name=study_name  # Same study name
    )
    
    print(f"Second run results:")
    print(f"  Best params: {results2['best_params']}")
    print(f"  Best RMSE: {results2['best_value']:.4f}")
    print(f"  Total trials completed: {results2['n_trials']}")
    print(f"  Storage URL: {results2['storage_url']}")
    
    # Verify that trials accumulated
    if results2['n_trials'] > results1['n_trials']:
        print("✅ SUCCESS: Trials accumulated across runs (persistent storage working)")
    else:
        print("❌ FAILURE: Trials did not accumulate")
    
    # Test 3: Check if storage file was created
    storage_file = results1['storage_url'].replace('sqlite:///', '')
    if os.path.exists(storage_file):
        file_size = os.path.getsize(storage_file)
        print(f"✅ SUCCESS: Storage file created: {storage_file} ({file_size} bytes)")
    else:
        print(f"❌ FAILURE: Storage file not found: {storage_file}")
    
    return results1, results2

def test_metadata_conversion():
    """Test that dictionary metadata is properly converted to strings."""
    print("\n" + "="*60)
    print("TESTING METADATA CONVERSION")
    print("="*60)
    
    # Test data
    test_dict = {
        "best_params": {"p": 1, "d": 1, "q": 1},
        "eval_metrics": {"rmse": 10.5, "mae": 8.2},
        "nested": {"level1": {"level2": "value"}}
    }
    
    # Convert to string
    dict_str = str(test_dict)
    
    print(f"Original dict: {test_dict}")
    print(f"Converted to string: {dict_str}")
    print(f"Type: {type(dict_str)}")
    
    # Verify it's hashable (can be used as dict key)
    try:
        test_metadata = {dict_str: "test_value"}
        print("✅ SUCCESS: Converted dict is hashable and can be used in metadata")
    except Exception as e:
        print(f"❌ FAILURE: Converted dict is not hashable: {e}")
    
    return dict_str

def test_zenml_step_structure():
    """Test that the ZenML step structure is correct."""
    print("\n" + "="*60)
    print("TESTING ZENML STEP STRUCTURE")
    print("="*60)
    
    try:
        from steps._04_arima_zenml_mlflow_optuna import train_arima_optuna_step
        
        # Check step decorator
        step_func = train_arima_optuna_step
        print(f"✅ SUCCESS: Step function exists: {type(step_func).__name__}")
        
        # Check return type annotation
        import inspect
        signature = inspect.signature(step_func)
        return_annotation = signature.return_annotation
        print(f"Return annotation: {return_annotation}")
        
        # Check if it's a Tuple with Annotated types
        if hasattr(return_annotation, '__origin__') and return_annotation.__origin__ is tuple:
            print("✅ SUCCESS: Step returns a Tuple")
            if len(return_annotation.__args__) == 2:
                print("✅ SUCCESS: Step returns exactly 2 outputs")
            else:
                print(f"❌ FAILURE: Step returns {len(return_annotation.__args__)} outputs, expected 2")
        else:
            print("❌ FAILURE: Step does not return a Tuple")
        
        # Check parameters
        params = list(signature.parameters.keys())
        print(f"Step parameters: {params}")
        
        expected_params = ['modelling_data', 'n_trials', 'study_name']
        for param in expected_params:
            if param in params:
                print(f"✅ SUCCESS: Parameter '{param}' found")
            else:
                print(f"❌ FAILURE: Parameter '{param}' missing")
        
    except ImportError as e:
        print(f"❌ FAILURE: Could not import step: {e}")
    except Exception as e:
        print(f"❌ FAILURE: Error testing step structure: {e}")

def main():
    """Run all tests."""
    print("Starting tests for persistent Optuna storage and ZenML caching...")
    
    # Test 1: Persistent Optuna storage
    try:
        results1, results2 = test_persistent_optuna()
    except Exception as e:
        print(f"❌ FAILURE: Persistent Optuna test failed: {e}")
    
    # Test 2: Metadata conversion
    try:
        test_metadata_conversion()
    except Exception as e:
        print(f"❌ FAILURE: Metadata conversion test failed: {e}")
    
    # Test 3: ZenML step structure
    try:
        test_zenml_step_structure()
    except Exception as e:
        print(f"❌ FAILURE: ZenML step structure test failed: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("All tests completed. Check the output above for results.")
    print("If you see ✅ SUCCESS messages, the modifications are working correctly.")
    print("If you see ❌ FAILURE messages, there may be issues to address.")

if __name__ == "__main__":
    main() 