#!/usr/bin/env python3
"""
Test script to verify the ARIMA pipeline integration works correctly.
This script tests the key components without running the full pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_arima_functions():
    """Test the ARIMA functions imported from the module."""
    print("Testing ARIMA function imports...")
    
    try:
        from steps._04_arima_zenml_mlflow_optuna import (
            evaluate_forecast,
            run_optuna,
            train_best_arima
        )
        print("✓ Successfully imported ARIMA functions")
    except ImportError as e:
        print(f"✗ Failed to import ARIMA functions: {e}")
        return False
    
    # Create synthetic time series data for testing
    print("\nCreating synthetic test data...")
    dates = pd.date_range(start='2012-01-01', end='2024-12-31', freq='W')
    np.random.seed(42)
    
    # Create realistic book sales data with trend and seasonality
    trend = np.linspace(100, 150, len(dates))
    seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  # Weekly seasonality
    noise = np.random.normal(0, 10, len(dates))
    sales_data = trend + seasonality + noise
    sales_data = np.maximum(sales_data, 0)  # Ensure non-negative sales
    
    # Create train/test split
    split_point = len(dates) - 32  # Last 32 weeks as test
    train_dates = dates[:split_point]
    test_dates = dates[split_point:]
    train_sales = sales_data[:split_point]
    test_sales = sales_data[split_point:]
    
    train_series = pd.Series(train_sales, index=train_dates)
    test_series = pd.Series(test_sales, index=test_dates)
    
    print(f"✓ Created synthetic data: {len(train_series)} train points, {len(test_series)} test points")
    
    # Test evaluation function
    print("\nTesting evaluation function...")
    try:
        # Create dummy predictions
        dummy_predictions = test_series.values + np.random.normal(0, 5, len(test_series))
        metrics = evaluate_forecast(test_series.values, dummy_predictions)
        print(f"✓ Evaluation function works: {metrics}")
    except Exception as e:
        print(f"✗ Evaluation function failed: {e}")
        return False
    
    # Test Optuna optimization (with fewer trials for speed)
    print("\nTesting Optuna optimization...")
    try:
        best_params = run_optuna(train_series, test_series, n_trials=5)
        print(f"✓ Optuna optimization works: {best_params}")
    except Exception as e:
        print(f"✗ Optuna optimization failed: {e}")
        return False
    
    # Test model training
    print("\nTesting model training...")
    try:
        fitted_model = train_best_arima(train_series, best_params)
        print(f"✓ Model training works: {type(fitted_model)}")
        
        # Test forecasting
        forecast = fitted_model.forecast(steps=len(test_series))
        print(f"✓ Forecasting works: {len(forecast)} predictions")
        
        # Test residuals
        residuals = fitted_model.resid
        print(f"✓ Residuals extraction works: {len(residuals)} residuals")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False
    
    print("\n✓ All ARIMA functions tested successfully!")
    return True

def test_pipeline_structure():
    """Test that the pipeline structure is correct."""
    print("\nTesting pipeline structure...")
    
    try:
        from pipelines.zenml_pipeline_with_modelling_prep import (
            book_sales_pipeline_with_modelling_prep,
            train_arima_models_step
        )
        print("✓ Successfully imported pipeline components")
    except ImportError as e:
        print(f"✗ Failed to import pipeline components: {e}")
        return False
    
    # Test that the pipeline function exists and has the right signature
    import inspect
    sig = inspect.signature(book_sales_pipeline_with_modelling_prep)
    params = list(sig.parameters.keys())
    
    expected_params = ['output_dir', 'selected_isbns', 'column_name', 'split_size', 'n_trials']
    for param in expected_params:
        if param not in params:
            print(f"✗ Missing parameter: {param}")
            return False
    
    print("✓ Pipeline structure is correct")
    return True

def test_data_preparation():
    """Test the data preparation functions."""
    print("\nTesting data preparation functions...")
    
    try:
        from steps._03_5_modelling_prep import prepare_data_after_2012, prepare_multiple_books_data
        print("✓ Successfully imported data preparation functions")
    except ImportError as e:
        print(f"✗ Failed to import data preparation functions: {e}")
        return False
    
    # Create synthetic book data
    dates = pd.date_range(start='2010-01-01', end='2024-12-31', freq='W')
    np.random.seed(42)
    
    # Create multiple books data
    books_data = {}
    for i, book_name in enumerate(['The Alchemist', 'The Very Hungry Caterpillar']):
        sales_data = 100 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52) + np.random.normal(0, 10, len(dates))
        sales_data = np.maximum(sales_data, 0)
        
        df = pd.DataFrame({
            'Volume': sales_data,
            'Title': book_name,
            'ISBN': f'978000000000{i}'
        }, index=dates)
        
        books_data[book_name] = df
    
    # Test multiple books preparation
    try:
        prepared_data = prepare_multiple_books_data(books_data, column_name='Volume', split_size=32)
        print(f"✓ Multiple books preparation works: {len(prepared_data)} books prepared")
        
        for book_name, (train, test) in prepared_data.items():
            if train is not None and test is not None:
                print(f"  - {book_name}: {len(train)} train, {len(test)} test")
        
    except Exception as e:
        print(f"✗ Multiple books preparation failed: {e}")
        return False
    
    print("✓ Data preparation functions work correctly")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("ARIMA Pipeline Integration Test")
    print("=" * 60)
    
    tests = [
        ("ARIMA Functions", test_arima_functions),
        ("Pipeline Structure", test_pipeline_structure),
        ("Data Preparation", test_data_preparation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The ARIMA pipeline integration is ready.")
        print("You can now run the full pipeline with confidence.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above before running the full pipeline.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 