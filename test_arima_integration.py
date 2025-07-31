#!/usr/bin/env python3
"""
Test script to verify ARIMA integration in the pipeline.
This script tests the ARIMA functions without running the full pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the ARIMA functions from the pipeline
from pipelines.zenml_pipeline_with_modelling_prep import (
    prepare_time_series_data,
    evaluate_forecast,
    objective,
    run_optuna_optimization,
    train_final_arima_model
)

def create_test_data():
    """Create synthetic test data for ARIMA testing."""
    print("Creating synthetic test data...")
    
    # Create date range
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Create synthetic sales data with trend and seasonality
    np.random.seed(42)
    n_weeks = len(dates)
    
    # Base trend
    trend = np.linspace(100, 200, n_weeks)
    
    # Seasonal component (52-week cycle)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    
    # Random noise
    noise = np.random.normal(0, 10, n_weeks)
    
    # Combine components
    volume = trend + seasonal + noise
    volume = np.maximum(volume, 0)  # Ensure non-negative
    
    # Create DataFrame
    test_data = []
    for i, date in enumerate(dates):
        test_data.append({
            'book_name': 'Test Book',
            'date': date,
            'volume': volume[i],
            'data_type': 'train' if i < len(dates) - 32 else 'test',
            'isbn': '9781234567890'
        })
    
    df = pd.DataFrame(test_data)
    print(f"Created test data with {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Volume range: {df['volume'].min():.2f} to {df['volume'].max():.2f}")
    
    return df

def test_arima_functions():
    """Test the ARIMA helper functions."""
    print("\n" + "="*50)
    print("Testing ARIMA Helper Functions")
    print("="*50)
    
    # Create test data
    test_df = create_test_data()
    
    try:
        # Test 1: Prepare time series data
        print("\n1. Testing prepare_time_series_data...")
        train_series, test_series = prepare_time_series_data(test_df, target_col="volume", split_weeks=32)
        print(f"✓ Train series: {len(train_series)} weeks")
        print(f"✓ Test series: {len(test_series)} weeks")
        print(f"✓ Train range: {train_series.index.min()} to {train_series.index.max()}")
        print(f"✓ Test range: {test_series.index.min()} to {test_series.index.max()}")
        
        # Test 2: Evaluate forecast function
        print("\n2. Testing evaluate_forecast...")
        # Create dummy predictions
        dummy_pred = test_series.values + np.random.normal(0, 5, len(test_series))
        metrics = evaluate_forecast(test_series.values, dummy_pred)
        print(f"✓ MAE: {metrics['mae']:.2f}")
        print(f"✓ RMSE: {metrics['rmse']:.2f}")
        print(f"✓ MAPE: {metrics['mape']:.2f}%")
        
        # Test 3: Test Optuna optimization (with fewer trials for speed)
        print("\n3. Testing Optuna optimization...")
        best_params = run_optuna_optimization(train_series, test_series, n_trials=5)
        print(f"✓ Best parameters: {best_params}")
        
        # Test 4: Test final model training
        print("\n4. Testing final model training...")
        final_model = train_final_arima_model(train_series, best_params)
        print(f"✓ Model trained successfully")
        print(f"✓ Model AIC: {final_model.aic:.2f}")
        print(f"✓ Model BIC: {final_model.bic:.2f}")
        
        # Test 5: Test forecasting
        print("\n5. Testing forecasting...")
        forecast = final_model.forecast(steps=len(test_series))
        forecast_metrics = evaluate_forecast(test_series.values, forecast.values)
        print(f"✓ Forecast MAE: {forecast_metrics['mae']:.2f}")
        print(f"✓ Forecast RMSE: {forecast_metrics['rmse']:.2f}")
        print(f"✓ Forecast MAPE: {forecast_metrics['mape']:.2f}%")
        
        print("\n" + "="*50)
        print("✓ All ARIMA functions tested successfully!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_import():
    """Test that the pipeline can be imported successfully."""
    print("\n" + "="*50)
    print("Testing Pipeline Import")
    print("="*50)
    
    try:
        from pipelines.zenml_pipeline_with_modelling_prep import (
            book_sales_pipeline_with_modelling_prep,
            train_arima_optuna_step
        )
        print("✓ Pipeline imported successfully")
        print("✓ ARIMA step imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ARIMA Integration Test")
    print("="*50)
    
    # Test pipeline import
    import_success = test_pipeline_import()
    
    if import_success:
        # Test ARIMA functions
        arima_success = test_arima_functions()
        
        if arima_success:
            print("\n🎉 All tests passed! ARIMA integration is working correctly.")
        else:
            print("\n❌ ARIMA function tests failed.")
    else:
        print("\n❌ Pipeline import failed.")
    
    print("\nTest completed.") 