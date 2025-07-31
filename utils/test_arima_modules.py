"""
Test script to demonstrate ARIMA modules functionality.
This script creates sample data and shows how to use both modules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_sample_book_data():
    """Create sample book sales data for testing."""
    print("Creating sample book sales data...")
    
    # Generate dates (weekly data from 2012 to 2024)
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='W-SAT')
    
    # Create sample data for The Alchemist
    np.random.seed(42)
    n_weeks = len(dates)
    
    # Base trend with some seasonality and noise
    trend = np.linspace(50, 80, n_weeks)  # Increasing trend
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)  # Annual seasonality
    noise = np.random.normal(0, 10, n_weeks)
    
    # Combine components
    volume_alchemist = np.maximum(0, trend + seasonal + noise).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'ISBN': '9780722532935',
        'Title': 'The Alchemist',
        'Volume': volume_alchemist,
        'End Date': dates
    })
    
    # Set index
    data.set_index('End Date', inplace=True)
    
    print(f"Created sample data with {len(data)} weeks")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Volume range: {data['Volume'].min()} to {data['Volume'].max()}")
    
    return data

def test_arima_modules():
    """Test both ARIMA modules with sample data."""
    print("="*80)
    print("TESTING ARIMA MODULES")
    print("="*80)
    
    # Create sample data
    data = create_sample_book_data()
    
    # Test core ARIMA module
    print("\n1. TESTING CORE ARIMA MODULE")
    print("-" * 40)
    
    try:
        from steps._04_arima import run_complete_arima_analysis
        
        # Run complete analysis
        results = run_complete_arima_analysis(
            data=data,
            isbn='9780722532935',
            forecast_horizon=32,
            use_auto_arima=True,
            seasonal=False,
            title='The Alchemist (Sample Data)'
        )
        
        print("✅ Core ARIMA module test completed successfully!")
        
        # Test plotting module
        print("\n2. TESTING PLOTTING MODULE")
        print("-" * 40)
        
        from steps._04_arima_plots import plot_forecast_results, create_diagnostic_report
        
        # Plot forecast results
        print("Generating forecast plot...")
        plot_forecast_results(
            results['train_data'],
            results['test_data'],
            results['forecast'],
            'The Alchemist - Sample Forecast'
        )
        
        # Create diagnostic report
        print("Generating diagnostic report...")
        create_diagnostic_report(
            results['model']['fitted_model'],
            results['train_data'],
            results['test_data'],
            results['forecast'],
            'The Alchemist - Sample Diagnostic Report'
        )
        
        print("✅ Plotting module test completed successfully!")
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Sample data created: {len(data)} weeks")
        print(f"✅ ARIMA model fitted: ARIMA{results['model']['order']}")
        print(f"✅ Forecast generated: {len(results['forecast']['forecast'])} weeks")
        print(f"✅ Accuracy metrics calculated")
        print(f"✅ Plots generated")
        print("\nAll modules working correctly!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_arima_modules() 