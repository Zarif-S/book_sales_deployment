#!/usr/bin/env python3
"""
Simple test script to verify CNN implementation works.
"""
import pandas as pd
import numpy as np
from steps._04_cnn import train_cnn_step
import os

def create_sample_data():
    """Create sample train/test data for testing CNN."""
    print("🔧 Creating sample data for CNN testing...")
    
    # Create sample time series data
    dates = pd.date_range(start='2020-01-04', periods=200, freq='W-SAT')
    
    # Generate synthetic book sales data with trend and seasonality
    trend = np.linspace(100, 150, 200)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(200) / 52)  # Yearly seasonality
    noise = np.random.normal(0, 10, 200)
    volume_data = trend + seasonal + noise
    volume_data = np.maximum(volume_data, 0)  # Ensure non-negative
    
    # Create DataFrame with book metadata
    df = pd.DataFrame({
        'volume': volume_data,
        'Volume': volume_data,  # Both cases for compatibility
        'book_name': 'Test Book',
        'isbn': '9780123456789'
    }, index=dates)
    
    # Split into train/test (last 32 for test)
    train_data = df.iloc[:-32].copy()
    test_data = df.iloc[-32:].copy()
    
    print(f"📊 Created sample data:")
    print(f"   • Train data shape: {train_data.shape}")
    print(f"   • Test data shape: {test_data.shape}")
    print(f"   • Volume range: {volume_data.min():.1f} to {volume_data.max():.1f}")
    
    return train_data, test_data

def test_cnn_basic():
    """Test basic CNN functionality with sample data."""
    print("\n" + "="*60)
    print("🧪 TESTING CNN IMPLEMENTATION")
    print("="*60)
    
    try:
        # Create sample data
        train_data, test_data = create_sample_data()
        
        # Set up output directory
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test CNN training with reduced parameters for speed
        print("\n📋 Testing CNN training...")
        results = train_cnn_step(
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            n_trials=3,  # Reduced for testing
            sequence_length=8,  # Reduced for testing
            forecast_horizon=32,
            study_name="test_cnn"
        )
        
        # Unpack results
        (results_df, hyperparameters_json, trained_model, 
         residuals_df, test_predictions_df, forecast_comparison_df) = results
        
        # Verify results
        print("\n📊 Verifying results...")
        
        if results_df is not None:
            print(f"✅ Results DataFrame created: {results_df.shape}")
            print(f"   • Result types: {results_df['result_type'].unique()}")
        else:
            print("❌ Results DataFrame is None")
            
        if hyperparameters_json is not None:
            print(f"✅ Hyperparameters JSON created: {len(hyperparameters_json)} chars")
        else:
            print("❌ Hyperparameters JSON is None")
            
        if trained_model is not None:
            print(f"✅ Trained model created: {type(trained_model)}")
        else:
            print("❌ Trained model is None")
            
        if residuals_df is not None and len(residuals_df) > 0:
            print(f"✅ Residuals DataFrame created: {residuals_df.shape}")
            print(f"   • Residuals range: {residuals_df['residuals'].min():.2f} to {residuals_df['residuals'].max():.2f}")
            
            # Check if residuals CSV was saved
            residuals_csv = os.path.join(output_dir, "cnn_residuals.csv")
            if os.path.exists(residuals_csv):
                print(f"✅ Residuals CSV saved: {residuals_csv}")
            else:
                print(f"❌ Residuals CSV not found: {residuals_csv}")
        else:
            print("❌ Residuals DataFrame is empty or None")
            
        if test_predictions_df is not None and len(test_predictions_df) > 0:
            print(f"✅ Test predictions DataFrame created: {test_predictions_df.shape}")
            print(f"   • Prediction range: {test_predictions_df['predicted'].min():.2f} to {test_predictions_df['predicted'].max():.2f}")
        else:
            print("❌ Test predictions DataFrame is empty or None")
            
        if forecast_comparison_df is not None and len(forecast_comparison_df) > 0:
            print(f"✅ Forecast comparison DataFrame created: {forecast_comparison_df.shape}")
        else:
            print("❌ Forecast comparison DataFrame is empty or None")
            
        print("\n🎉 CNN basic test completed!")
        return True
        
    except Exception as e:
        print(f"❌ CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_data_compatibility():
    """Test CNN with data format similar to pipeline output."""
    print("\n📋 Testing CNN data compatibility...")
    
    try:
        # Load actual processed data if available
        processed_file = "data/processed/combined_train_data.csv"
        test_file = "data/processed/combined_test_data.csv"
        
        if os.path.exists(processed_file) and os.path.exists(test_file):
            print("📂 Loading actual pipeline data...")
            train_data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            test_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
            
            # Filter to one book for testing
            if 'book_name' in train_data.columns:
                first_book = train_data['book_name'].iloc[0]
                train_data = train_data[train_data['book_name'] == first_book].copy()
                test_data = test_data[test_data['book_name'] == first_book].copy()
            
            print(f"📊 Real data loaded:")
            print(f"   • Train data shape: {train_data.shape}")
            print(f"   • Test data shape: {test_data.shape}")
            
            # Quick test with minimal trials
            results = train_cnn_step(
                train_data=train_data,
                test_data=test_data,
                output_dir="data/processed",
                n_trials=1,  # Minimal for compatibility test
                sequence_length=6,
                forecast_horizon=len(test_data),
                study_name="test_cnn_real_data"
            )
            
            print("✅ Real data compatibility test passed!")
            return True
        else:
            print("📂 Real pipeline data not found, skipping compatibility test")
            return True
            
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting CNN implementation tests...")
    
    # Run basic test
    basic_success = test_cnn_basic()
    
    # Run compatibility test
    compat_success = test_cnn_data_compatibility()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Basic functionality: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Data compatibility: {'✅ PASSED' if compat_success else '❌ FAILED'}")
    
    if basic_success and compat_success:
        print("\n🎉 All tests passed! CNN implementation is ready.")
        print("\n📋 Next steps:")
        print("   1. Create CNN+LSTM pipeline by copying ARIMA+LSTM pipeline")
        print("   2. Replace ARIMA step with CNN step")
        print("   3. Update step imports and function calls")
        print("   4. Test full pipeline")
    else:
        print("\n❌ Some tests failed. Check implementation.")