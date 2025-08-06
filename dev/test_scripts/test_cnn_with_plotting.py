#!/usr/bin/env python3
"""
Test script to demonstrate CNN model with comprehensive plotting functionality.
This script shows how to use the _04_cnn.py module with the comprehensive_model_evaluation
function from hybrid_plotting.py to generate plots.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add the steps directory to the path
sys.path.append('steps')

def create_sample_data(n_points=200):
    """Create sample time series data for testing."""
    print("📊 Creating sample time series data...")

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]

    # Create synthetic time series with trend, seasonality, and noise
    np.random.seed(42)
    trend = np.linspace(100, 200, n_points)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_points) / 30)  # Monthly seasonality
    noise = np.random.normal(0, 10, n_points)

    volume = trend + seasonality + noise
    volume = np.maximum(volume, 0)  # Ensure non-negative values

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'volume': volume
    })

    # Set date as index
    df.set_index('date', inplace=True)

    print(f"✅ Created sample data with {len(df)} points")
    print(f"📊 Data range: {df['volume'].min():.2f} to {df['volume'].max():.2f}")

    return df

def split_data_for_cnn(df, train_ratio=0.8):
    """Split data into train and test sets for CNN."""
    print("📊 Splitting data for CNN training...")

    n_train = int(len(df) * train_ratio)
    train_data = df.iloc[:n_train]
    test_data = df.iloc[n_train:]

    print(f"✅ Training set: {len(train_data)} points")
    print(f"✅ Test set: {len(test_data)} points")

    return train_data, test_data

def main():
    """Main function to test CNN with plotting."""
    print("🚀 Starting CNN with plotting test...")
    print("=" * 60)

    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Create sample data
    df = create_sample_data(n_points=200)

    # Step 2: Split data
    train_data, test_data = split_data_for_cnn(df)

    # Step 3: Import and run CNN training
    try:
        from _04_cnn import train_cnn_step

        print("\n🔧 Running CNN training with plotting...")
        print("=" * 60)

        # Run CNN training with reduced trials for faster testing
        results = train_cnn_step(
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            n_trials=5,  # Reduced for faster testing
            sequence_length=12,
            forecast_horizon=32,
            study_name="test_cnn_optimization"
        )

        # Unpack results
        results_df, best_hyperparameters_json, final_model, \
        residuals_df, forecast_df = results

        print("\n✅ CNN training completed successfully!")
        print("=" * 60)

        # Display results summary
        print("\n📊 Results Summary:")
        print(f"• Model signature: {residuals_df['model_signature'].iloc[0] if len(residuals_df) > 0 else 'N/A'}")
        print(f"• Training residuals: {len(residuals_df)} points")
        print(f"• Forecast predictions: {len(forecast_df)} points")

        if len(forecast_df) > 0:
            # Calculate some basic metrics
            actual = forecast_df['actual'].values
            predicted = forecast_df['predicted'].values

            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            print(f"• Test MAE: {mae:.2f}")
            print(f"• Test RMSE: {rmse:.2f}")

        # Check for generated plots
        print("\n📁 Generated files:")
        plot_files = [f for f in os.listdir(output_dir) if f.endswith(('.html', '.png', '.csv'))]
        for file in plot_files:
            print(f"  • {file}")

        print("\n🎉 Test completed successfully!")
        print(f"📁 Check the '{output_dir}' directory for generated plots and data files.")

    except ImportError as e:
        print(f"❌ Error importing CNN module: {e}")
        print("Make sure you're running this from the project root directory.")
    except Exception as e:
        print(f"❌ Error during CNN training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
