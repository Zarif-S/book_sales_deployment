#!/usr/bin/env python3
"""
Script to run CNN model with comprehensive plotting using real project data.
This script demonstrates how to use the _04_cnn.py module with the comprehensive_model_evaluation
function from hybrid_plotting.py to generate plots for your book sales data.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add the steps directory to the path
sys.path.append('steps')

def load_project_data():
    """Load data from the project's data directory."""
    print("📊 Loading project data...")
    
    # Try to load from processed data first
    processed_data_path = "data/processed"
    if os.path.exists(processed_data_path):
        csv_files = [f for f in os.listdir(processed_data_path) if f.endswith('.csv')]
        if csv_files:
            # Load the first CSV file found
            data_file = os.path.join(processed_data_path, csv_files[0])
            print(f"📁 Loading data from: {data_file}")
            df = pd.read_csv(data_file)
            
            # Try to set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Ensure we have volume column
            if 'volume' not in df.columns and 'Volume' in df.columns:
                df['volume'] = df['Volume']
            
            print(f"✅ Loaded data with {len(df)} rows and columns: {list(df.columns)}")
            return df
    
    # If no processed data, try raw data
    raw_data_path = "data/raw"
    if os.path.exists(raw_data_path):
        csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
        if csv_files:
            data_file = os.path.join(raw_data_path, csv_files[0])
            print(f"📁 Loading data from: {data_file}")
            df = pd.read_csv(data_file)
            
            # Try to set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Ensure we have volume column
            if 'volume' not in df.columns and 'Volume' in df.columns:
                df['volume'] = df['Volume']
            
            print(f"✅ Loaded data with {len(df)} rows and columns: {list(df.columns)}")
            return df
    
    print("⚠️  No data files found in data/processed or data/raw")
    print("📊 Creating sample data instead...")
    return create_sample_data()

def create_sample_data(n_points=200):
    """Create sample time series data if no real data is available."""
    print("📊 Creating sample time series data...")
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_points)]
    
    # Create synthetic time series with trend, seasonality, and noise
    np.random.seed(42)
    trend = np.linspace(100, 200, n_points)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_points) / 30)  # Monthly seasonality
    noise = np.random.normal(0, 10, n_points)
    
    volume = trend + seasonality + noise
    volume = np.maximum(volume, 0)  # Ensure non-negative values
    
    # Create DataFrame
    df = pd.DataFrame({
        'volume': volume
    }, index=dates)
    
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
    """Main function to run CNN with plotting."""
    print("🚀 Starting CNN with comprehensive plotting...")
    print("=" * 60)
    
    # Create output directory
    output_dir = "outputs/cnn_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load or create data
    df = load_project_data()
    
    # Step 2: Split data
    train_data, test_data = split_data_for_cnn(df)
    
    # Step 3: Import and run CNN training
    try:
        from _04_cnn import train_cnn_step
        
        print("\n🔧 Running CNN training with comprehensive plotting...")
        print("=" * 60)
        
        # Run CNN training
        results = train_cnn_step(
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            n_trials=20,  # Adjust based on your needs
            sequence_length=12,
            forecast_horizon=32,
            study_name="cnn_book_sales_optimization"
        )
        
        # Unpack results
        results_df, best_hyperparameters_json, final_model, \
        residuals_df, test_predictions_df, forecast_comparison_df = results
        
        print("\n✅ CNN training completed successfully!")
        print("=" * 60)
        
        # Display results summary
        print("\n📊 Results Summary:")
        print(f"• Model signature: {residuals_df['model_signature'].iloc[0] if len(residuals_df) > 0 else 'N/A'}")
        print(f"• Training residuals: {len(residuals_df)} points")
        print(f"• Test predictions: {len(test_predictions_df)} points")
        
        if len(test_predictions_df) > 0:
            # Calculate some basic metrics
            actual = test_predictions_df['actual'].values
            predicted = test_predictions_df['predicted'].values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            print(f"• Test MAE: {mae:.2f}")
            print(f"• Test RMSE: {rmse:.2f}")
            print(f"• Test MAPE: {mape:.2f}%")
        
        # Check for generated plots
        print("\n📁 Generated files:")
        plot_files = [f for f in os.listdir(output_dir) if f.endswith(('.html', '.png', '.csv'))]
        for file in plot_files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  • {file} ({file_size:.1f} KB)")
        
        print("\n🎉 CNN with plotting completed successfully!")
        print(f"📁 Check the '{output_dir}' directory for generated plots and data files.")
        print("📊 Open the HTML files in your browser to view interactive plots.")
        
    except ImportError as e:
        print(f"❌ Error importing CNN module: {e}")
        print("Make sure you're running this from the project root directory.")
    except Exception as e:
        print(f"❌ Error during CNN training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 