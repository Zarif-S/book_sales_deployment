import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler


def load_arima_residuals_from_csv(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load residuals from CSV file saved by the ARIMA pipeline.
    """
    print("🔍 Loading ARIMA residuals from CSV file...")
    
    import os
    
    # Look for residuals CSV in the data directory
    residuals_csv_path = os.path.join(data_dir, "arima_residuals.csv")
    
    if not os.path.exists(residuals_csv_path):
        print(f"⚠️  Residuals CSV not found at: {residuals_csv_path}")
        print("📋 Checking for alternative locations...")
        
        # Check other possible locations
        alternative_paths = [
            "data/arima_residuals.csv",
            "arima_residuals.csv",
            os.path.join(data_dir, "..", "arima_residuals.csv")
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                residuals_csv_path = alt_path
                print(f"✅ Found residuals CSV at: {residuals_csv_path}")
                break
        else:
            raise FileNotFoundError(f"Residuals CSV not found. Expected at: {residuals_csv_path}")
    
    try:
        residuals_df = pd.read_csv(residuals_csv_path)
        
        # Convert date column to datetime if it's not already
        if 'date' in residuals_df.columns:
            residuals_df['date'] = pd.to_datetime(residuals_df['date'])
        
        print(f"✅ Loaded residuals from CSV: {residuals_csv_path}")
        print(f"📈 Residuals shape: {residuals_df.shape}")
        print(f"📅 Date range: {residuals_df['date'].min()} to {residuals_df['date'].max()}")
        print(f"📊 Residuals stats - Mean: {residuals_df['residuals'].mean():.4f}, Std: {residuals_df['residuals'].std():.4f}")
        print(f"🔧 Model signature: {residuals_df['model_signature'].iloc[0]}")
        
        return residuals_df
        
    except Exception as e:
        print(f"⚠️  Could not load residuals CSV: {e}")
        raise ValueError(f"Failed to load residuals CSV: {e}")


def prepare_residuals_for_lstm(residuals_df: pd.DataFrame, sequence_length: int = 10) -> Dict[str, np.ndarray]:
    """
    Prepare residuals data for LSTM model training.
    """
    print(f"🔧 Preparing residuals for LSTM with sequence length: {sequence_length}")
    
    # Convert to numpy array and normalize
    residuals_array = residuals_df['residuals'].values
    print(f"📊 Original residuals shape: {residuals_array.shape}")
    print(f"📊 Residuals stats - Mean: {residuals_array.mean():.4f}, Std: {residuals_array.std():.4f}")
    
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals_array.reshape(-1, 1)).flatten()
    print(f"📊 Scaled residuals stats - Mean: {residuals_scaled.mean():.4f}, Std: {residuals_scaled.std():.4f}")
    
    # Create sequences
    X, y = [], []
    for i in range(len(residuals_scaled) - sequence_length):
        X.append(residuals_scaled[i:(i + sequence_length)])
        y.append(residuals_scaled[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    print(f"📊 Created sequences - X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    print(f"📊 Train/Test split at index: {split_idx}")
    print(f"📊 Training samples: {split_idx}, Test samples: {len(X) - split_idx}")
    
    result = {
        'X_train': X[:split_idx],
        'X_test': X[split_idx:],
        'y_train': y[:split_idx],
        'y_test': y[split_idx:],
        'scaler': scaler,
        'sequence_length': sequence_length
    }
    
    print("✅ LSTM data preparation completed successfully!")
    return result


def create_sample_residuals_for_testing() -> pd.DataFrame:
    """
    Create sample residuals data for testing when no pipeline data is available.
    """
    print("🧪 Creating sample residuals data for testing...")
    
    # Create realistic sample data
    sample_dates = pd.date_range(start='2020-01-01', periods=200, freq='W')
    sample_residuals = np.random.normal(0, 50, 200)  # Mock residuals
    
    residuals_df = pd.DataFrame({
        'date': sample_dates,
        'residuals': sample_residuals,
        'model_signature': 'SARIMAX_(2,1,3)_(1,1,3,52)_TEST'
    })
    
    print(f"📊 Created sample residuals with shape: {residuals_df.shape}")
    return residuals_df


def main():
    """
    Main function to demonstrate the LSTM preparation workflow.
    """
    print("🚀 Starting LSTM Residuals Preparation Demo")
    print("=" * 50)
    
    try:
        # Try to load from CSV first
        print("📋 Attempting to load residuals from CSV...")
        residuals_df = load_arima_residuals_from_csv()
        print("✅ Successfully loaded residuals from CSV!")
        
    except Exception as e:
        print(f"⚠️  Could not load from CSV: {e}")
        print("🧪 Falling back to sample data for demonstration...")
        residuals_df = create_sample_residuals_for_testing()
    
    print("\n" + "=" * 50)
    
    # Prepare data for LSTM
    print("📋 Preparing data for LSTM model...")
    lstm_data = prepare_residuals_for_lstm(residuals_df, sequence_length=10)
    
    print("\n" + "=" * 50)
    print("📊 Final LSTM Data Summary:")
    print(f"   • X_train shape: {lstm_data['X_train'].shape}")
    print(f"   • X_test shape: {lstm_data['X_test'].shape}")
    print(f"   • y_train shape: {lstm_data['y_train'].shape}")
    print(f"   • y_test shape: {lstm_data['y_test'].shape}")
    print(f"   • Sequence length: {lstm_data['sequence_length']}")
    print(f"   • Scaler fitted: {hasattr(lstm_data['scaler'], 'mean_')}")
    
    print("\n🎉 LSTM data preparation completed successfully!")
    return lstm_data


if __name__ == "__main__":
    main()
