import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
import os


def load_arima_residuals_from_csv(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load residuals from CSV file saved by the ARIMA pipeline.
    """
    print("🔍 Loading ARIMA residuals from CSV file...")
    
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
        
        # Convert date column to datetime and set as index
        if 'date' in residuals_df.columns:
            residuals_df['date'] = pd.to_datetime(residuals_df['date'])
            residuals_df = residuals_df.set_index('date')
        
        print(f"✅ Loaded residuals from CSV: {residuals_csv_path}")
        print(f"📈 Residuals shape: {residuals_df.shape}")
        print(f"📅 Date range: {residuals_df.index.min()} to {residuals_df.index.max()}")
        print(f"📊 Residuals stats - Mean: {residuals_df['residuals'].mean():.4f}, Std: {residuals_df['residuals'].std():.4f}")
        print(f"🔧 Model signature: {residuals_df['model_signature'].iloc[0]}")
        
        return residuals_df
        
    except Exception as e:
        print(f"⚠️  Could not load residuals CSV: {e}")
        raise ValueError(f"Failed to load residuals CSV: {e}")


def create_input_sequences(lookback: int, forecast: int, data: np.ndarray) -> Dict[str, list]:
    """
    Create input-output sequences for LSTM training.
    
    Args:
        lookback: Number of past observations to use as input
        forecast: Number of future observations to predict
        data: 1D array of scaled residuals data
    
    Returns:
        Dictionary with input_sequences and output_sequences
    """
    print(f"🔧 Creating sequences with lookback={lookback}, forecast={forecast}")
    
    input_sequences = []
    output_sequences = []
    
    # Create sequences where each input is 'lookback' timesteps and output is 'forecast' timesteps
    for i in range(len(data) - lookback - forecast + 1):
        input_seq = data[i:(i + lookback)]              # Past 'lookback' observations
        output_seq = data[(i + lookback):(i + lookback + forecast)]  # Next 'forecast' observations
        
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    
    print(f"📊 Created {len(input_sequences)} sequences")
    print(f"📊 Input sequence shape: ({len(input_sequences)}, {lookback})")
    print(f"📊 Output sequence shape: ({len(output_sequences)}, {forecast})")
    
    return {
        'input_sequences': input_sequences,
        'output_sequences': output_sequences
    }


def prepare_residuals_for_lstm_training(
    residuals_df: pd.DataFrame, 
    lookback: int = 12, 
    forecast: int = 32,
    train_test_split: float = 0.8
) -> Dict[str, np.ndarray]:
    """
    Prepare residuals data for LSTM training following the hybrid SARIMA+LSTM workflow.
    
    Args:
        residuals_df: DataFrame with residuals from ARIMA model
        lookback: Number of past observations to use as input (default: 12)
        forecast: Number of future observations to predict (default: 32)
        train_test_split: Proportion of data to use for training (default: 0.8)
        
    Returns:
        Dictionary containing prepared training data
    """
    print("🔧 Preparing residuals for LSTM training with hybrid SARIMA+LSTM workflow")
    print(f"📊 Parameters: lookback={lookback}, forecast={forecast}, train_split={train_test_split}")
    
    # Initialize the MinMaxScaler with feature_range=(0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler on the SARIMA residuals and transform
    residuals_values = residuals_df['residuals'].values.reshape(-1, 1)
    residuals_scaled = scaler.fit_transform(residuals_values).flatten()
    
    print(f"📊 Original residuals shape: {residuals_values.shape}")
    print(f"📊 Residuals stats - Mean: {residuals_df['residuals'].mean():.4f}, Std: {residuals_df['residuals'].std():.4f}")
    print(f"📊 Scaled residuals stats - Mean: {residuals_scaled.mean():.4f}, Std: {residuals_scaled.std():.4f}")
    
    # Create a DataFrame with scaled residuals for easier handling
    result_df = residuals_df.copy()
    result_df['residuals_scaled'] = residuals_scaled
    
    # Create input-output sequences for the scaled residuals
    sequences = create_input_sequences(lookback, forecast, residuals_scaled)
    
    # Convert to numpy arrays
    X_combined = np.array(sequences["input_sequences"])
    Y_combined = np.array(sequences["output_sequences"])
    
    # Reshape the input sequences for LSTM [samples, time steps, features]
    X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)
    
    print(f"📊 Combined sequences - X shape: {X_combined.shape}, Y shape: {Y_combined.shape}")
    
    # Calculate split point based on train_test_split
    train_length = int(train_test_split * len(X_combined))
    
    # Split the sequences into train and test sets
    X_train = X_combined[:train_length]
    X_test = X_combined[train_length:]
    Y_train = Y_combined[:train_length]
    Y_test = Y_combined[train_length:]
    
    print(f"📊 Train/Test split at index: {train_length}")
    print(f"📊 X_train shape: {X_train.shape}")
    print(f"📊 Y_train shape: {Y_train.shape}")
    print(f"📊 X_test shape: {X_test.shape}")
    print(f"📊 Y_test shape: {Y_test.shape}")
    
    # Return comprehensive data structure
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'scaler': scaler,
        'lookback': lookback,
        'forecast': forecast,
        'residuals_df': result_df,
        'train_length': train_length,
        'original_residuals': residuals_df['residuals'].values,
        'scaled_residuals': residuals_scaled
    }
    
    print("✅ LSTM data preparation completed successfully!")
    return result


def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original scale.
    
    Args:
        predictions: Scaled predictions from LSTM model
        scaler: Fitted MinMaxScaler used for scaling
        
    Returns:
        Predictions in original scale
    """
    print(f"🔄 Inverse transforming predictions with shape: {predictions.shape}")
    
    # Reshape predictions if needed for inverse transform
    if predictions.ndim == 1:
        predictions_reshaped = predictions.reshape(-1, 1)
    elif predictions.ndim == 2 and predictions.shape[1] > 1:
        # If predictions have multiple features, flatten first
        predictions_reshaped = predictions.reshape(-1, 1)
    else:
        predictions_reshaped = predictions
    
    # Inverse transform
    predictions_original = scaler.inverse_transform(predictions_reshaped)
    
    # Return in original shape if needed
    if predictions.ndim == 1:
        predictions_original = predictions_original.flatten()
    elif predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions_original = predictions_original.reshape(predictions.shape)
    
    print(f"📊 Inverse transformed predictions shape: {predictions_original.shape}")
    print(f"📊 Inverse transformed stats - Mean: {predictions_original.mean():.4f}, Std: {predictions_original.std():.4f}")
    
    return predictions_original


def create_sample_residuals_for_testing() -> pd.DataFrame:
    """
    Create sample residuals data for testing when no pipeline data is available.
    """
    print("🧪 Creating sample residuals data for testing...")
    
    # Create realistic sample data with proper length for testing
    sample_dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
    # Create residuals with some structure (not purely random)
    t = np.arange(100)
    sample_residuals = 50 * np.sin(0.1 * t) + 20 * np.random.normal(0, 1, 100)
    
    residuals_df = pd.DataFrame({
        'date': sample_dates,
        'residuals': sample_residuals,
        'model_signature': 'SARIMAX_(2,1,3)_(1,1,3,52)_TEST'
    })
    
    # Set date as index
    residuals_df = residuals_df.set_index('date')
    
    print(f"📊 Created sample residuals with shape: {residuals_df.shape}")
    print(f"📊 Sample residuals stats - Mean: {residuals_df['residuals'].mean():.4f}, Std: {residuals_df['residuals'].std():.4f}")
    
    return residuals_df


def demonstrate_lstm_workflow():
    """
    Demonstrate the complete LSTM workflow with proper scaling and sequence creation.
    """
    print("🚀 Starting LSTM Residuals Preparation Demo")
    print("=" * 60)
    
    try:
        # Try to load real residuals from CSV first
        print("📋 Attempting to load residuals from CSV...")
        residuals_df = load_arima_residuals_from_csv()
        print("✅ Successfully loaded residuals from CSV!")
        
    except Exception as e:
        print(f"⚠️  Could not load from CSV: {e}")
        print("🧪 Falling back to sample data for demonstration...")
        residuals_df = create_sample_residuals_for_testing()
    
    print("\n" + "=" * 60)
    
    # Prepare data for LSTM with parameters similar to your workflow
    print("📋 Preparing data for LSTM model...")
    lstm_data = prepare_residuals_for_lstm_training(
        residuals_df, 
        lookback=12, 
        forecast=32,
        train_test_split=0.8
    )
    
    print("\n" + "=" * 60)
    print("📊 Final LSTM Data Summary:")
    print(f"   • X_train shape: {lstm_data['X_train'].shape}")
    print(f"   • X_test shape: {lstm_data['X_test'].shape}")
    print(f"   • Y_train shape: {lstm_data['Y_train'].shape}")
    print(f"   • Y_test shape: {lstm_data['Y_test'].shape}")
    print(f"   • Lookback window: {lstm_data['lookback']}")
    print(f"   • Forecast horizon: {lstm_data['forecast']}")
    print(f"   • Train samples: {lstm_data['train_length']}")
    print(f"   • Scaler fitted: {hasattr(lstm_data['scaler'], 'scale_')}")
    print(f"   • Original data length: {len(lstm_data['original_residuals'])}")
    
    # Demonstrate inverse transformation
    print("\n📋 Testing inverse transformation...")
    sample_predictions = np.random.random((5, 32))  # Mock predictions
    inverse_predictions = inverse_transform_predictions(sample_predictions, lstm_data['scaler'])
    print(f"   • Sample predictions transformed from {sample_predictions.shape} to {inverse_predictions.shape}")
    
    print("\n🎉 LSTM data preparation workflow completed successfully!")
    print("🔧 Ready for LSTM model training with Keras Tuner!")
    
    return lstm_data


def build_lstm_model(hp, input_shape: Tuple[int, int], output_shape: int):
    """
    Build LSTM model with hyperparameter tuning.
    
    Args:
        hp: Keras Tuner hyperparameters object
        input_shape: Shape of input sequences (timesteps, features)
        output_shape: Number of output predictions
        
    Returns:
        Compiled Keras model
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential()
    
    # First LSTM layer
    model.add(layers.LSTM(
        units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))
    
    # Second LSTM layer
    model.add(layers.LSTM(
        units=hp.Int('lstm_units_2', min_value=16, max_value=64, step=16),
        return_sequences=False
    ))
    model.add(layers.Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1)))
    
    # Dense layers
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    model.add(layers.Dropout(hp.Float('dropout_3', 0, 0.3, step=0.1)))
    
    # Output layer
    model.add(layers.Dense(output_shape))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_lstm_with_tuner(
    lstm_data: Dict[str, Any],
    max_trials: int = 5,
    epochs: int = 50,
    project_name: str = 'lstm_residuals_tuning'
) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    """
    Train LSTM model with Keras Tuner hyperparameter optimization.
    
    Args:
        lstm_data: Prepared LSTM data from prepare_residuals_for_lstm_training
        max_trials: Maximum tuning trials
        epochs: Training epochs
        project_name: Tuner project name
        
    Returns:
        Tuple of (best_model, tuner, train_predictions, test_predictions)
    """
    print("🚀 Starting LSTM training with Keras Tuner...")
    print(f"📊 Training parameters: max_trials={max_trials}, epochs={epochs}")
    
    # Import required libraries
    try:
        import tensorflow as tf
        from tensorflow import keras
        import keras_tuner as kt
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
    except ImportError as e:
        print(f"❌ Required libraries not installed: {e}")
        print("📋 Please install: pip install tensorflow keras-tuner")
        raise
    
    # Extract data
    X_train = lstm_data['X_train']
    X_test = lstm_data['X_test']
    Y_train = lstm_data['Y_train']
    Y_test = lstm_data['Y_test']
    
    print(f"📊 Training data shapes:")
    print(f"   • X_train: {X_train.shape}")
    print(f"   • Y_train: {Y_train.shape}")
    print(f"   • X_test: {X_test.shape}")
    print(f"   • Y_test: {Y_test.shape}")
    
    # Initialize tuner
    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, X_train.shape[1:], Y_train.shape[1]),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        project_name=project_name,
        overwrite=True
    )
    
    print(f"🔍 Tuner search space:")
    tuner.search_space_summary()
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Start hyperparameter search
    print("🔍 Starting hyperparameter search...")
    tuner.search(
        x=X_train,
        y=Y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get best hyperparameters and model
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    
    print("✅ Hyperparameter search completed!")
    print(f"🏆 Best hyperparameters:")
    print(f"   • LSTM units 1: {best_hyperparameters.get('lstm_units_1')}")
    print(f"   • LSTM units 2: {best_hyperparameters.get('lstm_units_2')}")
    print(f"   • Dense units: {best_hyperparameters.get('dense_units')}")
    print(f"   • Learning rate: {best_hyperparameters.get('learning_rate'):.6f}")
    
    # Make predictions
    print("📊 Making predictions with best model...")
    train_predictions = best_model.predict(X_train, verbose=0)
    test_predictions = best_model.predict(X_test, verbose=0)
    
    print(f"📊 Prediction shapes:")
    print(f"   • Train predictions: {train_predictions.shape}")
    print(f"   • Test predictions: {test_predictions.shape}")
    
    # Display model summary
    print("\n🔧 Best Model Architecture:")
    best_model.summary()
    
    return best_model, tuner, train_predictions, test_predictions


def complete_lstm_workflow():
    """
    Complete LSTM workflow: load data, train model, make predictions, and plot results.
    """
    print("🚀 Starting Complete LSTM Workflow")
    print("=" * 70)
    
    try:
        # Step 1: Load and prepare data
        print("📋 Step 1: Loading and preparing data...")
        residuals_df = load_arima_residuals_from_csv()
        
        lstm_data = prepare_residuals_for_lstm_training(
            residuals_df, 
            lookback=12, 
            forecast=32,
            train_test_split=0.8
        )
        
        print("✅ Data preparation completed!")
        
        # Step 2: Train LSTM model
        print("\n📋 Step 2: Training LSTM model...")
        best_model, tuner, train_predictions, test_predictions = train_lstm_with_tuner(
            lstm_data,
            max_trials=3,  # Keep low for demo
            epochs=50,
            project_name='lstm_residuals_hybrid'
        )
        
        print("✅ LSTM training completed!")
        
        # Step 3: Create plotting data
        print("\n📋 Step 3: Preparing data for plotting...")
        
        # Create mock train/test series for demonstration (in real use, these come from your volume data)
        all_dates = residuals_df.index
        train_size = int(0.8 * len(all_dates))
        
        # For demonstration, create realistic volume-like data
        np.random.seed(42)
        trend = np.linspace(100, 200, len(all_dates))
        seasonality = 50 * np.sin(2 * np.pi * np.arange(len(all_dates)) / 52)
        noise = np.random.normal(0, 20, len(all_dates))
        volume_data = trend + seasonality + noise
        volume_data = np.maximum(volume_data, 0)  # Ensure non-negative
        
        train_series = pd.Series(volume_data[:train_size], index=all_dates[:train_size])
        test_series = pd.Series(volume_data[train_size:train_size+32], index=all_dates[train_size:train_size+32])
        
        # Create mock first model forecast (representing ARIMA predictions)
        mock_arima_forecast = test_series.values + np.random.normal(0, 10, len(test_series))
        
        print("✅ Plotting data prepared!")
        
        # Step 4: Comprehensive evaluation and plotting
        print("\n📋 Step 4: Creating comprehensive evaluation...")
        
        # Import plotting functions
        from hybrid_plotting import comprehensive_model_evaluation
        
        results = comprehensive_model_evaluation(
            series_train=train_series,
            series_test=test_series,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            Y_test=lstm_data['Y_test'],
            scaler=lstm_data['scaler'],
            first_model_forecast=mock_arima_forecast,
            model_signature="ARIMA+LSTM Hybrid",
            save_plots=True
        )
        
        print("✅ Evaluation completed!")
        
        # Step 5: Display results and show plots
        print("\n" + "=" * 70)
        print("🎉 COMPLETE WORKFLOW RESULTS:")
        print("=" * 70)
        print(f"📊 Data Summary:")
        print(f"   • Residuals data points: {len(residuals_df)}")
        print(f"   • LSTM training sequences: {lstm_data['X_train'].shape[0]}")
        print(f"   • LSTM test sequences: {lstm_data['X_test'].shape[0]}")
        
        print(f"\n📊 Model Performance:")
        print(f"   • Hybrid MAE: {results['hybrid_metrics']['MAE']:.2f}")
        print(f"   • Hybrid RMSE: {results['hybrid_metrics']['RMSE']:.2f}")
        print(f"   • Hybrid MAPE: {results['hybrid_metrics']['MAPE']:.2f}%")
        
        if results['first_model_metrics']:
            print(f"   • First Model MAE: {results['first_model_metrics']['MAE']:.2f}")
            print(f"   • First Model MAPE: {results['first_model_metrics']['MAPE']:.2f}%")
        
        print(f"\n📁 Outputs saved to 'plots/' directory")
        print(f"   • Interactive HTML plots")
        print(f"   • PNG images (if kaleido installed)")
        print(f"   • CSV comparison data")
        
        # Show interactive plots
        print(f"\n📈 Displaying interactive plots...")
        results['figures']['main_plot'].show()
        results['figures']['residuals_plot'].show()
        
        return results, best_model, lstm_data
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# Legacy function for backward compatibility
def prepare_residuals_for_lstm(residuals_df: pd.DataFrame, sequence_length: int = 10) -> Dict[str, np.ndarray]:
    """
    Legacy function - use prepare_residuals_for_lstm_training instead.
    """
    print("⚠️  Using legacy function. Consider using prepare_residuals_for_lstm_training for full workflow.")
    return prepare_residuals_for_lstm_training(residuals_df, lookback=sequence_length, forecast=1)


if __name__ == "__main__":
    # Run the complete workflow
    results, best_model, lstm_data = complete_lstm_workflow()