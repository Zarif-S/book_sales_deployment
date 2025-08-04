import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
import os
import keras_tuner as kt


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


def prepare_residuals_for_lstm_hybrid_training(
    residuals_df: pd.DataFrame,
    sarima_forecasts: np.ndarray,
    lookback: int = 12,
    forecast: int = 32
) -> Dict[str, np.ndarray]:
    """
    Prepare residuals data for hybrid SARIMA+LSTM training matching notebook approach.
    
    Key difference: Creates combined data where:
    - Training period: Uses historical residuals 
    - Test period: Uses SARIMA forecasts (scaled) as LSTM input
    
    Args:
        residuals_df: DataFrame with residuals from ARIMA model
        sarima_forecasts: SARIMA forecast values for test period
        lookback: Number of past observations to use as input (default: 12)
        forecast: Number of future observations to predict (default: 32)
        
    Returns:
        Dictionary containing prepared training data matching notebook approach
    """
    print("🔧 Preparing hybrid SARIMA+LSTM data (notebook-style)")
    print(f"📊 Parameters: lookback={lookback}, forecast={forecast}")
    print(f"📊 Residuals shape: {residuals_df.shape}")
    print(f"📊 SARIMA forecasts shape: {sarima_forecasts.shape}")
    
    # Initialize single scaler (matching notebook approach)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on residuals (training data)
    residuals_values = residuals_df['residuals'].values.reshape(-1, 1)
    residuals_scaled = scaler.fit_transform(residuals_values).flatten()
    
    # Transform SARIMA forecasts using the SAME scaler (critical!)
    sarima_forecasts_scaled = scaler.transform(sarima_forecasts.reshape(-1, 1)).flatten()
    
    print(f"📊 Residuals stats - Mean: {residuals_df['residuals'].mean():.4f}, Std: {residuals_df['residuals'].std():.4f}")
    print(f"📊 SARIMA forecasts stats - Mean: {sarima_forecasts.mean():.4f}, Std: {sarima_forecasts.std():.4f}")
    
    # Create combined data following notebook approach
    # Cutoff point: total_length - forecast_horizon
    cutoff = len(residuals_scaled) - forecast
    
    print(f"📊 Cutoff point: {cutoff} (total: {len(residuals_scaled)}, forecast: {forecast})")
    
    # Combined data: residuals[:cutoff] + sarima_forecasts[cutoff:]
    combined_data = np.concatenate([
        residuals_scaled[:cutoff],      # Historical residuals for training
        sarima_forecasts_scaled         # SARIMA forecasts for test input
    ])
    
    print(f"📊 Combined data shape: {combined_data.shape}")
    print(f"📊 Combined data stats - Mean: {combined_data.mean():.4f}, Std: {combined_data.std():.4f}")
    
    # Create sequences using combined data
    sequences = create_input_sequences(lookback, forecast, combined_data)
    
    # Convert to numpy arrays
    X_combined = np.array(sequences["input_sequences"])
    Y_combined = np.array(sequences["output_sequences"])
    
    # Reshape for LSTM [samples, time steps, features]
    X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)
    
    print(f"📊 Combined sequences - X shape: {X_combined.shape}, Y shape: {Y_combined.shape}")
    
    # Split following notebook approach: train on residuals, test uses SARIMA forecasts
    train_length = cutoff - lookback  # Available training sequences
    
    X_train = X_combined[:train_length]
    X_test = X_combined[train_length:]
    Y_train = Y_combined[:train_length] 
    Y_test = Y_combined[train_length:]
    
    print(f"📊 Hybrid split:")
    print(f"   • X_train shape: {X_train.shape} (trained on residuals)")
    print(f"   • X_test shape: {X_test.shape} (uses SARIMA forecasts as input)")
    print(f"   • Y_train shape: {Y_train.shape}")
    print(f"   • Y_test shape: {Y_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'scaler': scaler,
        'lookback': lookback,
        'forecast': forecast,
        'residuals_df': residuals_df,
        'train_length': train_length,
        'cutoff': cutoff,
        'combined_data': combined_data,
        'residuals_scaled': residuals_scaled,
        'sarima_forecasts_scaled': sarima_forecasts_scaled
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
        lookback=21,
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


def tuned_model(hp):
    """
    Build LSTM model with hyperparameter tuning including tunable lookback.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

    model = Sequential()

    # Tune the lookback parameter
    lookback = hp.Int('lookback', min_value=6, max_value=52, step=5)

    model.add(Input(shape=(lookback, 1)))
    model.add(LSTM(hp.Int('input_unit', min_value=4, max_value=128, step=8), return_sequences=True))

    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=4, max_value=128, step=8), return_sequences=True))

    model.add(LSTM(hp.Int('layer_2_neurons', min_value=4, max_value=128, step=8)))
    model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))

    # Output layer - we'll set this dynamically
    forecast = 32  # Set default, will be overridden
    model.add(Dense(forecast))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model


class TunableLookbackHyperModel(kt.HyperModel):
    """HyperModel that handles tunable lookback by creating sequences dynamically."""
    
    def __init__(self, residuals_data, forecast_horizon):
        super().__init__()
        self.residuals_data = residuals_data
        self.forecast_horizon = forecast_horizon
    
    def build(self, hp):
        """Build model with current hyperparameters."""
        # Import here to avoid issues
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow import keras
        
        model = Sequential()

        # Tune the lookback parameter
        lookback = hp.Int('lookback', min_value=6, max_value=52, step=5)

        model.add(Input(shape=(lookback, 1)))
        model.add(LSTM(hp.Int('input_unit', min_value=4, max_value=128, step=8), return_sequences=True))

        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=4, max_value=128, step=8), return_sequences=True))

        model.add(LSTM(hp.Int('layer_2_neurons', min_value=4, max_value=128, step=8)))
        model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))

        # Output layer
        model.add(Dense(self.forecast_horizon))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        """Custom fit method that creates sequences based on tuned lookback."""
        # Get the lookback from hyperparameters
        lookback = hp.get('lookback')
        
        # Create sequences with the current lookback
        sequences = create_input_sequences(lookback, self.forecast_horizon, self.residuals_data)
        X_combined = np.array(sequences["input_sequences"])
        Y_combined = np.array(sequences["output_sequences"])
        X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)
        
        # Split data for training
        train_length = int(0.8 * len(X_combined))
        X_train = X_combined[:train_length]
        Y_train = Y_combined[:train_length]
        
        # Train the model with the generated sequences
        return model.fit(
            X_train, 
            Y_train, 
            validation_split=0.2,
            *args, 
            **kwargs
        )


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

        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    except ImportError as e:
        print(f"❌ Required libraries not installed: {e}")
        print("📋 Please install: pip install tensorflow keras-tuner")
        raise

    # Extract residuals data for tunable lookback approach
    if 'combined_data' in lstm_data:
        residuals_data = lstm_data['combined_data']
    else:
        residuals_data = lstm_data['scaled_residuals']
    
    forecast_horizon = lstm_data['forecast']

    print(f"📊 Using residuals data shape: {residuals_data.shape}")
    print(f"📊 Forecast horizon: {forecast_horizon}")

    # Create the tunable hypermodel
    hypermodel = TunableLookbackHyperModel(residuals_data, forecast_horizon)
    
    tuner = kt.RandomSearch(
        hypermodel,
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
    # Note: We don't pass x and y here since our custom fit method handles data creation
    tuner.search(
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Get best hyperparameters and model
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    print("✅ Hyperparameter search completed!")
    print(f"🏆 Best hyperparameters:")
    print(f"   • Lookback: {best_hyperparameters.get('lookback')}")
    print(f"   • Input units: {best_hyperparameters.get('input_unit')}")
    print(f"   • Number of layers: {best_hyperparameters.get('n_layers')}")
    print(f"   • Final LSTM units: {best_hyperparameters.get('layer_2_neurons')}")
    print(f"   • Dropout rate: {best_hyperparameters.get('Dropout_rate')}")
    
    # Show additional layer details if they exist
    n_layers = best_hyperparameters.get('n_layers')
    for i in range(n_layers):
        layer_param_name = f'lstm_{i}_units'
        try:
            layer_units = best_hyperparameters.get(layer_param_name)
            print(f"   • LSTM layer {i} units: {layer_units}")
        except:
            # Parameter doesn't exist, skip
            pass

    # Make predictions using best hyperparameters
    print("📊 Making predictions with best model...")
    best_lookback = best_hyperparameters.get('lookback')
    
    # Recreate sequences with best lookback
    sequences = create_input_sequences(best_lookback, forecast_horizon, residuals_data)
    X_combined = np.array(sequences["input_sequences"])
    Y_combined = np.array(sequences["output_sequences"])
    X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)
    
    # Split data using the same ratio
    train_length = int(0.8 * len(X_combined))
    X_train_final = X_combined[:train_length]
    X_test_final = X_combined[train_length:]
    
    train_predictions = best_model.predict(X_train_final, verbose=0)
    test_predictions = best_model.predict(X_test_final, verbose=0)

    print(f"📊 Best lookback: {best_lookback}")
    print(f"📊 Prediction shapes:")
    print(f"   • Train predictions: {train_predictions.shape}")
    print(f"   • Test predictions: {test_predictions.shape}")

    # Display model summary
    print("\n🔧 Best Model Architecture:")
    best_model.summary()

    return best_model, tuner, train_predictions, test_predictions


def load_sarima_artifacts_from_zenml():
    """
    Load SARIMA residuals and test predictions from ZenML artifacts.
    """
    print("🔧 Loading SARIMA artifacts from ZenML...")
    
    try:
        from zenml.client import Client
        
        # Load first artifact (residuals)
        print("📋 Loading artifact 1: 89a6f830-c433-45ab-b327-277e8c8b5fef")
        artifact1 = Client().get_artifact_version("89a6f830-c433-45ab-b327-277e8c8b5fef")
        data1 = artifact1.load()
        
        # Load second artifact (test predictions)
        print("📋 Loading artifact 2: 34d97525-cb24-42cc-800f-e3d931a1fccf")  
        artifact2 = Client().get_artifact_version("34d97525-cb24-42cc-800f-e3d931a1fccf")
        data2 = artifact2.load()
        
        # Inspect the data
        print(f"📊 Artifact 1 type: {type(data1)}")
        if hasattr(data1, 'shape'):
            print(f"📊 Artifact 1 shape: {data1.shape}")
        if hasattr(data1, 'columns'):
            print(f"📊 Artifact 1 columns: {list(data1.columns)}")
        if hasattr(data1, 'head'):
            print("📊 Artifact 1 head:")
            print(data1.head())
            
        print(f"\n📊 Artifact 2 type: {type(data2)}")
        if hasattr(data2, 'shape'):
            print(f"📊 Artifact 2 shape: {data2.shape}")
        if hasattr(data2, 'columns'):
            print(f"📊 Artifact 2 columns: {list(data2.columns)}")
        if hasattr(data2, 'head'):
            print("📊 Artifact 2 head:")
            print(data2.head())
            
        return data1, data2
        
    except Exception as e:
        print(f"❌ Error loading ZenML artifacts: {e}")
        print("📋 Falling back to CSV loading...")
        return None, None


def complete_lstm_workflow(sarima_forecasts: np.ndarray = None):
    """
    Complete LSTM workflow: load data, train model, make predictions, and plot results.
    
    Args:
        sarima_forecasts: SARIMA forecast values for hybrid approach (optional)
    """
    print("🚀 Starting Complete LSTM Workflow")
    print("=" * 70)

    try:
        # Step 1: Load and prepare data
        print("📋 Step 1: Loading and preparing data...")
        
        # First, try to load from ZenML artifacts
        data1, data2 = load_sarima_artifacts_from_zenml()
        
        if data1 is not None and data2 is not None:
            print("✅ Successfully loaded ZenML artifacts!")
            
            # Determine which artifact contains residuals and which contains forecasts
            # We'll inspect and determine the appropriate data to use
            residuals_df = None
            sarima_forecasts = None
            
            # Identify residuals DataFrame (has historical training residuals)
            if hasattr(data1, 'columns') and 'residuals' in data1.columns and 'predicted' not in data1.columns:
                residuals_df = data1
                test_predictions_df = data2
                print("📋 Artifact 1: Historical residuals DataFrame")
                print("📋 Artifact 2: Test predictions DataFrame")
            elif hasattr(data2, 'columns') and 'residuals' in data2.columns and 'predicted' not in data2.columns:
                residuals_df = data2
                test_predictions_df = data1
                print("📋 Artifact 2: Historical residuals DataFrame") 
                print("📋 Artifact 1: Test predictions DataFrame")
            else:
                print("⚠️  Could not identify artifacts properly, using CSV fallback")
                residuals_df = load_arima_residuals_from_csv()
                test_predictions_df = None
            
            # Extract SARIMA forecasts from test_predictions DataFrame
            if test_predictions_df is not None and hasattr(test_predictions_df, 'columns') and 'predicted' in test_predictions_df.columns:
                sarima_forecasts = test_predictions_df['predicted'].values
                print(f"📋 Extracted {len(sarima_forecasts)} SARIMA forecasts from 'predicted' column")
                print(f"📊 SARIMA forecasts range: {sarima_forecasts.min():.2f} to {sarima_forecasts.max():.2f}")
            else:
                sarima_forecasts = None
                print("⚠️  Could not extract SARIMA forecasts from test predictions")
            
        else:
            print("📋 Using CSV fallback for residuals...")
            residuals_df = load_arima_residuals_from_csv()
            sarima_forecasts = None

        # Use hybrid approach if we have forecasts, otherwise simple approach
        if sarima_forecasts is not None:
            print("📋 Using hybrid SARIMA+LSTM approach (notebook-style)")
            print(f"📊 SARIMA forecasts shape: {sarima_forecasts.shape}")
            lstm_data = prepare_residuals_for_lstm_hybrid_training(
                residuals_df,
                sarima_forecasts,
                lookback=21,
                forecast=32
            )
        else:
            print("📋 Using simple LSTM approach (no SARIMA forecasts available)")
            lstm_data = prepare_residuals_for_lstm_training(
                residuals_df,
                lookback=21,
                forecast=32,
                train_test_split=0.8
            )

        print("✅ Data preparation completed!")

        # Step 2: Train LSTM model
        print("\n📋 Step 2: Training LSTM model...")
        best_model, tuner, train_predictions, test_predictions = train_lstm_with_tuner(
            lstm_data,
            max_trials=10,  # Increased for better hyperparameter search
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
