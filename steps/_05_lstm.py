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
    Prepare residuals data for hybrid SARIMA+LSTM training with correct data alignment.

    Fixed approach:
    1. Use only training residuals (not duplicated forecasts)
    2. Proper train/test split aligned with forecast period
    3. Single scaler fitted on training residuals only

    Args:
        residuals_df: DataFrame with residuals from ARIMA model
        sarima_forecasts: SARIMA forecast values for test period (should be 32 values)
        lookback: Number of past observations to use as input (default: 12)
        forecast: Number of future observations to predict (default: 32)

    Returns:
        Dictionary containing prepared training data with correct alignment
    """
    print("🔧 Preparing hybrid SARIMA+LSTM data (FIXED - no duplication)")
    print(f"📊 Parameters: lookback={lookback}, forecast={forecast}")
    print(f"📊 Residuals shape: {residuals_df.shape}")
    print(f"📊 SARIMA forecasts shape: {sarima_forecasts.shape}")

    # Step 1: Extract residuals values and ensure correct length
    residuals_values = residuals_df['residuals'].values
    print(f"📊 Residuals length: {len(residuals_values)}")
    print(f"📊 SARIMA forecasts length: {len(sarima_forecasts)}")

    # Ensure we have exactly 32 forecasts for the test period
    if len(sarima_forecasts) != forecast:
        print(f"⚠️  Warning: Expected {forecast} forecasts, got {len(sarima_forecasts)}")
        # Truncate or pad as needed
        if len(sarima_forecasts) > forecast:
            sarima_forecasts = sarima_forecasts[:forecast]
        else:
            # Pad with last value if needed
            padding_length = forecast - len(sarima_forecasts)
            padding = np.full(padding_length, sarima_forecasts[-1])
            sarima_forecasts = np.concatenate([sarima_forecasts, padding])
        print(f"📊 Adjusted SARIMA forecasts length: {len(sarima_forecasts)}")

    # Step 2: Split residuals into train/test (matching the ARIMA model split)
    # The last 'forecast' periods should align with the test period
    train_residuals = residuals_values[:-forecast]  # All but last 32
    test_residuals = residuals_values[-forecast:]   # Last 32 (for comparison)

    print(f"📊 Train residuals length: {len(train_residuals)}")
    print(f"📊 Test residuals length: {len(test_residuals)}")

    # Step 3: Scale using single scaler fitted ONLY on training residuals
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler on training residuals only
    train_residuals_scaled = scaler.fit_transform(train_residuals.reshape(-1, 1)).flatten()

    # Transform test residuals and forecasts using the same scaler
    test_residuals_scaled = scaler.transform(test_residuals.reshape(-1, 1)).flatten()
    sarima_forecasts_scaled = scaler.transform(sarima_forecasts.reshape(-1, 1)).flatten()

    print(f"📊 Train residuals scaled stats - Mean: {train_residuals_scaled.mean():.4f}, Std: {train_residuals_scaled.std():.4f}")
    print(f"📊 Test residuals scaled stats - Mean: {test_residuals_scaled.mean():.4f}, Std: {test_residuals_scaled.std():.4f}")
    print(f"📊 SARIMA forecasts scaled stats - Mean: {sarima_forecasts_scaled.mean():.4f}, Std: {sarima_forecasts_scaled.std():.4f}")

    # Step 4: Create training sequences using ONLY training residuals
    print(f"📊 Creating sequences from training residuals...")
    train_sequences = create_input_sequences(lookback, forecast, train_residuals_scaled)

    X_train = np.array(train_sequences["input_sequences"])
    Y_train = np.array(train_sequences["output_sequences"])

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    print(f"📊 Training sequences - X shape: {X_train.shape}, Y shape: {Y_train.shape}")

    # Step 5: Create test sequences for validation (use overlap with training for X, forecasts for Y)
    if len(train_residuals_scaled) >= lookback:
        # Use last 'lookback' values from training as test input
        X_test_input = train_residuals_scaled[-lookback:].reshape(1, lookback, 1)
        # Use scaled SARIMA forecasts as test target
        Y_test = sarima_forecasts_scaled.reshape(1, -1)

        print(f"📊 Test sequences - X shape: {X_test_input.shape}, Y shape: {Y_test.shape}")
    else:
        print("⚠️  Warning: Not enough training data for test sequence creation")
        X_test_input = np.array([]).reshape(0, lookback, 1)
        Y_test = np.array([]).reshape(0, forecast)

    # Step 6: Create result dataframe for tracking (without duplication)
    result_sarima_df = pd.DataFrame({
        'period': range(1, len(residuals_values) + 1),
        'residuals': residuals_values,
        'data_type': ['train'] * len(train_residuals) + ['test'] * len(test_residuals),
        'scaled_residuals': np.concatenate([train_residuals_scaled, test_residuals_scaled])
    })

    # Add forecast information for test periods only
    result_sarima_df.loc[result_sarima_df['data_type'] == 'test', 'sarima_forecast'] = sarima_forecasts
    result_sarima_df.loc[result_sarima_df['data_type'] == 'test', 'scaled_sarima_forecast'] = sarima_forecasts_scaled

    print(f"📊 Result dataframe shape: {result_sarima_df.shape}")
    print(f"📊 Training periods: {(result_sarima_df['data_type'] == 'train').sum()}")
    print(f"📊 Test periods: {(result_sarima_df['data_type'] == 'test').sum()}")

    return {
        'X_train': X_train,
        'X_test': X_test_input,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'scaler': scaler,
        'lookback': lookback,
        'forecast': forecast,
        'residuals_df': residuals_df,
        'train_length': len(X_train),
        'train_residuals': train_residuals,
        'test_residuals': test_residuals,
        'sarima_forecasts': sarima_forecasts,
        'result_sarima_df': result_sarima_df,
        'train_residuals_scaled': train_residuals_scaled,
        'test_residuals_scaled': test_residuals_scaled,
        'sarima_forecasts_scaled': sarima_forecasts_scaled
    }




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
        lstm_data: Prepared LSTM data from prepare_residuals_for_lstm_hybrid_training
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
    if 'train_residuals_scaled' in lstm_data:
        residuals_data = lstm_data['train_residuals_scaled']
    else:
        # Fallback to combined_data if available
        residuals_data = lstm_data.get('combined_data', lstm_data['train_residuals_scaled'])

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

        # Load SARIMA residuals from training period
        print("📋 Loading SARIMA residuals: 0b321a90-cf93-4bd3-bb69-3a1fd475bcb4")
        artifact1 = Client().get_artifact_version("0b321a90-cf93-4bd3-bb69-3a1fd475bcb4")
        residuals_data = artifact1.load()

        # Load SARIMA forecasts from test period  
        print("📋 Loading SARIMA forecasts: 277a3bc0-e7cd-4cf6-82d7-330a76c59e17")
        artifact2 = Client().get_artifact_version("277a3bc0-e7cd-4cf6-82d7-330a76c59e17")
        forecasts_data = artifact2.load()

        # Clean column names
        residuals_data.columns = residuals_data.columns.str.strip()
        forecasts_data.columns = forecasts_data.columns.str.strip()
        
        # Inspect the data
        print(f"📊 Residuals data type: {type(residuals_data)}")
        if hasattr(residuals_data, 'shape'):
            print(f"📊 Residuals data shape: {residuals_data.shape}")
        if hasattr(residuals_data, 'columns'):
            print(f"📊 Residuals data columns: {list(residuals_data.columns)}")

        print(f"\n📊 Forecasts data type: {type(forecasts_data)}")
        if hasattr(forecasts_data, 'shape'):
            print(f"📊 Forecasts data shape: {forecasts_data.shape}")
        if hasattr(forecasts_data, 'columns'):
            print(f"📊 Forecasts data columns: {list(forecasts_data.columns)}")

        return residuals_data, forecasts_data

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
        residuals_data, forecasts_data = load_sarima_artifacts_from_zenml()

        if residuals_data is not None and forecasts_data is not None:
            print("✅ Successfully loaded ZenML artifacts!")

            # Set up the data correctly
            residuals_df = residuals_data
            test_predictions_df = forecasts_data

            # Extract SARIMA forecasts from test_predictions DataFrame using 'predicted_volume' column
            if 'predicted_volume' in test_predictions_df.columns:
                sarima_forecasts = test_predictions_df['predicted_volume'].values
                print(f"📋 Extracted {len(sarima_forecasts)} SARIMA forecasts from 'predicted_volume' column")
                print(f"📊 SARIMA forecasts range: {sarima_forecasts.min():.2f} to {sarima_forecasts.max():.2f}")
            else:
                print(f"⚠️  'predicted_volume' column not found. Available columns: {list(test_predictions_df.columns)}")
                sarima_forecasts = None

        else:
            print("📋 ZenML artifacts not available, trying CSV fallback...")
            try:
                residuals_df = load_arima_residuals_from_csv()
                print("✅ Successfully loaded residuals from CSV!")
                
                # For CSV fallback, we'll need to create mock SARIMA forecasts
                # Use a simple approach: last 32 residuals as proxy forecasts
                if len(residuals_df) >= 32:
                    sarima_forecasts = residuals_df['residuals'].values[-32:] * 0.5  # Scaled down proxy
                    print(f"📋 Created proxy SARIMA forecasts from residuals: {len(sarima_forecasts)} values")
                else:
                    sarima_forecasts = np.random.normal(0, 1, 32)  # Random fallback
                    print("⚠️  Using random proxy forecasts (insufficient residuals data)")
            except Exception as e:
                print(f"❌ CSV fallback also failed: {e}")
                raise ValueError("❌ Could not load data from ZenML or CSV. Please ensure data is available.")

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
            raise ValueError("❌ SARIMA forecasts are required for hybrid approach. Cannot proceed without forecasts.")

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

        # Step 3: Create plotting data (FIXED - avoid duplication)
        print("\n📋 Step 3: Preparing data for plotting...")

        # Use the corrected data from LSTM preparation
        # The test_residuals and sarima_forecasts are now properly aligned
        test_residuals = lstm_data['test_residuals']
        test_forecasts = lstm_data['sarima_forecasts']

        print(f"📊 Test residuals length: {len(test_residuals)}")
        print(f"📊 Test forecasts length: {len(test_forecasts)}")

        # Use actual dates and data from forecasts_data artifact for test period
        if 'date' in test_predictions_df.columns:
            test_dates = pd.to_datetime(test_predictions_df['date'])
            # Use actual volume data directly from the artifact
            if 'actual_volume' in test_predictions_df.columns:
                test_series = pd.Series(test_predictions_df['actual_volume'].values, index=test_dates, name='Volume')
            else:
                # Fallback: reconstruct from residuals + forecasts
                test_actual_reconstructed = test_residuals + test_forecasts
                test_series = pd.Series(test_actual_reconstructed, index=test_dates, name='Volume')
        else:
            # Fallback: use residuals index if available
            if isinstance(residuals_df.index, pd.DatetimeIndex):
                test_dates = residuals_df.index[-len(test_residuals):]
            else:
                # Last resort: create dates (but this shouldn't happen with correct artifacts)
                test_dates = pd.date_range(start='2023-12-16', periods=len(test_residuals), freq='W-SAT')
            
            test_actual_reconstructed = test_residuals + test_forecasts
            test_series = pd.Series(test_actual_reconstructed, index=test_dates, name='Volume')

        print(f"✅ Created test series with shape: {test_series.shape}")
        print(f"📊 Test series range: {test_series.min():.0f} to {test_series.max():.0f}")
        print(f"📊 Test series mean: {test_series.mean():.0f}")

        # Create training series from the training portion of residuals
        train_residuals = lstm_data['train_residuals']
        
        # Use actual dates from residuals data if available
        if 'date' in residuals_df.columns:
            # Get the training portion of dates (all but last 32)
            train_dates = pd.to_datetime(residuals_df['date'].values[:-len(test_residuals)])
        elif isinstance(residuals_df.index, pd.DatetimeIndex):
            train_dates = residuals_df.index[:len(train_residuals)]
        else:
            # Create proper weekly dates ending before test period
            train_end = test_dates[0] - pd.Timedelta(weeks=1)
            train_dates = pd.date_range(end=train_end, periods=len(train_residuals), freq='W-SAT')

        # For training data, approximate actual values (this is for visualization only)
        # Use a reasonable baseline + residuals
        baseline_train = np.full_like(train_residuals, test_series.mean())
        train_actual_reconstructed = baseline_train + train_residuals
        train_series = pd.Series(train_actual_reconstructed, index=train_dates, name='Volume')

        print(f"✅ Created training series with shape: {train_series.shape}")
        print(f"📊 Training series range: {train_series.min():.0f} to {train_series.max():.0f}")

        # Set arima_forecast for plotting
        arima_forecast = test_forecasts

        print(f"📅 Training date range: {train_series.index.min()} to {train_series.index.max()}")
        print(f"📅 Test date range: {test_series.index.min()} to {test_series.index.max()}")
        print(f"📊 Training series length: {len(train_series)}")
        print(f"📊 Test series length: {len(test_series)}")

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
            first_model_forecast=arima_forecast,
            model_signature="ARIMA+LSTM Hybrid",
            save_plots=True
        )

        print("✅ Evaluation completed!")

        print("✅ Evaluation completed - using single approach to avoid duplication!")

        # Step 4.5: Create comparison table
        print("\n📋 Step 4.5: Creating comparison table...")

        # Load original data artifact from ZenML (modelling_data)
        try:
            from zenml.client import Client
            artifact = Client().get_artifact_version("43d0efe9-ad58-4c55-8666-f6d1d28b43df")
            original_data_artifact = artifact.load()
            print("✅ Successfully loaded ZenML modelling_data artifact")
        except Exception as e:
            print(f"⚠️  Warning: Could not load ZenML artifact: {e}")
            original_data_artifact = None

        # Get the components for the comparison table
        original_data = test_series.values  # Original test data
        hybrid_forecast = results['hybrid_forecast']  # Hybrid predictions from evaluation
        sarima_forecast = arima_forecast if arima_forecast is not None else np.zeros_like(original_data)
        lstm_residuals = results['lstm_residuals_prediction']  # LSTM residuals/predictions

        # Prepare artifact data if available
        if original_data_artifact is not None:
            print(f"📊 Original data artifact type: {type(original_data_artifact)}")
            print(f"📊 Original data artifact shape: {original_data_artifact.shape if hasattr(original_data_artifact, 'shape') else 'No shape'}")

            # Handle DataFrame case - extract 'volume' column specifically
            if hasattr(original_data_artifact, 'columns'):
                print(f"📊 Available columns: {list(original_data_artifact.columns)}")
                if 'volume' in original_data_artifact.columns:
                    volume_data = original_data_artifact['volume']
                    print("✅ Found 'volume' column in artifact")
                elif 'Volume' in original_data_artifact.columns:
                    volume_data = original_data_artifact['Volume']
                    print("✅ Found 'Volume' column in artifact")
                else:
                    # Try to find a numeric column
                    numeric_cols = original_data_artifact.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        volume_data = original_data_artifact[numeric_cols[0]]
                        print(f"✅ Using first numeric column: {numeric_cols[0]}")
                    else:
                        print("⚠️  No numeric columns found, using zeros")
                        artifact_data_values = np.zeros(len(original_data))
                        volume_data = None

                # Extract data from volume column if found
                if volume_data is not None:
                    # Extract relevant portion that matches test data timeframe
                    if hasattr(original_data_artifact, 'loc') and len(test_series.index) > 0:
                        try:
                            # Try to align artifact data with test series index
                            artifact_data_aligned = volume_data.loc[test_series.index]
                            artifact_data_values = artifact_data_aligned.values
                        except:
                            # Fallback: use first n values from volume column
                            artifact_data_values = volume_data.values[:len(original_data)]
                    else:
                        # Simple fallback: use first n values from volume column
                        artifact_data_values = volume_data.values[:len(original_data)]
            else:
                # Handle non-DataFrame case
                if hasattr(original_data_artifact, 'values'):
                    artifact_data_values = original_data_artifact.values[:len(original_data)]
                else:
                    artifact_data_values = original_data_artifact[:len(original_data)]
        else:
            artifact_data_values = np.zeros_like(original_data)

        # Debug: Check shapes of all arrays
        print("\n🔍 Debugging array shapes:")
        print(f"   • original_data shape: {np.array(original_data).shape}")
        print(f"   • hybrid_forecast shape: {np.array(hybrid_forecast).shape}")
        print(f"   • sarima_forecast shape: {np.array(sarima_forecast).shape}")
        print(f"   • lstm_residuals shape: {np.array(lstm_residuals).shape}")
        print(f"   • artifact_data_values shape: {np.array(artifact_data_values).shape}")

        # Flatten any multi-dimensional arrays to 1D
        original_data = np.array(original_data).flatten()
        hybrid_forecast = np.array(hybrid_forecast).flatten()
        sarima_forecast = np.array(sarima_forecast).flatten()
        lstm_residuals = np.array(lstm_residuals).flatten()
        artifact_data_values = np.array(artifact_data_values).flatten()

        print("\n🔧 After flattening:")
        print(f"   • original_data shape: {original_data.shape}")
        print(f"   • hybrid_forecast shape: {hybrid_forecast.shape}")
        print(f"   • sarima_forecast shape: {sarima_forecast.shape}")
        print(f"   • lstm_residuals shape: {lstm_residuals.shape}")
        print(f"   • artifact_data_values shape: {artifact_data_values.shape}")

        # Ensure all arrays have the same length
        min_length = min(len(original_data), len(hybrid_forecast), len(sarima_forecast), len(lstm_residuals), len(artifact_data_values))
        print(f"\n📏 Using minimum length: {min_length}")

        # Add test_predictions data for clarity
        test_predictions_actual = test_predictions_df['actual'].values[:min_length] if 'actual' in test_predictions_df.columns else np.zeros(min_length)
        test_predictions_predicted = test_predictions_df['predicted'].values[:min_length] if 'predicted' in test_predictions_df.columns else np.zeros(min_length)
        test_predictions_residuals = test_predictions_df['residuals'].values[:min_length] if 'residuals' in test_predictions_df.columns else np.zeros(min_length)

        # Create comparison DataFrame with test_predictions data included
        comparison_df = pd.DataFrame({
            'Original Data': original_data[:min_length],
            'original_data_artifact_modelling_data': artifact_data_values[:min_length],
            'Hybrid SARIMA + LSTM': hybrid_forecast[:min_length],
            'SARIMA': sarima_forecast[:min_length],
            'LSTM': lstm_residuals[:min_length],
            'test_predictions_actual': test_predictions_actual,
            'test_predictions_predicted': test_predictions_predicted,
            'test_predictions_residuals': test_predictions_residuals
        }, index=test_series.index[:min_length])

        print("\n" + "=" * 80)
        print("📊 PREDICTION COMPARISON TABLE (First 8 rows):")
        print("=" * 80)
        print(comparison_df.head(5).round(1))
        print("=" * 80)

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




if __name__ == "__main__":
    # Run the complete workflow
    results, best_model, lstm_data = complete_lstm_workflow()
