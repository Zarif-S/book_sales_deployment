import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import optuna
import warnings
import json

warnings.filterwarnings('ignore')

def load_pipeline_data():
    """Load train and test data from pipeline CSV files for consistent comparison."""
    print("📂 Loading pipeline train/test data from CSV files...")

    train_path = "data/processed/combined_train_data.csv"
    test_path = "data/processed/combined_test_data.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Pipeline train data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Pipeline test data not found: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"✅ Loaded train data: {train_data.shape}")
    print(f"✅ Loaded test data: {test_data.shape}")
    print(f"📊 Train columns: {list(train_data.columns)}")
    print(f"📊 Test columns: {list(test_data.columns)}")

    return train_data, test_data


def load_original_data_from_csv(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load original sales data from CSV file for LSTM training.
    """
    print("🔍 Loading original sales data from CSV file...")

    # Look for original data CSV in the data directory
    original_csv_path = os.path.join(data_dir, "combined_test_data.csv")

    if not os.path.exists(original_csv_path):
        print(f"⚠️  Original data CSV not found at: {original_csv_path}")
        print("📋 Checking for alternative locations...")

        # Check other possible locations
        alternative_paths = [
            "data/combined_test_data.csv",
            "combined_test_data.csv",
            os.path.join(data_dir, "..", "combined_test_data.csv")
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                original_csv_path = alt_path
                print(f"✅ Found original data CSV at: {original_csv_path}")
                break
        else:
            raise FileNotFoundError(f"Original data CSV not found. Expected at: {original_csv_path}")

    try:
        original_df = pd.read_csv(original_csv_path)

        # Convert date column to datetime and set as index
        if 'date' in original_df.columns:
            original_df['date'] = pd.to_datetime(original_df['date'])
            original_df = original_df.set_index('date')

        print(f"✅ Loaded original data from CSV: {original_csv_path}")
        print(f"📈 Original data shape: {original_df.shape}")
        print(f"📅 Date range: {original_df.index.min()} to {original_df.index.max()}")

        # Check for volume column
        volume_col = None
        for col in ['Volume', 'volume', 'sales', 'Sales']:
            if col in original_df.columns:
                volume_col = col
                break

        if volume_col:
            print(f"📊 Using volume column: {volume_col}")
            print(f"📊 Volume stats - Mean: {original_df[volume_col].mean():.4f}, Std: {original_df[volume_col].std():.4f}")
        else:
            print(f"⚠️  No volume column found. Available columns: {list(original_df.columns)}")

        return original_df

    except Exception as e:
        print(f"⚠️  Could not load original data CSV: {e}")
        raise ValueError(f"Failed to load original data CSV: {e}")


def create_input_sequences(lookback: int, forecast: int, data: np.ndarray) -> Dict[str, list]:
    """
    Create input-output sequences for LSTM training.

    Args:
        lookback: Number of past observations to use as input
        forecast: Number of future observations to predict
        data: 1D array of scaled sales data

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
    print(f"📊 Output sequence shape: ({len(input_sequences)}, {forecast})")

    return {
        'input_sequences': input_sequences,
        'output_sequences': output_sequences
    }


def prepare_original_data_for_lstm_training(
    original_df: pd.DataFrame,
    lookback: int = 12,
    forecast: int = 32,
    train_split_ratio: float = 0.8
) -> Dict[str, np.ndarray]:
    """
    Prepare original sales data for LSTM training.

    Args:
        original_df: DataFrame with original sales data
        lookback: Number of past observations to use as input (default: 12)
        forecast: Number of future observations to predict (default: 32)
        train_split_ratio: Ratio for train/test split (default: 0.8)

    Returns:
        Dictionary containing prepared training data
    """
    print("🔧 Preparing original sales data for LSTM training")
    print(f"📊 Parameters: lookback={lookback}, forecast={forecast}, train_split={train_split_ratio}")
    print(f"📊 Original data shape: {original_df.shape}")

    # Step 1: Extract volume data
    volume_col = None
    for col in ['Volume', 'volume', 'sales', 'Sales']:
        if col in original_df.columns:
            volume_col = col
            break

    if not volume_col:
        raise ValueError("No volume column found in data")

    volume_values = original_df[volume_col].values
    print(f"📊 Volume data length: {len(volume_values)}")
    print(f"📊 Volume stats - Mean: {volume_values.mean():.2f}, Std: {volume_values.std():.2f}")

    # Step 2: Split data into train/test
    split_point = int(len(volume_values) * train_split_ratio)
    train_volume = volume_values[:split_point]
    test_volume = volume_values[split_point:]

    print(f"📊 Train volume length: {len(train_volume)}")
    print(f"📊 Test volume length: {len(test_volume)}")

    # Step 3: Scale using single scaler fitted on training data only
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler on training data only
    train_volume_scaled = scaler.fit_transform(train_volume.reshape(-1, 1)).flatten()

    # Transform test data using the same scaler
    test_volume_scaled = scaler.transform(test_volume.reshape(-1, 1)).flatten()

    print(f"📊 Train volume scaled stats - Mean: {train_volume_scaled.mean():.4f}, Std: {train_volume_scaled.std():.4f}")
    print(f"📊 Test volume scaled stats - Mean: {test_volume_scaled.mean():.4f}, Std: {test_volume_scaled.std():.4f}")

    # Step 4: Create training sequences using training data
    print(f"📊 Creating sequences from training data...")
    train_sequences = create_input_sequences(lookback, forecast, train_volume_scaled)

    # Check if we have enough sequences
    if len(train_sequences["input_sequences"]) == 0:
        raise ValueError(f"❌ Not enough data to create sequences. Need at least {lookback + forecast} data points, but only have {len(train_volume_scaled)} training points.")

    X_train = np.array(train_sequences["input_sequences"])
    Y_train = np.array(train_sequences["output_sequences"])

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    print(f"📊 Training sequences - X shape: {X_train.shape}, Y shape: {Y_train.shape}")

    # Step 5: Create test sequences for validation
    if len(test_volume_scaled) >= lookback + forecast:
        test_sequences = create_input_sequences(lookback, forecast, test_volume_scaled)
        X_test = np.array(test_sequences["input_sequences"])
        Y_test = np.array(test_sequences["output_sequences"])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    else:
        print("⚠️  Warning: Not enough test data for full sequences")
        # Use last 'lookback' values from training as test input
        if len(train_volume_scaled) >= lookback:
            X_test = train_volume_scaled[-lookback:].reshape(1, lookback, 1)
            # Use first 'forecast' values from test data as target
            Y_test = test_volume_scaled[:forecast].reshape(1, -1)
        else:
            X_test = np.array([]).reshape(0, lookback, 1)
            Y_test = np.array([]).reshape(0, forecast)

    print(f"📊 Test sequences - X shape: {X_test.shape}, Y shape: {Y_test.shape}")

    # Step 6: Create result dataframe for tracking
    result_df = pd.DataFrame({
        'period': range(1, len(volume_values) + 1),
        'volume': volume_values,
        'data_type': ['train'] * len(train_volume) + ['test'] * len(test_volume),
        'scaled_volume': np.concatenate([train_volume_scaled, test_volume_scaled])
    })

    print(f"📊 Result dataframe shape: {result_df.shape}")
    print(f"📊 Training periods: {(result_df['data_type'] == 'train').sum()}")
    print(f"📊 Test periods: {(result_df['data_type'] == 'test').sum()}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'scaler': scaler,
        'lookback': lookback,
        'forecast': forecast,
        'original_df': original_df,
        'train_length': len(X_train),
        'train_volume': train_volume,
        'test_volume': test_volume,
        'train_volume_scaled': train_volume_scaled,
        'test_volume_scaled': test_volume_scaled,
        'result_df': result_df,
        'volume_col': volume_col
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


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics with error handling (same as ARIMA/CNN)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def create_lstm_model(lookback: int, forecast_horizon: int, trial: optuna.Trial = None):
    """Create LSTM model for time series forecasting."""
    # Import here to avoid issues
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow import keras

    if trial is not None:
        # Hyperparameter optimization
        input_units = trial.suggest_int('input_units', 16, 128, step=16)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        final_units = trial.suggest_int('final_units', 16, 128, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    else:
        # Default parameters
        input_units = 64
        n_layers = 2
        final_units = 32
        dropout_rate = 0.3
        learning_rate = 0.001

    model = Sequential()
    model.add(Input(shape=(lookback, 1)))

    # First LSTM layer
    model.add(LSTM(input_units, return_sequences=(n_layers > 1)))

    # Additional LSTM layers
    for i in range(n_layers - 1):
        if i == n_layers - 2:  # Last hidden layer
            model.add(LSTM(final_units, return_sequences=False))
        else:
            layer_units = trial.suggest_int(f'lstm_{i}_units', 16, 128, step=16) if trial else 64
            model.add(LSTM(layer_units, return_sequences=True))

    model.add(Dropout(dropout_rate))
    model.add(Dense(forecast_horizon))

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model


def objective_lstm(trial, volume_data: np.ndarray, forecast_horizon: int) -> float:
    """Optuna objective function for LSTM optimization."""
    try:
        # Tune lookback parameter
        max_lookback = min(52, len(volume_data) // 3)  # Use 1/3 of data max
        min_lookback = min(6, max_lookback // 4)
        lookback = trial.suggest_int('lookback', min_lookback, max_lookback, step=2)

        # Create sequences with tuned lookback
        sequences = create_input_sequences(lookback, forecast_horizon, volume_data)
        if len(sequences["input_sequences"]) < 10:  # Need minimum sequences
            return float("inf")

        X_combined = np.array(sequences["input_sequences"])
        Y_combined = np.array(sequences["output_sequences"])
        X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)

        # Split data for training/validation
        train_size = int(0.7 * len(X_combined))
        val_size = int(0.15 * len(X_combined))

        X_train = X_combined[:train_size]
        Y_train = Y_combined[:train_size]
        X_val = X_combined[train_size:train_size + val_size]
        Y_val = Y_combined[train_size:train_size + val_size]

        if len(X_val) == 0:
            return float("inf")

        # Create model with trial hyperparameters
        model = create_lstm_model(lookback, forecast_horizon, trial)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Train model
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        # Evaluate on validation set
        val_pred = model.predict(X_val, verbose=0)
        from sklearn.metrics import mean_squared_error
        val_loss = mean_squared_error(Y_val.flatten(), val_pred.flatten())

        return val_loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")


def run_optuna_optimization_lstm(volume_data: np.ndarray, forecast_horizon: int,
                                n_trials: int, study_name: str = "lstm_optimization") -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization for LSTM with persistent storage."""
    print(f"Starting LSTM Optuna optimization with {n_trials} trials...")

    # Set up Optuna storage (same as ARIMA and CNN)
    storage_dir = os.path.expanduser("~/zenml_optuna_storage")
    os.makedirs(storage_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(storage_dir, f'{study_name}.db')}"

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize"
        )

        print(f"📊 Study info: {len(study.trials)} existing trials found")
        if len(study.trials) > 0:
            print(f"🔄 Resuming optimization from existing study")
            print(f"💾 Best value so far: {study.best_value:.6f}")

        study.optimize(
            lambda trial: objective_lstm(trial, volume_data, forecast_horizon),
            n_trials=n_trials,
            timeout=1800,  # 30 minutes timeout
            n_jobs=1
        )

        return {
            "best_params": dict(study.best_params),
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "study_name": study_name,
            "storage_url": storage_url
        }

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return default parameters
        return {
            "best_params": {
                "lookback": 21,
                "input_units": 64,
                "n_layers": 2,
                "final_units": 32,
                "dropout_rate": 0.3,
                "learning_rate": 0.001
            },
            "best_value": float("inf"),
            "n_trials": 0,
            "study_name": study_name,
            "storage_url": storage_url,
            "error": str(e)
        }


def train_lstm_with_optuna(
    lstm_data: Dict[str, Any],
    n_trials: int = 50,
    study_name: str = 'lstm_optimization'
) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
    """
    Train LSTM model with Optuna hyperparameter optimization.

    Args:
        lstm_data: Prepared LSTM data from prepare_original_data_for_lstm_training
        n_trials: Maximum optimization trials
        study_name: Optuna study name

    Returns:
        Tuple of (best_model, optimization_results, train_predictions, test_predictions)
    """
    print("🚀 Starting LSTM training with Optuna optimization...")
    print(f"📊 Training parameters: n_trials={n_trials}")

    # Import required libraries
    try:
        import tensorflow as tf
        from tensorflow import keras

        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    except ImportError as e:
        print(f"❌ Required libraries not installed: {e}")
        print("📋 Please install: pip install tensorflow")
        raise

    # Extract volume data for optimization
    volume_data = lstm_data['train_volume_scaled']
    forecast_horizon = lstm_data['forecast']

    print(f"📊 Using volume data shape: {volume_data.shape}")
    print(f"📊 Forecast horizon: {forecast_horizon}")

    # Run Optuna optimization
    optimization_results = run_optuna_optimization_lstm(
        volume_data, forecast_horizon, n_trials, study_name
    )

    best_params = optimization_results["best_params"]
    print(f"✅ Optimization completed!")
    print(f"🏆 Best LSTM parameters: {best_params}")

    # Train final model with best parameters
    best_lookback = best_params['lookback']

    # Create sequences with best lookback
    sequences = create_input_sequences(best_lookback, forecast_horizon, volume_data)
    X_combined = np.array(sequences["input_sequences"])
    Y_combined = np.array(sequences["output_sequences"])
    X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)

    # Split data for final training
    train_length = int(0.8 * len(X_combined))
    X_train_final = X_combined[:train_length]
    Y_train_final = Y_combined[:train_length]
    X_test_final = X_combined[train_length:]

    # Create and train final model
    print("📊 Training final LSTM model with best parameters...")
    final_model = create_lstm_model(best_lookback, forecast_horizon)

    # Set the best hyperparameters manually
    final_model = create_lstm_model(best_lookback, forecast_horizon, None)  # Use defaults, then compile with best params

    # Recompile with best learning rate
    optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    final_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7)

    # Train final model
    history = final_model.fit(
        X_train_final, Y_train_final,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Make predictions
    train_predictions = final_model.predict(X_train_final, verbose=0)
    test_predictions = final_model.predict(X_test_final, verbose=0) if len(X_test_final) > 0 else np.array([])

    print(f"📊 Final model training completed!")
    print(f"📊 Best lookback: {best_lookback}")
    print(f"📊 Prediction shapes:")
    print(f"   • Train predictions: {train_predictions.shape}")
    print(f"   • Test predictions: {test_predictions.shape}")

    return final_model, optimization_results, train_predictions, test_predictions


def complete_lstm_original_data_workflow():
    """
    Complete LSTM workflow: load original data, train model, make predictions, and plot results.
    This version trains on original sales data instead of ARIMA residuals.
    """
    print("🚀 Starting Complete LSTM Workflow (Original Data)")
    print("=" * 70)

    try:
        # Step 1: Load and prepare data
        print("📋 Step 1: Loading and preparing original data...")

        # Load original sales data from CSV
        print("📋 Loading original sales data...")
        try:
            original_df = load_original_data_from_csv()
            print("✅ Successfully loaded original sales data from CSV!")
        except Exception as e:
            print(f"❌ Failed to load original sales data: {e}")
            raise ValueError("❌ Could not load original sales data. Please run the pipeline first.")

        # Prepare data for LSTM training
        print("📋 Preparing data for LSTM training...")

        # Calculate appropriate parameters based on data size
        total_data_points = len(original_df)
        print(f"📊 Total data points available: {total_data_points}")

        # Adjust parameters to ensure we have enough data for sequences
        if total_data_points < 50:
            # For small datasets, use smaller lookback and forecast
            lookback = min(8, total_data_points // 4)
            forecast = min(8, total_data_points // 4)
            train_split_ratio = 0.7  # Use more data for training
            print(f"📊 Small dataset detected. Adjusted parameters: lookback={lookback}, forecast={forecast}")
        else:
            lookback = 21
            forecast = 32
            train_split_ratio = 0.8

        lstm_data = prepare_original_data_for_lstm_training(
            original_df,
            lookback=lookback,
            forecast=forecast,
            train_split_ratio=train_split_ratio
        )

        print("✅ Data preparation completed!")

        # Step 2: Train LSTM model
        print("\n📋 Step 2: Training LSTM model...")
        best_model, optimization_results, train_predictions, test_predictions = train_lstm_with_optuna(
            lstm_data,
            n_trials=50,
            study_name='lstm_original_data'
        )

        print("✅ LSTM training completed!")

        # Step 3: Create plotting data
        print("\n📋 Step 3: Preparing data for plotting...")

        # Get the original data for comparison
        train_volume = lstm_data['train_volume']
        test_volume = lstm_data['test_volume']
        scaler = lstm_data['scaler']

        # Create date ranges for plotting
        if 'date' in original_df.columns or isinstance(original_df.index, pd.DatetimeIndex):
            dates = original_df.index if isinstance(original_df.index, pd.DatetimeIndex) else pd.to_datetime(original_df['date'])
            train_dates = dates[:len(train_volume)]
            test_dates = dates[len(train_volume):len(train_volume) + len(test_volume)]
        else:
            # Create synthetic dates if not available
            train_dates = pd.date_range(start='2020-01-01', periods=len(train_volume), freq='W-SAT')
            test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(weeks=1), periods=len(test_volume), freq='W-SAT')

        # Create series for plotting
        train_series = pd.Series(train_volume, index=train_dates, name='Volume')
        test_series = pd.Series(test_volume, index=test_dates, name='Volume')

        print(f"✅ Created training series with shape: {train_series.shape}")
        print(f"✅ Created test series with shape: {test_series.shape}")
        print(f"📊 Training series range: {train_series.min():.0f} to {train_series.max():.0f}")
        print(f"📊 Test series range: {test_series.min():.0f} to {test_series.max():.0f}")

        # Step 4: Create predictions
        print("\n📋 Step 4: Creating LSTM predictions...")

        # Inverse transform predictions to original scale
        train_predictions_original = inverse_transform_predictions(train_predictions.flatten(), scaler)
        test_predictions_original = inverse_transform_predictions(test_predictions.flatten(), scaler)

        print(f"📊 Train predictions shape: {train_predictions_original.shape}")
        print(f"📊 Test predictions shape: {test_predictions_original.shape}")
        print(f"📊 Train predictions range: {train_predictions_original.min():.2f} to {train_predictions_original.max():.2f}")
        print(f"📊 Test predictions range: {test_predictions_original.min():.2f} to {test_predictions_original.max():.2f}")

        # Step 5: Calculate metrics
        print("\n📋 Step 5: Calculating performance metrics...")

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Calculate metrics for test set
        test_actual = test_series.values
        test_pred = test_predictions_original

        # Ensure same length
        min_length = min(len(test_actual), len(test_pred))
        test_actual = test_actual[:min_length]
        test_pred = test_pred[:min_length]

        lstm_mae = mean_absolute_error(test_actual, test_pred)
        lstm_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        lstm_mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

        print(f"\n📊 LSTM Performance Metrics:")
        print(f"   • MAE: {lstm_mae:.2f}")
        print(f"   • RMSE: {lstm_rmse:.2f}")
        print(f"   • MAPE: {lstm_mape:.2f}%")

        # Step 6: Create comparison table
        print("\n📋 Step 6: Creating comparison table...")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Actual Volume': test_actual,
            'LSTM Predictions': test_pred,
        }, index=test_dates[:min_length])

        print("\n" + "=" * 80)
        print("📊 LSTM PREDICTION COMPARISON TABLE (First 8 rows):")
        print("=" * 80)
        print(comparison_df.head(8).round(1))
        print("=" * 80)

        # Step 7: Try to create plots
        print("\n📋 Step 7: Creating evaluation plots...")

        try:
            from utils.hybrid_plotting import comprehensive_model_evaluation

            # Ensure outputs directory exists
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, 'outputs')
            os.makedirs(output_dir, exist_ok=True)

            plotting_results = comprehensive_model_evaluation(
                series_train=train_series,
                series_test=test_series,
                train_predictions=train_predictions_original,
                test_predictions=test_predictions_original,
                Y_test=test_pred.reshape(1, -1),  # Reshape to match expected format
                scaler=scaler,
                first_model_forecast=test_pred,  # Use LSTM predictions as first model
                model_signature="LSTM_Original_Data_Book_Sales_Forecast",
                save_plots=True,
                output_dir=output_dir
            )

            print("✅ Plotting completed!")

        except ImportError as e:
            print(f"⚠️  Plotting module not available: {e}")
            print("📊 Continuing without plots...")
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")
            print("📊 Continuing without plots...")

        # Step 8: Display final results
        print("\n" + "=" * 70)
        print("🎉 COMPLETE LSTM WORKFLOW RESULTS (Original Data):")
        print("=" * 70)
        print(f"📊 Data Summary:")
        print(f"   • Original data points: {len(original_df)}")
        print(f"   • Training data points: {len(train_volume)}")
        print(f"   • Test data points: {len(test_volume)}")
        print(f"   • LSTM training sequences: {lstm_data['X_train'].shape[0]}")
        print(f"   • LSTM test sequences: {lstm_data['X_test'].shape[0]}")

        print(f"\n📊 Model Performance:")
        print(f"   • MAE: {lstm_mae:.2f}")
        print(f"   • RMSE: {lstm_rmse:.2f}")
        print(f"   • MAPE: {lstm_mape:.2f}%")

        print(f"\n📁 Data Sources:")
        print(f"   • Original sales data: combined_test_data.csv")
        print(f"   • Volume column used: {lstm_data['volume_col']}")
        print(f"   • Train/test split ratio: 80/20")

        print(f"\n📁 Outputs:")
        print(f"   • Comparison table saved to comparison DataFrame")
        print(f"   • Model artifacts and predictions available")

        # Create results dictionary
        results = {
            'lstm_forecast': test_pred,
            'actual_values': test_actual,
            'lstm_metrics': {
                'MAE': lstm_mae,
                'RMSE': lstm_rmse,
                'MAPE': lstm_mape
            },
            'train_predictions': train_predictions_original,
            'test_predictions': test_predictions_original,
            'comparison_df': comparison_df
        }

        return results, best_model, lstm_data

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def train_lstm_step(train_data, test_data, output_dir, n_trials=50,
                    lookback=21, forecast_horizon=32, study_name="lstm_optimization"):
    """
    LSTM training step with persistent Optuna storage and ZenML caching.
    Same interface as CNN and ARIMA for pipeline compatibility.
    """
    print("Starting LSTM training with Optuna optimization...")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Parameters: trials={n_trials}, lookback={lookback}, forecast={forecast_horizon}")

    try:
        # Ensure we have the volume column
        if "volume" not in train_data.columns and "Volume" in train_data.columns:
            train_data = train_data.copy()
            train_data["volume"] = train_data["Volume"]
        if "volume" not in test_data.columns and "Volume" in test_data.columns:
            test_data = test_data.copy()
            test_data["volume"] = test_data["Volume"]

        # Prepare data for LSTM
        combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)

        # Create volume series
        if 'date' not in combined_data.columns:
            if "End Date" in combined_data.columns:
                combined_data['date'] = pd.to_datetime(combined_data['End Date'])
            else:
                combined_data['date'] = pd.date_range('2020-01-01', periods=len(combined_data), freq='D')

        combined_data['date'] = pd.to_datetime(combined_data['date'])
        combined_data = combined_data.sort_values('date')

        # Prepare for LSTM training
        lstm_data = prepare_original_data_for_lstm_training(
            combined_data.set_index('date'),
            lookback=lookback,
            forecast=forecast_horizon,
            train_split_ratio=len(train_data) / len(combined_data)
        )

        # Run optimization
        optimization_results = run_optuna_optimization_lstm(
            lstm_data['train_volume_scaled'], forecast_horizon, n_trials, study_name
        )

        best_params = optimization_results["best_params"]
        print(f"Best LSTM parameters: {best_params}")

        # Train final model
        final_model, _, train_predictions, test_predictions = train_lstm_with_optuna(
            lstm_data, n_trials=0, study_name=study_name  # n_trials=0 to skip optimization, use existing results
        )

        # Calculate evaluation metrics
        test_actual = lstm_data['test_volume'][:len(test_predictions.flatten())]
        test_pred = inverse_transform_predictions(test_predictions.flatten(), lstm_data['scaler'])
        test_pred = test_pred[:len(test_actual)]

        eval_metrics = evaluate_forecast(test_actual, test_pred)
        print(f"Evaluation metrics: {eval_metrics}")

        # Create results matching ARIMA/CNN format
        results_data = []

        # Model configuration
        results_data.append({
            "result_type": "model_config",
            "component": "architecture",
            "parameter": "lstm_lookback",
            "value": str(best_params["lookback"]),
            "timestamp": pd.Timestamp.now(),
            "metadata": str(best_params)
        })

        # Evaluation metrics
        for metric_name, metric_value in eval_metrics.items():
            results_data.append({
                "result_type": "evaluation",
                "component": "test_metrics",
                "parameter": metric_name,
                "value": f"{metric_value:.4f}",
                "timestamp": pd.Timestamp.now(),
                "metadata": "test_set_performance"
            })

        results_df = pd.DataFrame(results_data)

        # Create hyperparameters JSON
        hyperparameters_dict = {
            "best_params": best_params,
            "optimization_results": optimization_results,
            "eval_metrics": eval_metrics,
            "study_name": study_name,
            "lookback": best_params["lookback"],
            "forecast_horizon": forecast_horizon,
            "model_signature": f"LSTM_lookback{best_params['lookback']}_units{best_params['input_units']}"
        }

        best_hyperparameters_json = json.dumps(hyperparameters_dict, indent=2, default=str)

        print("LSTM training completed successfully!")
        print(f"Best LSTM parameters: {best_params}")
        print(f"Test performance - RMSE: {eval_metrics['rmse']:.2f}, MAE: {eval_metrics['mae']:.2f}")

        # Create empty DataFrames for compatibility (LSTM doesn't generate residuals)
        residuals_df = pd.DataFrame({
            "date": pd.to_datetime([]),
            "residuals": [],
            "model_signature": hyperparameters_dict["model_signature"]
        })

        # Create forecast DataFrame (consolidating test_predictions and forecast_comparison)
        forecast_df = pd.DataFrame({
            "period": range(1, len(test_actual) + 1),
            "date": lstm_data['original_df'].index[len(lstm_data['train_volume']):len(lstm_data['train_volume']) + len(test_pred)],
            "actual": test_actual,
            "predicted": test_pred,
            "residuals": test_actual - test_pred,
            "absolute_error": np.abs(test_actual - test_pred),
            "percentage_error": np.abs((test_actual - test_pred) / test_actual) * 100,
            "squared_error": (test_actual - test_pred) ** 2,
            "model_signature": hyperparameters_dict["model_signature"]
        })

        return (results_df, best_hyperparameters_json, final_model,
                residuals_df, forecast_df)

    except Exception as e:
        print(f"LSTM training failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return error results matching format
        error_df = pd.DataFrame([{
            "result_type": "error",
            "component": "training_error",
            "parameter": "error_message",
            "value": str(e),
            "timestamp": pd.Timestamp.now(),
            "metadata": "training_failure"
        }])

        error_hyperparameters_dict = {
            "error": str(e),
            "best_params": {"lookback": 21, "input_units": 64, "n_layers": 2,
                           "final_units": 32, "dropout_rate": 0.3, "learning_rate": 0.001},
            "study_name": study_name,
            "model_signature": "ERROR_LSTM_MODEL"
        }

        error_hyperparameters_json = json.dumps(error_hyperparameters_dict, indent=2, default=str)

        # Create empty DataFrames for error case
        error_residuals_df = pd.DataFrame({
            "date": pd.to_datetime([]),
            "residuals": [],
            "model_signature": "ERROR_LSTM_MODEL"
        })

        error_forecast_df = pd.DataFrame({
            "period": [],
            "date": pd.to_datetime([]),
            "actual": [],
            "predicted": [],
            "residuals": [],
            "absolute_error": [],
            "percentage_error": [],
            "squared_error": [],
            "model_signature": []
        })

        return (error_df, error_hyperparameters_json, None,
                error_residuals_df, error_forecast_df)


if __name__ == "__main__":
    """
    Standalone LSTM execution using pipeline data for fair model comparison.
    """
    print("🚀 Running LSTM standalone training with pipeline data...")
    print("=" * 60)

    try:
        # Load pipeline data
        train_data, test_data = load_pipeline_data()

        # Create output directory - use centralized outputs directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🔧 Running LSTM training...")
        print("=" * 60)

        # Run LSTM training
        results = train_lstm_step(
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            n_trials=50,
            lookback=21,
            forecast_horizon=32,
            study_name="standalone_lstm_optimization"
        )

        results_df, hyperparameters_json, model, residuals_df, forecast_df = results

        print("\n✅ LSTM standalone training completed successfully!")
        print("=" * 60)

        # Extract metrics from hyperparameters
        hyperparams = json.loads(hyperparameters_json)
        eval_metrics = hyperparams.get('eval_metrics', {})
        best_params = hyperparams.get('best_params', {})

        print(f"\n📊 LSTM Results Summary:")
        print(f"• Model signature: {hyperparams.get('model_signature', 'LSTM_Model')}")
        print(f"• Best parameters: {best_params}")
        print(f"• Training residuals: {len(residuals_df)} points")
        print(f"• Test predictions: {len(test_predictions_df)} points")
        print(f"• Test MAE: {eval_metrics.get('mae', 0):.2f}")
        print(f"• Test RMSE: {eval_metrics.get('rmse', 0):.2f}")
        print(f"• Test MAPE: {eval_metrics.get('mape', 0):.2f}%")

        # Create organized subdirectories for outputs
        residuals_dir = os.path.join(output_dir, "data", "residuals")
        predictions_dir = os.path.join(output_dir, "data", "predictions")
        comparisons_dir = os.path.join(output_dir, "data", "comparisons")

        os.makedirs(residuals_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(comparisons_dir, exist_ok=True)

        # Save results to organized CSV locations (enhanced forecast comparison includes all metrics)
        forecast_comparison_df.to_csv(f"{comparisons_dir}/lstm_forecast_comparison.csv", index=False)

        # Add plotting functionality
        print(f"\n📋 Creating LSTM forecast plots...")
        try:
            # Create standalone LSTM plotting function
            def create_lstm_standalone_plot(series_train, series_test, lstm_predictions,
                                         eval_metrics, best_params, output_dir):
                """Create standalone LSTM forecast plot."""
                import plotly.graph_objects as go
                import os

                # Create model signature for LSTM
                model_signature = f"LSTM_lookback{best_params['lookback']}_units{best_params['input_units']}"

                # Create the main plot
                fig = go.Figure()

                # Add training data
                fig.add_trace(go.Scatter(
                    x=series_train.index,
                    y=series_train.values,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='blue', width=2),
                    opacity=0.8
                ))

                # Add actual test data
                fig.add_trace(go.Scatter(
                    x=series_test.index,
                    y=series_test.values,
                    mode='lines+markers',
                    name='Actual Test Data',
                    line=dict(color='black', width=3),
                    marker=dict(size=5)
                ))

                # Add LSTM predictions
                fig.add_trace(go.Scatter(
                    x=series_test.index,
                    y=lstm_predictions,
                    mode='lines+markers',
                    name='LSTM Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))

                # Update layout
                title_text = f'LSTM Book Sales Forecast<br><sub>MAE: {eval_metrics["mae"]:.2f} | MAPE: {eval_metrics["mape"]:.2f}% | RMSE: {eval_metrics["rmse"]:.2f}</sub>'

                fig.update_layout(
                    title=title_text,
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )

                # Save plots - create necessary directories
                os.makedirs(f"{output_dir}/plots/interactive", exist_ok=True)
                os.makedirs(f"{output_dir}/plots/static", exist_ok=True)

                # Create descriptive file names with proper folder structure
                html_filename = f"{output_dir}/plots/interactive/lstm_standalone_forecast.html"
                png_filename = f"{output_dir}/plots/static/lstm_standalone_forecast.png"

                # Save files
                fig.write_html(html_filename)
                fig.write_image(png_filename, width=1200, height=500)

                print(f"📁 LSTM standalone plots saved to: {output_dir}")
                print(f"   • HTML: {html_filename}")
                print(f"   • PNG: {png_filename}")

                return {
                    'figure': fig,
                    'model_signature': model_signature,
                    'metrics': eval_metrics
                }

            # Prepare data for plotting
            # Create date series based on test data
            if "End Date" in test_data.columns:
                test_dates = pd.to_datetime(test_data["End Date"])
            else:
                test_dates = pd.date_range('2023-12-16', periods=len(test_predictions_df), freq='W-SAT')

            if "End Date" in train_data.columns:
                # Use actual train dates
                train_dates = pd.to_datetime(train_data["End Date"])
                train_values = train_data["Volume"].values
            else:
                # Create synthetic train dates
                train_dates = pd.date_range('2020-01-01', periods=len(train_data), freq='W-SAT')
                train_values = train_data["Volume"].values if "Volume" in train_data.columns else train_data["volume"].values

            # Create series for plotting
            train_series = pd.Series(train_values, index=train_dates, name='Volume')
            test_series = pd.Series(test_predictions_df['actual'].values, index=test_dates[:len(test_predictions_df)], name='Volume')

            # Create standalone LSTM plot
            plotting_results = create_lstm_standalone_plot(
                series_train=train_series,
                series_test=test_series,
                lstm_predictions=test_predictions_df['predicted'].values,
                eval_metrics=eval_metrics,
                best_params=best_params,
                output_dir=output_dir
            )

            print("✅ LSTM standalone plotting completed!")

        except ImportError as e:
            print(f"⚠️  Plotting module not available: {e}")
            print("📊 Continuing without plots...")
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")
            print("📊 Continuing without plots...")
            import traceback
            traceback.print_exc()

        print(f"\n📁 Generated files in '{output_dir}/':")
        for file in os.listdir(output_dir):
            print(f"  • {file}")

        print(f"\n🎉 LSTM standalone execution completed successfully!")
        print(f"📁 Check the '{output_dir}' directory for generated plots and data files.")

    except Exception as e:
        print(f"\n❌ LSTM standalone training failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
