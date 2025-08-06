import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
import os
import keras_tuner as kt


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


class TunableLookbackHyperModel(kt.HyperModel):
    """HyperModel that handles tunable lookback by creating sequences dynamically."""

    def __init__(self, volume_data, forecast_horizon):
        super().__init__()
        self.volume_data = volume_data
        self.forecast_horizon = forecast_horizon

    def build(self, hp):
        """Build model with current hyperparameters."""
        # Import here to avoid issues
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow import keras

        model = Sequential()

        # Tune the lookback parameter - adjust based on data size
        max_lookback = min(52, len(self.volume_data) // 2)
        min_lookback = min(6, max_lookback // 2)
        lookback = hp.Int('lookback', min_value=min_lookback, max_value=max_lookback, step=2)

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
        sequences = create_input_sequences(lookback, self.forecast_horizon, self.volume_data)
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
    max_trials: int = 50,
    epochs: int = 50,
    project_name: str = 'lstm_original_data_tuning'
) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    """
    Train LSTM model with Keras Tuner hyperparameter optimization.

    Args:
        lstm_data: Prepared LSTM data from prepare_original_data_for_lstm_training
        max_trials: Maximum tuning trials
        epochs: Training epochs
        project_name: Tuner project name

    Returns:
        Tuple of (best_model, tuner, train_predictions, test_predictions)
    """
    print("🚀 Starting LSTM training with Keras Tuner (Original Data)...")
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

    # Extract volume data for tunable lookback approach
    volume_data = lstm_data['train_volume_scaled']
    forecast_horizon = lstm_data['forecast']

    print(f"📊 Using volume data shape: {volume_data.shape}")
    print(f"📊 Forecast horizon: {forecast_horizon}")

    # Create the tunable hypermodel
    hypermodel = TunableLookbackHyperModel(volume_data, forecast_horizon)

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
    sequences = create_input_sequences(best_lookback, forecast_horizon, volume_data)
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
        best_model, tuner, train_predictions, test_predictions = train_lstm_with_tuner(
            lstm_data,
            max_trials=50,
            epochs=50,
            project_name='lstm_original_data'
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
            from hybrid_plotting import comprehensive_model_evaluation
            
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


if __name__ == "__main__":
    # Run the complete workflow
    results, best_model, lstm_data = complete_lstm_original_data_workflow() 