import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import optuna
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_time_series_from_df(df: pd.DataFrame, target_col: str = "volume",
                              date_col: str = "date") -> pd.Series:
    """Convert DataFrame to time series for CNN modeling."""
    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work[date_col]):
        df_work[date_col] = pd.to_datetime(df_work[date_col])

    df_work = df_work.sort_values(date_col)
    time_series = df_work.groupby(date_col)[target_col].sum()

    return time_series

def prepare_sequences_for_cnn(series: pd.Series, sequence_length: int = 12,
                             forecast_horizon: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare time series data for CNN input."""
    print(f"Preparing CNN sequences: length={sequence_length}, horizon={forecast_horizon}")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    X, y = [], []

    # Create sequences
    for i in range(len(scaled_data) - sequence_length - forecast_horizon + 1):
        # Input sequence: past 'sequence_length' observations
        X.append(scaled_data[i:(i + sequence_length)])
        # Output: next value for single-step prediction
        y.append(scaled_data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for CNN: [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y, scaler

def create_cnn_model(sequence_length: int, trial: optuna.Trial = None) -> keras.Model:
    """Create CNN model for time series forecasting."""
    if trial is not None:
        # Hyperparameter optimization
        n_filters = trial.suggest_categorical('n_filters', [32, 64, 128, 256])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        pool_size = trial.suggest_categorical('pool_size', [2, 3])
        dense_units = trial.suggest_categorical('dense_units', [32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    else:
        # Default parameters
        n_filters = 64
        kernel_size = 3
        n_conv_layers = 2
        pool_size = 2
        dense_units = 128
        dropout_rate = 0.3
        learning_rate = 0.001

    model = Sequential()

    # First Conv1D layer
    model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu',
                     input_shape=(sequence_length, 1)))
    model.add(BatchNormalization())

    # Additional Conv1D layers
    for i in range(n_conv_layers - 1):
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
        model.add(BatchNormalization())

    # Pooling
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(dropout_rate))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Single output for next time step

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics with error handling."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}

def objective(trial, X_train: np.ndarray, X_val: np.ndarray,
              y_train: np.ndarray, y_val: np.ndarray, sequence_length: int) -> float:
    """Optuna objective function to minimize validation RMSE."""
    try:
        # Create model with trial hyperparameters
        model = create_cnn_model(sequence_length, trial)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Train model
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        # Evaluate on validation set
        val_pred = model.predict(X_val, verbose=0)
        val_loss = mean_squared_error(y_val, val_pred.flatten())

        return val_loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

def run_optuna_optimization(X_train: np.ndarray, y_train: np.ndarray,
                           sequence_length: int, n_trials: int,
                           study_name: str = "cnn_optimization") -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization with persistent storage."""
    print(f"Starting CNN Optuna optimization with {n_trials} trials...")

    # Create validation split from training data
    val_split = 0.2
    val_size = int(len(X_train) * val_split)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_opt = X_train[:-val_size]
    y_train_opt = y_train[:-val_size]

    print(f"Optimization data splits:")
    print(f"   Training: {X_train_opt.shape}")
    print(f"   Validation: {X_val.shape}")

    # Set up Optuna storage (same as ARIMA)
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
            lambda trial: objective(trial, X_train_opt, X_val, y_train_opt, y_val, sequence_length),
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
                "n_filters": 64,
                "kernel_size": 3,
                "n_conv_layers": 2,
                "pool_size": 2,
                "dense_units": 128,
                "dropout_rate": 0.3,
                "learning_rate": 0.001
            },
            "best_value": float("inf"),
            "n_trials": 0,
            "study_name": study_name,
            "storage_url": storage_url,
            "error": str(e)
        }

def train_final_cnn_model(X_train: np.ndarray, y_train: np.ndarray,
                         sequence_length: int, best_params: Dict[str, Any]) -> keras.Model:
    """Train final CNN model with best parameters."""
    print(f"Training final CNN model with best parameters...")
    print(f"Best params: {best_params}")

    # Build model with optimized hyperparameters
    model = Sequential()

    model.add(Conv1D(filters=best_params['n_filters'],
                     kernel_size=best_params['kernel_size'],
                     activation='relu', input_shape=(sequence_length, 1)))
    model.add(BatchNormalization())

    for i in range(best_params['n_conv_layers'] - 1):
        model.add(Conv1D(filters=best_params['n_filters'],
                         kernel_size=best_params['kernel_size'], activation='relu'))
        model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=best_params['pool_size']))
    model.add(Dropout(best_params['dropout_rate']))
    model.add(Flatten())
    model.add(Dense(best_params['dense_units'], activation='relu'))
    model.add(Dropout(best_params['dropout_rate']))
    model.add(Dense(1))

    # Compile with optimized learning rate
    optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7)

    # Train the final model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print(f"Final model training completed!")
    print(f"Final loss: {history.history['loss'][-1]:.6f}")

    return model

def make_multi_step_predictions(model: keras.Model, X_test: np.ndarray,
                               forecast_horizon: int, scaler) -> np.ndarray:
    """Make multi-step predictions using trained CNN model."""
    print(f"Making multi-step predictions for {forecast_horizon} steps...")

    predictions = []
    current_sequence = X_test[0].copy()  # Start with first test sequence

    for step in range(forecast_horizon):
        # Predict next value
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)[0][0]
        predictions.append(next_pred)

        # Update sequence: remove first value, append prediction
        current_sequence = np.append(current_sequence[1:], next_pred)

    # Convert back to original scale
    predictions = np.array(predictions)
    predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    print(f"Generated {len(predictions_original)} predictions")
    print(f"Prediction range: {predictions_original.min():.2f} to {predictions_original.max():.2f}")

    return predictions_original

def train_cnn_step(train_data, test_data, output_dir, n_trials=40,
                  sequence_length=12, forecast_horizon=32, study_name="cnn_optimization"):
    """
    CNN training step with persistent Optuna storage and ZenML caching.
    Same interface as train_arima_optuna_step for pipeline compatibility.
    """
    import pandas as pd
    import numpy as np
    import json

    print("Starting CNN training with Optuna optimization...")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Parameters: trials={n_trials}, seq_len={sequence_length}, forecast={forecast_horizon}")

    try:
        # Ensure we have the volume column
        if "volume" not in train_data.columns and "Volume" in train_data.columns:
            train_data = train_data.copy()
            train_data["volume"] = train_data["Volume"]
        if "volume" not in test_data.columns and "Volume" in test_data.columns:
            test_data = test_data.copy()
            test_data["volume"] = test_data["Volume"]

        # Create time series from DataFrames
        train_df_reset = train_data.reset_index()
        if "date" not in train_df_reset.columns:
            if "End Date" in train_df_reset.columns:
                train_df_reset["date"] = pd.to_datetime(train_df_reset["End Date"])
            elif pd.api.types.is_datetime64_any_dtype(train_data.index):
                train_df_reset["date"] = train_data.index
            else:
                raise ValueError("Could not determine date column for training data")

        test_df_reset = test_data.reset_index()
        if "date" not in test_df_reset.columns:
            if "End Date" in test_df_reset.columns:
                test_df_reset["date"] = pd.to_datetime(test_df_reset["End Date"])
            elif pd.api.types.is_datetime64_any_dtype(test_data.index):
                test_df_reset["date"] = test_data.index
            else:
                raise ValueError("Could not determine date column for test data")

        # Create time series
        train_series = create_time_series_from_df(train_df_reset, target_col="volume", date_col="date")
        test_series = create_time_series_from_df(test_df_reset, target_col="volume", date_col="date")

        print(f"Training series shape: {train_series.shape}")
        print(f"Test series shape: {test_series.shape}")

        # Prepare sequences for CNN
        X_train, y_train, train_scaler = prepare_sequences_for_cnn(
            train_series, sequence_length, forecast_horizon
        )

        print(f"CNN training sequences: X={X_train.shape}, y={y_train.shape}")

        # Run Optuna optimization
        optimization_results = run_optuna_optimization(
            X_train, y_train, sequence_length, n_trials, study_name
        )

        best_params = optimization_results["best_params"]
        print(f"Best CNN parameters: {best_params}")

        # Train final model with best parameters
        final_model = train_final_cnn_model(X_train, y_train, sequence_length, best_params)

        # Make predictions on test set
        combined_series = pd.concat([train_series, test_series])
        combined_scaled = train_scaler.transform(combined_series.values.reshape(-1, 1)).flatten()

        # Use the last sequence_length values from training to predict test
        test_input_sequence = combined_scaled[len(train_series)-sequence_length:len(train_series)]
        test_input = test_input_sequence.reshape(1, sequence_length, 1)

        # Make multi-step predictions
        test_predictions_scaled = make_multi_step_predictions(
            final_model, test_input, forecast_horizon, train_scaler
        )

        # Ensure we have the right number of predictions
        test_predictions_scaled = test_predictions_scaled[:len(test_series)]
        actual_test_values = test_series.values[:len(test_predictions_scaled)]

        print(f"Test predictions shape: {test_predictions_scaled.shape}")
        print(f"Actual test values shape: {actual_test_values.shape}")

        # Calculate evaluation metrics
        eval_metrics = evaluate_forecast(actual_test_values, test_predictions_scaled)
        print(f"Evaluation metrics: {eval_metrics}")

        # Extract residuals from training data
        train_predictions_list = []
        train_residuals_list = []

        print("Extracting training residuals...")
        for i in range(len(X_train)):
            train_pred_scaled = final_model.predict(X_train[i:i+1], verbose=0)[0][0]
            train_pred_original = train_scaler.inverse_transform([[train_pred_scaled]])[0][0]
            train_actual_original = train_scaler.inverse_transform([[y_train[i]]])[0][0]

            train_predictions_list.append(train_pred_original)
            train_residuals_list.append(train_actual_original - train_pred_original)

        train_residuals = np.array(train_residuals_list)

        # Create residuals DataFrame
        residuals_dates = train_series.index[sequence_length:sequence_length+len(train_residuals)]
        residuals_df = pd.DataFrame({
            "date": residuals_dates,
            "residuals": train_residuals,
            "model_signature": f"CNN_filters{best_params['n_filters']}_kernel{best_params['kernel_size']}_layers{best_params['n_conv_layers']}"
        })

        # Create test predictions DataFrame
        test_predictions_df = pd.DataFrame({
            "date": test_series.index[:len(test_predictions_scaled)],
            "actual": actual_test_values,
            "predicted": test_predictions_scaled,
            "residuals": actual_test_values - test_predictions_scaled,
            "absolute_error": np.abs(actual_test_values - test_predictions_scaled),
            "model_signature": f"CNN_filters{best_params['n_filters']}_kernel{best_params['kernel_size']}_layers{best_params['n_conv_layers']}"
        })

        # Create forecast comparison DataFrame
        forecast_comparison_df = pd.DataFrame({
            "period": range(1, len(actual_test_values) + 1),
            "date": test_series.index[:len(test_predictions_scaled)],
            "actual_volume": actual_test_values,
            "predicted_volume": test_predictions_scaled,
            "absolute_error": np.abs(actual_test_values - test_predictions_scaled),
            "percentage_error": np.abs((actual_test_values - test_predictions_scaled) / actual_test_values) * 100,
            "squared_error": (actual_test_values - test_predictions_scaled) ** 2,
            "model_signature": f"CNN_filters{best_params['n_filters']}_kernel{best_params['kernel_size']}_layers{best_params['n_conv_layers']}"
        })

        # Save residuals to CSV for LSTM integration
        residuals_csv_path = os.path.join(output_dir, "cnn_residuals.csv")
        residuals_df.to_csv(residuals_csv_path, index=False)
        print(f"Saved residuals to: {residuals_csv_path}")

        # Save CNN forecasts to CSV for LSTM integration
        cnn_forecasts_csv_path = os.path.join(output_dir, "cnn_forecasts.csv")
        test_predictions_df.to_csv(cnn_forecasts_csv_path, index=False)
        print(f"Saved CNN forecasts to: {cnn_forecasts_csv_path}")

        # Save forecast comparison to CSV
        forecast_comparison_csv_path = os.path.join(output_dir, "cnn_forecast_comparison.csv")
        forecast_comparison_df.to_csv(forecast_comparison_csv_path, index=False)
        print(f"Saved forecast comparison to: {forecast_comparison_csv_path}")

        # Create results DataFrame matching ARIMA format
        results_data = []

        # Model configuration
        results_data.append({
            "result_type": "model_config",
            "component": "architecture",
            "parameter": "cnn_filters",
            "value": str(best_params["n_filters"]),
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
            "sequence_length": sequence_length,
            "forecast_horizon": forecast_horizon,
            "model_signature": f"CNN_filters{best_params['n_filters']}_kernel{best_params['kernel_size']}_layers{best_params['n_conv_layers']}"
        }

        best_hyperparameters_json = json.dumps(hyperparameters_dict, indent=2, default=str)

        print("CNN training completed successfully!")
        print(f"Best CNN parameters: {best_params}")
        print(f"Test performance - RMSE: {eval_metrics['rmse']:.2f}, MAE: {eval_metrics['mae']:.2f}")

        # Step 6: Comprehensive evaluation and plotting
        print("\n📋 Step 6: Creating comprehensive evaluation plots...")
        
        try:
            # Create standalone CNN plotting function
            def create_cnn_standalone_plot(series_train, series_test, cnn_predictions, 
                                         eval_metrics, best_params, output_dir):
                """Create standalone CNN forecast plot."""
                import plotly.graph_objects as go
                import plotly.subplots as sp
                import os
                
                # Create model signature for CNN
                model_signature = f"CNN_Book_Sales_Forecast_filters{best_params['n_filters']}_kernel{best_params['kernel_size']}_layers{best_params['n_conv_layers']}"
                
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
                
                # Add CNN predictions
                fig.add_trace(go.Scatter(
                    x=series_test.index,
                    y=cnn_predictions,
                    mode='lines+markers',
                    name='CNN Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))
                
                # Update layout
                title_text = f'CNN Book Sales Forecast<br><sub>MAE: {eval_metrics["mae"]:.2f} | MAPE: {eval_metrics["mape"]:.2f}% | RMSE: {eval_metrics["rmse"]:.2f}</sub>'
                
                fig.update_layout(
                    title=title_text,
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )
                
                # Save plots
                os.makedirs(output_dir, exist_ok=True)
                
                # Create descriptive file names
                safe_model_name = model_signature.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                html_filename = f"{output_dir}/cnn_standalone_forecast.html"
                png_filename = f"{output_dir}/cnn_standalone_forecast.png"
                csv_filename = f"{output_dir}/cnn_standalone_comparison.csv"
                
                # Save files
                fig.write_html(html_filename)
                fig.write_image(png_filename, width=1200, height=500)
                
                # Ensure arrays have the same length
                min_length = min(len(series_test), len(cnn_predictions))
                series_test_aligned = series_test.iloc[:min_length]
                cnn_predictions_aligned = cnn_predictions[:min_length]
                
                # Save comparison data
                comparison_df = pd.DataFrame({
                    'date': series_test_aligned.index,
                    'actual': series_test_aligned.values,
                    'cnn_predicted': cnn_predictions_aligned,
                    'absolute_error': np.abs(series_test_aligned.values - cnn_predictions_aligned),
                    'percentage_error': np.abs((series_test_aligned.values - cnn_predictions_aligned) / series_test_aligned.values) * 100,
                    'squared_error': (series_test_aligned.values - cnn_predictions_aligned) ** 2,
                    'model_signature': model_signature
                })
                comparison_df.to_csv(csv_filename, index=False)
                
                print(f"📁 CNN standalone plots saved to: {output_dir}")
                print(f"   • HTML: {html_filename}")
                print(f"   • PNG: {png_filename}")
                print(f"   • CSV: {csv_filename}")
                
                return {
                    'figure': fig,
                    'comparison_df': comparison_df,
                    'model_signature': model_signature,
                    'metrics': eval_metrics
                }
            
            # Create standalone CNN plot
            plotting_results = create_cnn_standalone_plot(
                series_train=train_series,
                series_test=test_series,
                cnn_predictions=test_predictions_scaled,
                eval_metrics=eval_metrics,
                best_params=best_params,
                output_dir=output_dir
            )
            
            print("✅ CNN standalone plotting completed!")
            print(f"📊 Plotting results: {list(plotting_results.keys())}")
            
        except ImportError as e:
            print(f"⚠️  Plotting module not available: {e}")
            print("📊 Continuing without plots...")
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")
            print("📊 Continuing without plots...")
            import traceback
            traceback.print_exc()

        return (results_df, best_hyperparameters_json, final_model,
                residuals_df, test_predictions_df, forecast_comparison_df)

    except Exception as e:
        print(f"CNN training failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return error results matching ARIMA format
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
            "best_params": {"n_filters": 64, "kernel_size": 3, "n_conv_layers": 2,
                           "pool_size": 2, "dense_units": 128, "dropout_rate": 0.3, "learning_rate": 0.001},
            "study_name": study_name,
            "model_signature": "ERROR_CNN_MODEL"
        }

        error_hyperparameters_json = json.dumps(error_hyperparameters_dict, indent=2, default=str)

        # Create empty DataFrames for error case
        error_residuals_df = pd.DataFrame({
            "date": pd.to_datetime([]),
            "residuals": [],
            "model_signature": "ERROR_CNN_MODEL"
        })

        error_test_predictions_df = pd.DataFrame({
            "date": pd.to_datetime([]),
            "actual": [],
            "predicted": [],
            "residuals": [],
            "absolute_error": [],
            "model_signature": "ERROR_CNN_MODEL"
        })

        error_forecast_comparison_df = pd.DataFrame({
            "period": [],
            "date": pd.to_datetime([]),
            "actual_volume": [],
            "predicted_volume": [],
            "absolute_error": [],
            "percentage_error": [],
            "squared_error": [],
            "model_signature": "ERROR_CNN_MODEL"
        })

        return (error_df, error_hyperparameters_json, None,
                error_residuals_df, error_test_predictions_df, error_forecast_comparison_df)


if __name__ == "__main__":
    """
    Standalone CNN execution using pipeline data for fair model comparison.
    """
    print("🚀 Running CNN standalone training with pipeline data...")
    print("=" * 60)
    
    try:
        # Load pipeline data
        train_data, test_data = load_pipeline_data()
        
        # Create output directory
        output_dir = "cnn_standalone_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔧 Running CNN training...")
        print("=" * 60)
        
        # Run CNN training
        results = train_cnn_step(
            train_data=train_data,
            test_data=test_data,
            output_dir=output_dir,
            n_trials=50,
            sequence_length=12,
            forecast_horizon=32,
            study_name="standalone_cnn_optimization"
        )
        
        results_df, hyperparameters_json, model, residuals_df, test_predictions_df, forecast_comparison_df = results
        
        print("\n✅ CNN standalone training completed successfully!")
        print("=" * 60)
        
        # Extract metrics from hyperparameters
        hyperparams = json.loads(hyperparameters_json)
        eval_metrics = hyperparams.get('eval_metrics', {})
        best_params = hyperparams.get('best_params', {})
        
        print(f"\n📊 CNN Results Summary:")
        print(f"• Model signature: {hyperparams.get('model_signature', 'CNN_Model')}")
        print(f"• Best parameters: {best_params}")
        print(f"• Training residuals: {len(residuals_df)} points")
        print(f"• Test predictions: {len(test_predictions_df)} points")
        print(f"• Test MAE: {eval_metrics.get('mae', 0):.2f}")
        print(f"• Test RMSE: {eval_metrics.get('rmse', 0):.2f}")
        print(f"• Test MAPE: {eval_metrics.get('mape', 0):.2f}%")
        
        print(f"\n📁 Generated files in '{output_dir}/':")
        for file in os.listdir(output_dir):
            if file.endswith(('.csv', '.html', '.png')):
                print(f"  • {file}")
        
        print(f"\n🎉 CNN standalone execution completed successfully!")
        print(f"📁 Check the '{output_dir}' directory for generated plots and data files.")
        
    except Exception as e:
        print(f"\n❌ CNN standalone training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
