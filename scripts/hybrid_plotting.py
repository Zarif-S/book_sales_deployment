"""
Hybrid Model Plotting and Evaluation Module

This module provides functions to visualize and evaluate the performance of 
hybrid models (e.g., First Model + LSTM), including prediction plotting, 
metrics calculation, and comprehensive performance analysis using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Tuple, Optional, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Import our data loading functions
from _05_lstm import load_arima_residuals_from_csv, prepare_residuals_for_lstm_training


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE as a percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return float('inf')
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': calculate_mape(y_true, y_pred)
    }
    
    return metrics


def process_lstm_predictions(
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    Y_test: np.ndarray,
    scaler: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process LSTM residual predictions by inverse scaling.
    Note: LSTM predicts sequences of residuals. For hybrid forecasting, we typically
    want the last prediction sequence (final forecast horizon).
    
    Args:
        train_predictions: Raw training predictions from LSTM (residuals) - shape (n_train, forecast_horizon)
        test_predictions: Raw test predictions from LSTM (residuals) - shape (n_test, forecast_horizon)  
        Y_test: Test targets (residuals) - shape (n_test, forecast_horizon)
        scaler: Fitted MinMaxScaler from residuals training
        
    Returns:
        Tuple of (final_train_residuals, final_test_residuals, Y_test_residuals)
    """
    print("🔄 Processing LSTM residual predictions...")
    print(f"📊 Raw train predictions shape: {train_predictions.shape}")
    print(f"📊 Raw test predictions shape: {test_predictions.shape}")
    print(f"📊 Y_test shape: {Y_test.shape}")
    
    # For hybrid forecasting, we want the LAST prediction from the test set
    # This represents the model's prediction for the next forecast_horizon periods
    final_test_prediction = test_predictions[-1]  # Last test prediction sequence
    final_Y_test = Y_test[-1]  # Corresponding actual values
    
    # Also get the last training prediction for completeness
    final_train_prediction = train_predictions[-1]
    
    print(f"📊 Using final test prediction shape: {final_test_prediction.shape}")
    print(f"📊 Using final Y_test shape: {final_Y_test.shape}")
    
    # Inverse transform the residual predictions (from scaled residuals back to residuals)
    final_train_residuals = scaler.inverse_transform(final_train_prediction.reshape(-1, 1)).flatten()
    final_test_residuals = scaler.inverse_transform(final_test_prediction.reshape(-1, 1)).flatten()
    final_Y_test_residuals = scaler.inverse_transform(final_Y_test.reshape(-1, 1)).flatten()
    
    print(f"📊 Final train residuals shape: {final_train_residuals.shape}")
    print(f"📊 Final test residuals shape: {final_test_residuals.shape}")
    print(f"📊 Final Y_test residuals shape: {final_Y_test_residuals.shape}")
    
    # Print some sample values to verify they look like residuals (should be small values around 0)
    print(f"📊 Sample test residuals: {final_test_residuals[:5]}")
    print(f"📊 Test residuals stats - Mean: {final_test_residuals.mean():.2f}, Std: {final_test_residuals.std():.2f}")
    
    return final_train_residuals, final_test_residuals, final_Y_test_residuals


def create_hybrid_forecast(
    first_model_forecast: np.ndarray,
    lstm_residuals_prediction: np.ndarray
) -> np.ndarray:
    """
    Combine first model forecast with LSTM residuals prediction to create hybrid forecast.
    
    Args:
        first_model_forecast: First model forecast values
        lstm_residuals_prediction: LSTM predicted residuals
        
    Returns:
        Combined hybrid forecast
    """
    print("🔧 Creating hybrid forecast...")
    print(f"📊 First model forecast shape: {first_model_forecast.shape}")
    print(f"📊 LSTM residuals prediction shape: {lstm_residuals_prediction.shape}")
    
    # Ensure both arrays have the same length
    min_length = min(len(first_model_forecast), len(lstm_residuals_prediction))
    first_model_forecast = first_model_forecast[:min_length]
    lstm_residuals_prediction = lstm_residuals_prediction[:min_length]
    
    # Combine first model predictions with LSTM predicted residuals
    final_forecast = first_model_forecast + lstm_residuals_prediction
    
    print(f"📊 Final hybrid forecast shape: {final_forecast.shape}")
    print(f"📊 Hybrid forecast stats - Mean: {final_forecast.mean():.2f}, Std: {final_forecast.std():.2f}")
    
    return final_forecast


def plot_prediction_comparison(
    series_train: pd.Series,
    series_test: pd.Series,
    hybrid_forecast: np.ndarray,
    first_model_forecast: Optional[np.ndarray] = None,
    lstm_residuals: Optional[np.ndarray] = None,
    title: str = "Hybrid Model Forecast",
    height: int = 500
) -> Tuple[go.Figure, Dict[str, float]]:
    """
    Plot simple forecast comparison with training data, test data, and predictions.
    
    Args:
        series_train: Training data series
        series_test: Test data series (ground truth)
        hybrid_forecast: Hybrid model predictions
        first_model_forecast: Optional first model predictions
        lstm_residuals: Optional LSTM residuals predictions (ignored in simple version)
        title: Plot title
        height: Figure height
        
    Returns:
        Tuple of (figure, metrics_dict)
    """
    print("📈 Creating forecast comparison plot...")
    
    # Calculate metrics for hybrid forecast
    metrics = calculate_metrics(series_test.values, hybrid_forecast)
    
    # Create simple single plot
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(
        go.Scatter(x=series_train.index, y=series_train.values,
                  mode='lines', name='Training Data',
                  line=dict(color='blue', width=1),
                  opacity=0.7)
    )
    
    # Add test data (actual values)
    fig.add_trace(
        go.Scatter(x=series_test.index, y=series_test.values,
                  mode='lines+markers', name='Actual Test Data',
                  line=dict(color='red', width=3),
                  marker=dict(size=4))
    )
    
    # Add hybrid model predictions
    fig.add_trace(
        go.Scatter(x=series_test.index, y=hybrid_forecast,
                  mode='lines+markers', name='Hybrid Model',
                  line=dict(color='green', width=2, dash='dash'),
                  marker=dict(size=3))
    )
    
    # Add first model forecast if provided
    if first_model_forecast is not None:
        fig.add_trace(
            go.Scatter(x=series_test.index, y=first_model_forecast,
                      mode='lines', name='First Model Only',
                      line=dict(color='orange', width=1),
                      opacity=0.8)
        )
    
    # Add metrics annotation
    metrics_text = f"MAE: {metrics['MAE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | RMSE: {metrics['RMSE']:.2f}"
    
    # Update layout
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{metrics_text}</sub>", x=0.5),
        xaxis_title="Date",
        yaxis_title="Volume",
        height=height,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )
    
    print(f"📊 Hybrid Model Performance:")
    print(f"   • MAE: {metrics['MAE']:.2f}")
    print(f"   • MAPE: {metrics['MAPE']:.2f}%")
    print(f"   • RMSE: {metrics['RMSE']:.2f}")
    
    return fig, metrics


def plot_residuals_analysis(
    lstm_residuals_prediction: np.ndarray,
    actual_residuals: Optional[np.ndarray] = None,
    title: str = "LSTM Residuals Prediction Analysis",
    height: int = 500
) -> go.Figure:
    """
    Plot analysis of LSTM residuals predictions using Plotly.
    
    Args:
        lstm_residuals_prediction: LSTM predicted residuals
        actual_residuals: Actual residuals for comparison (optional)
        title: Plot title
        height: Figure height
        
    Returns:
        Plotly Figure
    """
    print("📈 Creating residuals analysis plot...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals Time Series', 'Residuals Distribution'),
        horizontal_spacing=0.12
    )
    
    # Time series plot of residuals
    fig.add_trace(
        go.Scatter(y=lstm_residuals_prediction, mode='lines+markers',
                  name='LSTM Predicted Residuals',
                  line=dict(color='green'),
                  marker=dict(size=3)),
        row=1, col=1
    )
    
    if actual_residuals is not None:
        fig.add_trace(
            go.Scatter(y=actual_residuals, mode='lines+markers',
                      name='Actual Residuals',
                      line=dict(color='red'),
                      marker=dict(size=3),
                      opacity=0.6),
            row=1, col=1
        )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", 
                  opacity=0.5, row=1, col=1)
    
    # Residuals distribution
    fig.add_trace(
        go.Histogram(x=lstm_residuals_prediction, nbinsx=20,
                    name='LSTM Predicted', marker_color='green',
                    opacity=0.7),
        row=1, col=2
    )
    
    if actual_residuals is not None:
        fig.add_trace(
            go.Histogram(x=actual_residuals, nbinsx=20,
                        name='Actual', marker_color='red',
                        opacity=0.5),
            row=1, col=2
        )
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", 
                  opacity=0.7, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        height=height,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig


def create_comparison_dataframe(
    actual_data: np.ndarray,
    hybrid_forecast: np.ndarray,
    first_model_forecast: Optional[np.ndarray] = None,
    lstm_residuals: Optional[np.ndarray] = None,
    index: Optional[pd.Index] = None
) -> pd.DataFrame:
    """
    Create a comparison DataFrame with all model predictions.
    
    Args:
        actual_data: Ground truth values
        hybrid_forecast: Hybrid model predictions
        first_model_forecast: First model predictions (optional)
        lstm_residuals: LSTM residuals predictions (optional)
        index: Index for the DataFrame (optional)
        
    Returns:
        Comparison DataFrame
    """
    print("📋 Creating comparison DataFrame...")
    
    comparison_data = {
        'Actual': actual_data,
        'Hybrid_Model': hybrid_forecast
    }
    
    if first_model_forecast is not None:
        comparison_data['First_Model_Only'] = first_model_forecast
    
    if lstm_residuals is not None:
        comparison_data['LSTM_Residuals'] = lstm_residuals
    
    df = pd.DataFrame(comparison_data, index=index)
    
    print(f"📊 Comparison DataFrame shape: {df.shape}")
    print("📊 DataFrame columns:", list(df.columns))
    
    return df


def comprehensive_model_evaluation(
    series_train: pd.Series,
    series_test: pd.Series,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    Y_test: np.ndarray,
    scaler: Any,
    first_model_forecast: Optional[np.ndarray] = None,
    model_signature: str = "Hybrid Model",
    save_plots: bool = False,
    output_dir: str = "plots"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation and plotting of hybrid model using Plotly.
    
    Args:
        series_train: Training data series
        series_test: Test data series
        train_predictions: Raw LSTM training predictions
        test_predictions: Raw LSTM test predictions
        Y_test: Test targets
        scaler: Fitted scaler
        first_model_forecast: First model forecast (optional)
        model_signature: Model identifier string
        save_plots: Whether to save plots
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with results and metrics
    """
    print("🚀 Starting comprehensive model evaluation...")
    print("=" * 60)
    
    # Process LSTM predictions
    train_pred_residuals, test_pred_residuals, Y_test_residuals = process_lstm_predictions(
        train_predictions, test_predictions, Y_test, scaler
    )
    
    # Create hybrid forecast if first model forecast is provided
    if first_model_forecast is not None:
        # Ensure we only use the test portion length
        test_length = len(series_test)
        lstm_residuals_test = test_pred_residuals[:test_length]
        first_model_test = first_model_forecast[:test_length]
        
        hybrid_forecast = create_hybrid_forecast(first_model_test, lstm_residuals_test)
    else:
        # If no first model forecast, use LSTM predictions directly
        hybrid_forecast = test_pred_residuals[:len(series_test)]
        print("⚠️  No first model forecast provided, using LSTM predictions only")
    
    # Create comparison DataFrame
    comparison_df = create_comparison_dataframe(
        actual_data=series_test.values,
        hybrid_forecast=hybrid_forecast,
        first_model_forecast=first_model_forecast[:len(series_test)] if first_model_forecast is not None else None,
        lstm_residuals=test_pred_residuals[:len(series_test)],
        index=series_test.index
    )
    
    print("\n" + "=" * 60)
    print("📊 Model Comparison Summary:")
    print(comparison_df.describe())
    
    # Create main prediction plot
    title = f'Hybrid Forecast - {model_signature}'
    fig_main, metrics = plot_prediction_comparison(
        series_train=series_train,
        series_test=series_test,
        hybrid_forecast=hybrid_forecast,
        first_model_forecast=first_model_forecast[:len(series_test)] if first_model_forecast is not None else None,
        lstm_residuals=test_pred_residuals[:len(series_test)],
        title=title
    )
    
    # Save plots if requested
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig_main.write_html(f"{output_dir}/hybrid_forecast_comparison.html")
        fig_main.write_image(f"{output_dir}/hybrid_forecast_comparison.png", 
                            width=1200, height=500)
        
        # Save comparison data
        comparison_df.to_csv(f"{output_dir}/model_comparison_data.csv")
        
        print(f"📁 Plots and data saved to: {output_dir}")
    
    # Calculate additional metrics
    if first_model_forecast is not None:
        first_model_metrics = calculate_metrics(series_test.values, first_model_forecast[:len(series_test)])
        print(f"\n📊 First Model Performance:")
        print(f"   • MAE: {first_model_metrics['MAE']:.2f}")
        print(f"   • MAPE: {first_model_metrics['MAPE']:.2f}%")
    else:
        first_model_metrics = None
    
    # Return comprehensive results
    results = {
        'hybrid_metrics': metrics,
        'first_model_metrics': first_model_metrics,
        'comparison_df': comparison_df,
        'hybrid_forecast': hybrid_forecast,
        'lstm_residuals_prediction': test_pred_residuals[:len(series_test)],
        'figures': {
            'main_plot': fig_main
        },
        'model_signature': model_signature
    }
    
    print("\n🎉 Comprehensive evaluation completed!")
    return results


# Demo function removed - use only real data through the main LSTM workflow


def plot_prediction(series_train, series_test, forecast, forecast_int=None, fitted_values=None, title="Forecast Plot"):
    """
    Updated plot_prediction function - exact replica of user's original implementation.
    
    Creates a clean, focused forecast plot with training data, test data, and forecast.
    Returns the figure and key metrics (MAE, MAPE).
    """
    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)

    # Create a figure
    fig = go.Figure()

    # Add the training data trace
    fig.add_trace(go.Scatter(x=series_train.index, y=series_train,
                             mode='lines', name='Train / Actual',
                             line=dict(color='blue')))

    # Add the test data (actual values) trace
    fig.add_trace(go.Scatter(x=series_test.index, y=series_test,
                             mode='lines', name='Test / Actual',
                             line=dict(color='black')))

    # Add the forecast data trace
    fig.add_trace(go.Scatter(x=series_test.index, y=forecast,
                             mode='lines', name='Forecast',
                             line=dict(color='red')))

    # Add fitted values if provided
    if fitted_values is not None:
        fig.add_trace(go.Scatter(x=fitted_values.index, y=fitted_values,
                                 mode='lines', name='Fitted Values',
                                 line=dict(color='green', dash='dash')))

    # If forecast intervals are available, add them as shaded areas
    if forecast_int is not None:
        fig.add_trace(go.Scatter(
            x=series_test.index.tolist() + series_test.index[::-1].tolist(),
            y=forecast_int['upper'].tolist() + forecast_int['lower'][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(169, 169, 169, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Confidence Interval',
            showlegend=False
        ))

    # Update layout for titles and labels
    fig.update_layout(
        title=title,  # Use the provided title parameter
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(font=dict(size=16)),
        template='plotly_white',
        width=800,
        height=500
    )

    # Return the figure for saving
    return fig, mae, mape


if __name__ == "__main__":
    print("📋 This module provides plotting functions for hybrid models.")
    print("🚀 Run the main LSTM workflow with: python steps/_05_lstm.py")