"""
ARIMA Plotting and Diagnostics Module

This module provides visualization and diagnostic functions for ARIMA modeling including:
- Residual analysis and plotting
- Forecast visualization with confidence intervals
- Model diagnostic plots
- Performance evaluation visualizations

This module is separate from the core ARIMA modeling to facilitate production deployment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

def analyze_residuals(fitted_model, title: str = "Model Residuals") -> Dict:
    """
    Analyze model residuals for diagnostic purposes.
    
    Args:
        fitted_model: Fitted ARIMA/SARIMA model
        title: Title for plots
        
    Returns:
        Dictionary with residual analysis results
    """
    print(f"\nAnalyzing residuals for {title}")
    
    # Get residuals
    residuals = fitted_model.resid
    
    # Basic statistics
    residual_stats = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'skewness': residuals.skew(),
        'kurtosis': residuals.kurtosis(),
        'jarque_bera_stat': fitted_model.test_normality()[0],
        'jarque_bera_pvalue': fitted_model.test_normality()[1]
    }
    
    print(f"Residual Statistics:")
    print(f"  Mean: {residual_stats['mean']:.4f}")
    print(f"  Std: {residual_stats['std']:.4f}")
    print(f"  Skewness: {residual_stats['skewness']:.4f}")
    print(f"  Kurtosis: {residual_stats['kurtosis']:.4f}")
    print(f"  Jarque-Bera p-value: {residual_stats['jarque_bera_pvalue']:.4f}")
    
    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    print(f"\nLjung-Box test for residual autocorrelation:")
    print(f"  p-value: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
    
    return {
        'residuals': residuals,
        'statistics': residual_stats,
        'ljung_box': lb_test
    }

def plot_residuals_analysis(residuals: pd.Series, title: str = "Model Residuals"):
    """
    Plot comprehensive residual analysis.
    
    Args:
        residuals: Model residuals
        title: Title for the plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals time series
    axes[0, 0].plot(residuals.index, residuals.values)
    axes[0, 0].set_title('Residuals Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals.dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Residuals Histogram')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ACF of residuals
    plot_acf(residuals.dropna(), ax=axes[1, 0], lags=20, alpha=0.05)
    axes[1, 0].set_title('ACF of Residuals')
    
    # PACF of residuals
    plot_pacf(residuals.dropna(), ax=axes[1, 1], lags=20, alpha=0.05)
    axes[1, 1].set_title('PACF of Residuals')
    
    plt.tight_layout()
    plt.suptitle(f'Residual Analysis: {title}', y=1.02)
    plt.show()

def plot_forecast_results(train_data: pd.Series, test_data: pd.Series, 
                         forecast_result: Dict, title: str = "ARIMA Forecast"):
    """
    Plot forecast results with confidence intervals.
    
    Args:
        train_data: Training data
        test_data: Test data (actual values)
        forecast_result: Forecast results dictionary
        title: Title for the plot
    """
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data.values,
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2)
    ))
    
    # Test data (actual values)
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data.values,
        mode='lines+markers',
        name='Actual Values',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast values
    forecast_values = forecast_result['forecast']
    forecast_index = pd.date_range(
        start=test_data.index[0], 
        periods=len(forecast_values), 
        freq='W-SAT'
    )
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Confidence intervals
    conf_int = forecast_result['conf_int']
    alpha = forecast_result['alpha']
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=conf_int.iloc[:, 1],  # Upper bound
        mode='lines',
        name=f'Upper CI ({1-alpha:.0%})',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=conf_int.iloc[:, 0],  # Lower bound
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)',
        name=f'Confidence Interval ({1-alpha:.0%})',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales Volume',
        template='plotly_white',
        width=1000,
        height=500,
        hovermode='x unified'
    )
    
    fig.show()

def plot_model_comparison(comparison_results: Dict, metric: str = 'rmse'):
    """
    Plot model comparison results.
    
    Args:
        comparison_results: Results from compare_arima_models function
        metric: Metric to plot ('rmse', 'mae', 'mape', 'directional_accuracy')
    """
    if not comparison_results['results']:
        print("No results to plot")
        return
    
    # Extract data for plotting
    models = list(comparison_results['results'].keys())
    metrics = [comparison_results['results'][model]['accuracy'][metric] for model in models]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(models)), metrics, alpha=0.7)
    
    # Highlight best model
    best_model = comparison_results.get('best_model')
    if best_model:
        best_idx = models.index(best_model)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.8)
    
    # Customize plot
    ax.set_xlabel('ARIMA Models')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison: {metric.upper()}')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f'ARIMA{model}' for model in models], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics)):
        if metric == 'directional_accuracy':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.1f}%', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_forecast_accuracy(actual: pd.Series, forecast: pd.Series, title: str = "Forecast Accuracy"):
    """
    Plot forecast accuracy analysis.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        title: Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scatter plot of actual vs forecast
    axes[0, 0].scatter(actual, forecast, alpha=0.6)
    axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Forecast Values')
    axes[0, 0].set_title('Actual vs Forecast')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time series comparison
    axes[0, 1].plot(actual.index, actual.values, label='Actual', linewidth=2)
    axes[0, 1].plot(forecast.index, forecast.values, label='Forecast', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Values')
    axes[0, 1].set_title('Time Series Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = actual - forecast
    axes[1, 0].plot(residuals.index, residuals.values)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Forecast Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

def plot_forecast_components(train_data: pd.Series, test_data: pd.Series, 
                           forecast_result: Dict, title: str = "Forecast Components"):
    """
    Plot forecast components with detailed breakdown.
    
    Args:
        train_data: Training data
        test_data: Test data
        forecast_result: Forecast results
        title: Title for the plot
    """
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data.values,
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    # Test data (actual)
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data.values,
        mode='lines+markers',
        name='Actual Values',
        line=dict(color='red', width=2),
        marker=dict(size=5)
    ))
    
    # Forecast
    forecast_values = forecast_result['forecast']
    forecast_index = pd.date_range(
        start=test_data.index[0], 
        periods=len(forecast_values), 
        freq='W-SAT'
    )
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green', width=2, dash='dash'),
        marker=dict(size=5)
    ))
    
    # Confidence intervals
    conf_int = forecast_result['conf_int']
    alpha = forecast_result['alpha']
    
    # Upper bound
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=conf_int.iloc[:, 1],
        mode='lines',
        name=f'Upper CI ({1-alpha:.0%})',
        line=dict(color='lightgray', width=1, dash='dot'),
        showlegend=False
    ))
    
    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=conf_int.iloc[:, 0],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(200,200,200,0.3)',
        name=f'Confidence Interval ({1-alpha:.0%})',
        line=dict(color='lightgray', width=1, dash='dot')
    ))
    
    # Add error bands
    errors = test_data - forecast_values
    error_std = errors.std()
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_values + error_std,
        mode='lines',
        name='+1 Std Error',
        line=dict(color='orange', width=1, dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_values - error_std,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)',
        name='±1 Std Error',
        line=dict(color='orange', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales Volume',
        template='plotly_white',
        width=1200,
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.show()

def create_diagnostic_report(fitted_model, train_data: pd.Series, test_data: pd.Series, 
                           forecast_result: Dict, title: str = "ARIMA Diagnostic Report"):
    """
    Create a comprehensive diagnostic report with all plots.
    
    Args:
        fitted_model: Fitted ARIMA model
        train_data: Training data
        test_data: Test data
        forecast_result: Forecast results
        title: Title for the report
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC REPORT: {title}")
    print(f"{'='*80}")
    
    # 1. Residual analysis
    print("\n1. RESIDUAL ANALYSIS")
    print("-" * 40)
    residual_results = analyze_residuals(fitted_model, title)
    plot_residuals_analysis(residual_results['residuals'], title)
    
    # 2. Forecast visualization
    print("\n2. FORECAST VISUALIZATION")
    print("-" * 40)
    plot_forecast_results(train_data, test_data, forecast_result, title)
    
    # 3. Forecast accuracy
    print("\n3. FORECAST ACCURACY ANALYSIS")
    print("-" * 40)
    plot_forecast_accuracy(test_data, forecast_result['forecast'], f"{title} - Accuracy")
    
    # 4. Detailed forecast components
    print("\n4. DETAILED FORECAST COMPONENTS")
    print("-" * 40)
    plot_forecast_components(train_data, test_data, forecast_result, title)
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC REPORT COMPLETED")
    print(f"{'='*80}")
    
    return {
        'residual_analysis': residual_results,
        'forecast_plot': True,
        'accuracy_plot': True,
        'components_plot': True
    }

if __name__ == "__main__":
    print("ARIMA Plotting and Diagnostics Module")
    print("This module provides visualization and diagnostic functions for ARIMA modeling.")
    print("\nUsage examples:")
    print("  from steps._04_arima_plots import *")
    print("  # Residual analysis:")
    print("  residual_results = analyze_residuals(fitted_model, 'Book Title')")
    print("  plot_residuals_analysis(residual_results['residuals'], 'Book Title')")
    print("  # Forecast visualization:")
    print("  plot_forecast_results(train_data, test_data, forecast_result, 'Book Title')")
    print("  # Complete diagnostic report:")
    print("  report = create_diagnostic_report(fitted_model, train_data, test_data, forecast_result, 'Book Title')") 