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
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.graphics.tsaplots as smgraphics
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

def plot_prediction(series_train: pd.Series, series_test: pd.Series, forecast: pd.Series, 
                   forecast_int: Optional[pd.DataFrame] = None, fitted_values: Optional[pd.Series] = None, 
                   title: str = "Forecast Plot") -> Tuple[go.Figure, float, float]:
    """
    Create a comprehensive forecast plot with training, test, and forecast data.
    
    Args:
        series_train: Training data series
        series_test: Test data series (actual values)
        forecast: Forecast values
        forecast_int: Confidence intervals DataFrame with 'lower' and 'upper' columns
        fitted_values: Fitted values from the model
        title: Title for the plot
        
    Returns:
        Tuple of (figure, mae, mape)
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
        # Handle different confidence interval formats
        if isinstance(forecast_int, pd.DataFrame):
            if 'upper' in forecast_int.columns and 'lower' in forecast_int.columns:
                upper_col = 'upper'
                lower_col = 'lower'
            elif 'mean_ci_upper' in forecast_int.columns and 'mean_ci_lower' in forecast_int.columns:
                upper_col = 'mean_ci_upper'
                lower_col = 'mean_ci_lower'
            else:
                # Use the first two columns as upper and lower bounds
                cols = forecast_int.columns
                if len(cols) >= 2:
                    upper_col = cols[1]
                    lower_col = cols[0]
                else:
                    upper_col = lower_col = cols[0]
            
            fig.add_trace(go.Scatter(
                x=series_test.index.tolist() + series_test.index[::-1].tolist(),
                y=forecast_int[upper_col].tolist() + forecast_int[lower_col][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(169, 169, 169, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Confidence Interval',
                showlegend=False
            ))

    # Update layout for titles and labels
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(font=dict(size=16)),
        template='plotly_white',
        width=800,
        height=500
    )

    # Return the figure for saving
    return fig, mae, mape

def plot_residuals_with_tests(residuals: pd.Series, title: str = "Model Residuals"):
    """
    Plot comprehensive residual analysis with statistical tests.
    
    Args:
        residuals: Model residuals
        title: Title for the plots
    """
    plt.figure(figsize=(12, 8))

    # Plot residuals over time
    plt.subplot(3, 1, 1)
    plt.plot(residuals, label='Residuals', marker='o')
    plt.title(f'{title} - Residuals Over Time')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Week')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.legend()

    # Plot histogram of residuals
    plt.subplot(3, 1, 2)
    sns.histplot(residuals, kde=True, bins=20, color='blue')
    plt.title(f'{title} - Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    # Plot ACF of residuals
    plt.subplot(3, 1, 3)
    plot_acf(residuals, lags=20, ax=plt.gca())
    plt.title(f'{title} - Residuals ACF')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')

    # Ljung-Box test
    ljung_test = acorr_ljungbox(residuals, lags=[1], return_df=True)
    print(f"Ljung-Box Test: {ljung_test}")

    # Jarque-Bera test for normality
    jb_test = jarque_bera(residuals)  # Get full return value
    jb_stat = jb_test[0]  # Extract the test statistic
    jb_p = jb_test[1]     # Extract the p-value
    print(f"Jarque-Bera Test: Stat={jb_stat}, p-value={jb_p}")

    plt.tight_layout()
    plt.show()

def plot_qq_residuals(residuals: pd.Series, title: str = "Q-Q Plot Analysis"):
    """
    Create Q-Q plots for residual normality analysis.
    
    Args:
        residuals: Model residuals
        title: Title for the plots
    """
    # Perform Jarque-Bera Test
    jb_stats = jarque_bera(residuals)
    print(f"Jarque-Bera test statistic: {jb_stats[0]}, p-value: {jb_stats[1]}")

    loc = None
    df = None

    # If residuals are non-normal, fit a Student-t distribution
    if jb_stats[1] < 0.05:
        df, loc, scale = t.fit(residuals)
        print(f"Fitted Student-t parameters: df={df}, loc={loc}, scale={scale}")

    # Q-Q Plot of residuals against standard normal distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Residuals vs. Standard Normal')

    # Q-Q Plot of residuals against fitted Student-t distribution
    plt.subplot(1, 2, 2)
    if df is not None:
        stats.probplot(residuals, dist="t", sparams=(df, loc, scale), plot=plt)
        plt.title(f'Q-Q Plot: Residuals vs. Fitted Student-t (df={df:.2f})')
    else:
        plt.text(0.5, 0.5, 'Student-t fitting not needed\n(residuals are normal)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Q-Q Plot: Residuals vs. Fitted Student-t')

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

def plot_forecast_with_components(train_data: pd.Series, test_data: pd.Series, 
                                forecast: pd.Series, forecast_int: pd.DataFrame,
                                fitted_values: pd.Series, title: str = "Forecast with Components"):
    """
    Create a detailed forecast plot with all components.
    
    Args:
        train_data: Training data
        test_data: Test data
        forecast: Forecast values
        forecast_int: Confidence intervals
        fitted_values: Fitted values
        title: Title for the plot
    """
    # Create the plot figure
    fig = go.Figure()

    # Add the observed data trace - training
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data, 
                            mode='lines', name='Observed - Training'))

    # Add the observed data trace - test
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data, 
                            mode='lines', name='Observed - Test'))

    # Add the fitted values trace
    fig.add_trace(go.Scatter(x=fitted_values.index, y=fitted_values, 
                            mode='lines', name='Fitted Values'))

    # Add the forecast trace (mean forecast)
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, 
                            mode='lines', name='Forecast'))

    # Add the forecast confidence intervals (lower bound)
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast_int['mean_ci_lower'], 
                            mode='lines', name='Confidence Interval Lower', 
                            line=dict(color='darkblue')))

    # Add the forecast confidence intervals (upper bound)
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast_int['mean_ci_upper'], 
                            mode='lines', name='Confidence Interval Upper', 
                            line=dict(color='darkblue')))

    # Add line connecting fitted values to the first forecast point
    fig.add_trace(go.Scatter(x=[fitted_values.index[-1], forecast.index[0]], 
                            y=[fitted_values.iloc[-1], forecast.iloc[0]], 
                            mode='lines', name='Connection Line'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        width=1000,
        height=600,
        hovermode='x unified'
    )

    return fig

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
        'kurtosis': residuals.kurtosis()
    }
    
    # Test normality (with proper method parameter)
    try:
        normality_test = fitted_model.test_normality(method='jarquebera')
        residual_stats['jarque_bera_stat'] = normality_test[0]
        residual_stats['jarque_bera_pvalue'] = normality_test[1]
    except Exception as e:
        print(f"Warning: Normality test failed: {e}")
        residual_stats['jarque_bera_stat'] = None
        residual_stats['jarque_bera_pvalue'] = None
    
    print(f"Residual Statistics:")
    print(f"  Mean: {residual_stats['mean']:.4f}")
    print(f"  Std: {residual_stats['std']:.4f}")
    print(f"  Skewness: {residual_stats['skewness']:.4f}")
    print(f"  Kurtosis: {residual_stats['kurtosis']:.4f}")
    if residual_stats['jarque_bera_pvalue'] is not None:
        print(f"  Jarque-Bera p-value: {residual_stats['jarque_bera_pvalue']:.4f}")
    else:
        print(f"  Jarque-Bera p-value: Not available")
    
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

# ============================================================================
# LSTM-SPECIFIC RESIDUAL ANALYSIS FUNCTIONS
# ============================================================================

def analyze_residuals_for_lstm(residuals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze residuals for LSTM model preparation.
    """
    print("\n=== Residuals Analysis for LSTM ===")
    
    # Basic statistics
    stats_dict = {
        'count': len(residuals_df),
        'mean': float(residuals_df['residuals'].mean()),
        'std': float(residuals_df['residuals'].std()),
        'min': float(residuals_df['residuals'].min()),
        'max': float(residuals_df['residuals'].max()),
        'skewness': float(residuals_df['residuals'].skew()),
        'kurtosis': float(residuals_df['residuals'].kurtosis())
    }
    
    print(f"Residuals Statistics:")
    for key, value in stats_dict.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    # Check for missing values
    missing_count = residuals_df['residuals'].isna().sum()
    print(f"  Missing values: {missing_count}")
    
    # Check for outliers (using IQR method)
    Q1 = residuals_df['residuals'].quantile(0.25)
    Q3 = residuals_df['residuals'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = residuals_df[
        (residuals_df['residuals'] < Q1 - 1.5 * IQR) | 
        (residuals_df['residuals'] > Q3 + 1.5 * IQR)
    ]
    print(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(residuals_df)*100:.2f}%)")
    
    return {
        'residuals_df': residuals_df,
        'statistics': stats_dict,
        'outliers': outliers,
        'analysis_summary': {
            'total_points': len(residuals_df),
            'outlier_percentage': len(outliers)/len(residuals_df)*100,
            'data_quality': 'good' if missing_count == 0 else 'needs_attention'
        }
    }

def plot_residuals_for_lstm(residuals_df: pd.DataFrame, save_plots: bool = True):
    """
    Create comprehensive plots of residuals for LSTM analysis.
    """
    print("\n=== Creating Residuals Plots ===")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ARIMA Residuals Analysis for LSTM Model', fontsize=16, fontweight='bold')
    
    # 1. Time series plot
    axes[0, 0].plot(residuals_df['date'], residuals_df['residuals'], linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Residuals Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Histogram with normal distribution overlay
    axes[0, 1].hist(residuals_df['residuals'], bins=30, alpha=0.7, density=True, edgecolor='black')
    # Overlay normal distribution
    x = np.linspace(residuals_df['residuals'].min(), residuals_df['residuals'].max(), 100)
    mu, sigma = residuals_df['residuals'].mean(), residuals_df['residuals'].std()
    normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    axes[0, 1].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    stats.probplot(residuals_df['residuals'].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    axes[1, 1].boxplot(residuals_df['residuals'], vert=True)
    axes[1, 1].set_title('Residuals Box Plot')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add simple test plots, temporarily
    print("\n=== Creating Simple Test Plots ===")
    
    # Create a new figure for the simple plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Simple Residuals Test Plots', fontsize=14, fontweight='bold')
    
    # Plot the histogram of the residuals
    ax1.hist(residuals_df['residuals'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Histogram of Residuals')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot the ACF of the residuals
    smgraphics.plot_acf(residuals_df['residuals'].dropna(), ax=ax2, lags=40)
    ax2.set_title('ACF of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if save_plots:
        # Create outputs directory if it doesn't exist
        outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        os.makedirs(f"{outputs_dir}/plots/static", exist_ok=True)
        
        plot_path = os.path.join(outputs_dir, 'plots', 'static', 'arima_residuals_for_lstm.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plots saved to: {plot_path}")
        
        # Save the simple test plots as well, temporarily
        simple_plot_path = os.path.join(outputs_dir, 'plots', 'static', 'simple_residuals_test.png')
        fig2.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
        print(f"Simple test plots saved to: {simple_plot_path}")
    
    return fig

def print_residuals_sample(residuals_df: pd.DataFrame, n_samples: int = 10):
    """
    Print a sample of the residuals data to verify it loaded correctly.
    """
    print(f"\n=== Residuals Sample (first {n_samples} rows) ===")
    print(residuals_df.head(n_samples).to_string(index=False))
    
    print(f"\n=== Residuals Sample (last {n_samples} rows) ===")
    print(residuals_df.tail(n_samples).to_string(index=False))
    
    print(f"\n=== Residuals Summary ===")
    print(f"Total rows: {len(residuals_df)}")
    print(f"Date range: {residuals_df['date'].min()} to {residuals_df['date'].max()}")
    print(f"Model signature: {residuals_df['model_signature'].iloc[0]}")
    print(f"Residuals range: {residuals_df['residuals'].min():.2f} to {residuals_df['residuals'].max():.2f}")

def run_complete_residuals_analysis(residuals_df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
    """
    Convenient wrapper function that runs both analysis and plotting.
    """
    print("\n=== Running Complete Residuals Analysis ===")
    
    # Run analysis
    analysis_results = analyze_residuals_for_lstm(residuals_df)
    
    # Create plots
    plot_figure = plot_residuals_for_lstm(residuals_df, save_plots=save_plots)
    
    # Add plot figure to results
    analysis_results['plot_figure'] = plot_figure
    
    return analysis_results

def create_sample_residuals_for_testing() -> pd.DataFrame:
    """
    Create sample residuals data for standalone testing of analysis functions.
    """
    print("Creating sample residuals data for testing...")
    
    # Create realistic sample data
    sample_dates = pd.date_range(start='2020-01-01', periods=200, freq='W')
    sample_residuals = np.random.normal(0, 50, 200)  # Mock residuals
    
    residuals_df = pd.DataFrame({
        'date': sample_dates,
        'residuals': sample_residuals,
        'model_signature': 'SARIMAX_(2,1,3)_(1,1,3,52)_TEST'
    })
    
    return residuals_df

def load_real_residuals_from_csv(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load real residuals from CSV file saved by the ARIMA pipeline.
    """
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
    
    residuals_df = pd.read_csv(residuals_csv_path)
    
    # Convert date column to datetime if it's not already
    if 'date' in residuals_df.columns:
        residuals_df['date'] = pd.to_datetime(residuals_df['date'])
    
    return residuals_df

# ============================================================================
# DEMO AND TESTING FUNCTIONS
# ============================================================================

def create_sample_data_for_demo():
    """Create realistic sample data for demonstration."""
    # Create dates
    dates = pd.date_range('2020-01-01', periods=150, freq='W-SAT')
    
    # Create realistic book sales data with trend and seasonality
    np.random.seed(42)  # For reproducible results
    
    # Base trend
    trend = np.linspace(100, 200, 150)
    
    # Seasonal pattern (yearly cycle)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(150) / 52)
    
    # Add some noise
    noise = np.random.normal(0, 20, 150)
    
    # Combine components
    train_data = trend[:100] + seasonal[:100] + noise[:100]
    test_data = trend[100:] + seasonal[100:] + noise[100:]
    
    # Create series
    train_series = pd.Series(train_data, index=dates[:100])
    test_series = pd.Series(test_data, index=dates[100:])
    
    # Create forecast (with some error)
    forecast_data = test_data + np.random.normal(0, 10, 50)
    forecast_series = pd.Series(forecast_data, index=dates[100:])
    
    # Create confidence intervals
    forecast_int = pd.DataFrame({
        'lower': forecast_data - 30,
        'upper': forecast_data + 30
    }, index=dates[100:])
    
    # Create residuals (mock)
    residuals = np.random.normal(0, 15, 100)
    residuals_series = pd.Series(residuals, index=dates[:100])
    
    # Create fitted values
    fitted_values = train_data + np.random.normal(0, 8, 100)
    fitted_series = pd.Series(fitted_values, index=dates[:100])
    
    return {
        'train_data': train_series,
        'test_data': test_series,
        'forecast': forecast_series,
        'forecast_int': forecast_int,
        'residuals': residuals_series,
        'fitted_values': fitted_series
    }

def create_mock_forecast_result(test_data, forecast, forecast_int):
    """Create a mock forecast result dictionary."""
    return {
        'forecast': forecast.values,
        'conf_int': forecast_int,
        'alpha': 0.05
    }

def run_plotting_demo():
    """Run a comprehensive demonstration of all plotting functions."""
    print("=" * 80)
    print("ARIMA PLOTTING MODULE DEMONSTRATION")
    print("=" * 80)
    print("Generating sample data and creating plots...")
    print("\nNote: This uses synthetic data for demonstration purposes.")
    print("In real usage, you would provide actual ARIMA model results.\n")
    
    # Create sample data
    data = create_sample_data_for_demo()
    
    print("1. BASIC FORECAST PLOT")
    print("-" * 40)
    fig, mae, mape = plot_prediction(
        series_train=data['train_data'],
        series_test=data['test_data'],
        forecast=data['forecast'],
        forecast_int=data['forecast_int'],
        fitted_values=data['fitted_values'],
        title="Sample ARIMA Forecast - The Alchemist"
    )
    
    # Save the plot
    import os
    os.makedirs('outputs', exist_ok=True)
    fig.write_html('outputs/sample_arima_forecast.html')
    print(f"✅ Forecast plot saved to: outputs/sample_arima_forecast.html")
    print(f"   MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    print("\n2. RESIDUALS ANALYSIS")
    print("-" * 40)
    plot_residuals_with_tests(data['residuals'], "Sample Model Residuals")
    
    print("\n3. Q-Q PLOTS FOR NORMALITY")
    print("-" * 40)
    plot_qq_residuals(data['residuals'], "Sample Residual Q-Q Analysis")
    
    print("\n4. COMPREHENSIVE RESIDUAL PLOTS")
    print("-" * 40)
    plot_residuals_analysis(data['residuals'], "Sample Model")
    
    print("\n5. FORECAST ACCURACY ANALYSIS")
    print("-" * 40)
    plot_forecast_accuracy(data['test_data'], data['forecast'], "Sample Forecast Accuracy")
    
    print("\n6. FORECAST WITH COMPONENTS")
    print("-" * 40)
    forecast_result = create_mock_forecast_result(data['test_data'], data['forecast'], data['forecast_int'])
    plot_forecast_components(data['train_data'], data['test_data'], forecast_result, "Sample Forecast Components")
    
    print("\n7. DETAILED FORECAST RESULTS")
    print("-" * 40)
    plot_forecast_results(data['train_data'], data['test_data'], forecast_result, "Sample ARIMA Results")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("✅ All plotting functions demonstrated successfully!")
    print("📁 Interactive forecast plot saved to: outputs/sample_arima_forecast.html")
    print("\nThis module provides comprehensive ARIMA diagnostic plotting.")
    print("For real usage, import the functions and provide actual model results.")
    
    print("\nUsage examples:")
    print("  from scripts.arima_plots import *")
    print("  # Basic forecast:")
    print("  fig, mae, mape = plot_prediction(train_data, test_data, forecast)")
    print("  # Residual analysis:")
    print("  plot_residuals_analysis(model.resid, 'My Model')")
    print("  # Complete diagnostics:")
    print("  create_diagnostic_report(fitted_model, train_data, test_data, forecast_result)")

if __name__ == "__main__":
    try:
        run_plotting_demo()
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        print("Please ensure you have all required dependencies installed:")
        print("  - pandas, numpy, matplotlib, plotly, seaborn, scipy, sklearn, statsmodels") 