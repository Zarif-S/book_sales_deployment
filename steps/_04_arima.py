"""
ARIMA Modeling Module for Book Sales Forecasting

This module implements core ARIMA modeling for book sales data including:
- Auto ARIMA model selection
- Model fitting and validation
- Residual analysis (statistical only)
- Forecasting with confidence intervals
- Model comparison and evaluation

This module is production-ready and focuses on core modeling functionality.
For visualization and plotting, use the separate _04_arima_plots.py module.
"""

import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Global list to store model results
model_results_list = []

def prepare_data_for_arima(data: pd.DataFrame, isbn: Union[str, int], 
                          forecast_horizon: int = 32) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare data for ARIMA modeling by splitting into training and test sets.
    
    Args:
        data: DataFrame with time series data (must have 'ISBN' and 'Volume' columns)
        isbn: ISBN of the book to model
        forecast_horizon: Number of weeks to use for forecasting (default: 32)
        
    Returns:
        Tuple of (training_data, test_data)
    """
    if 'ISBN' not in data.columns or 'Volume' not in data.columns:
        raise ValueError("Data must contain 'ISBN' and 'Volume' columns")
    
    # Filter data for specific book
    book_data = data[data['ISBN'] == isbn].copy()
    
    if book_data.empty:
        raise ValueError(f"No data found for ISBN: {isbn}")
    
    # Ensure index is datetime
    if not isinstance(book_data.index, pd.DatetimeIndex):
        if 'End Date' in book_data.columns:
            book_data['End Date'] = pd.to_datetime(book_data['End Date'])
            book_data.set_index('End Date', inplace=True)
        else:
            raise ValueError("Data must have datetime index or 'End Date' column")
    
    # Sort by date and extract volume
    book_data.sort_index(inplace=True)
    ts_data = book_data['Volume'].dropna()
    
    # Split into training and test sets
    if len(ts_data) <= forecast_horizon:
        raise ValueError(f"Data length ({len(ts_data)}) must be greater than forecast horizon ({forecast_horizon})")
    
    train_data = ts_data[:-forecast_horizon]
    test_data = ts_data[-forecast_horizon:]
    
    return train_data, test_data

def prepare_data_after_2012(book_data: pd.DataFrame, column_name: str, split_size: int = 32) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare training and testing data after 2012-01-01 based on a given split size.

    Args:
        book_data: DataFrame containing the book data with a time series index
        column_name: Column to split into train and test data
        split_size: Number of entries (weeks or months) to include in the test set

    Returns:
        Tuple of (training_data, test_data)
    """
    # Filter data for dates after 2012-01-01 inclusive
    data_after_2012 = book_data[book_data.index >= '2012-01-01']

    # Ensure there is enough data for splitting
    if len(data_after_2012) < split_size:
        raise ValueError(f"Not enough data available for the test set (at least {split_size} entries required).")

    # Split into train and test data
    train_data = data_after_2012[column_name].iloc[:-split_size]  # All data except the last split_size entries
    test_data = data_after_2012[column_name].iloc[-split_size:]   # Last split_size entries of data

    return train_data, test_data

def run_auto_arima(train_data: pd.Series, 
                   seasonal: bool = True,
                   m: int = 52,
                   max_p: int = 2,  # Reduced from 5 to prevent crashes
                   max_d: int = 0,  # Reduced from 2 to prevent crashes
                   max_q: int = 2,  # Reduced from 5 to prevent crashes
                   max_P: int = 2,
                   max_D: int = 0,  # Reduced from 1 to prevent crashes
                   max_Q: int = 2,  # Reduced from 5 to prevent crashes
                   stepwise: bool = True,
                   suppress_warnings: bool = True,
                   error_action: str = 'ignore',
                   trace: bool = False,
                   information_criterion: str = 'aic',
                   out_of_sample_size: int = 0) -> Dict:
    """
    Run Auto ARIMA to find the best model parameters with optimized settings.
    
    Args:
        train_data: Training time series data
        seasonal: Whether to use seasonal ARIMA (SARIMA)
        m: Seasonal period (52 for weekly data)
        max_p, max_d, max_q: Maximum orders for non-seasonal components (reduced to prevent crashes)
        max_P, max_D, max_Q: Maximum orders for seasonal components (reduced to prevent crashes)
        stepwise: Whether to use stepwise search
        suppress_warnings: Whether to suppress warnings
        error_action: How to handle errors
        trace: Whether to print trace information
        information_criterion: 'aic' or 'bic' for model selection
        out_of_sample_size: Number of observations to reserve for validation
        
    Returns:
        Dictionary with Auto ARIMA results
    """
    print(f"Running Auto ARIMA with parameters:")
    print(f"  Seasonal: {seasonal}")
    print(f"  Seasonal period (m): {m}")
    print(f"  Max orders - p:{max_p}, d:{max_d}, q:{max_q}")
    if seasonal:
        print(f"  Max seasonal orders - P:{max_P}, D:{max_D}, Q:{max_Q}")
    print(f"  Information criterion: {information_criterion}")
    print(f"  Out-of-sample size: {out_of_sample_size}")
    
    try:
        if seasonal:
            auto_model = auto_arima(
                train_data,
                seasonal=True,
                m=m,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                max_P=max_P,
                max_D=max_D,
                max_Q=max_Q,
                stepwise=stepwise,
                suppress_warnings=suppress_warnings,
                error_action=error_action,
                trace=trace,
                random_state=42,
                information_criterion=information_criterion,
                out_of_sample_size=out_of_sample_size,
                scoring='mse',
                stationary=True,  # Set to True as data is likely stationary
                test='kpss',  # KPSS test for stationarity
                seasonal_test='ocsb',  # OCSB test for seasonality
                maxiter=30,  # Limit iterations to prevent long runs
                n_jobs=1  # Single job to prevent memory issues
            )
        else:
            auto_model = auto_arima(
                train_data,
                seasonal=False,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                stepwise=stepwise,
                suppress_warnings=suppress_warnings,
                error_action=error_action,
                trace=trace,
                random_state=42,
                information_criterion=information_criterion,
                out_of_sample_size=out_of_sample_size,
                scoring='mse',
                stationary=True,
                test='kpss',
                maxiter=30,
                n_jobs=1
            )
        
        # Extract model information
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order if seasonal else None
        aic = auto_model.aic()
        bic = auto_model.bic()
        
        print(f"\nAuto ARIMA Results:")
        print(f"  Best order: {order}")
        if seasonal:
            print(f"  Best seasonal order: {seasonal_order}")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        
        return {
            'model': auto_model,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': aic,
            'bic': bic,
            'seasonal': seasonal
        }
        
    except Exception as e:
        print(f"Auto ARIMA failed: {e}")
        return None

def fit_arima_model(train_data: pd.Series, order: Tuple[int, int, int], 
                   seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Dict:
    """
    Fit an ARIMA/SARIMA model with specified parameters.
    
    Args:
        train_data: Training time series data
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, m) for SARIMA
        
    Returns:
        Dictionary with fitted model and results
    """
    print(f"Fitting ARIMA model with order: {order}")
    if seasonal_order:
        print(f"Seasonal order: {seasonal_order}")
    
    try:
        if seasonal_order:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(train_data, order=order)
        
        fitted_model = model.fit()
        
        print(f"Model fitting completed successfully")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        return {
            'fitted_model': fitted_model,
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'summary': fitted_model.summary()
        }
        
    except Exception as e:
        print(f"Model fitting failed: {e}")
        return None

def analyze_residuals(fitted_model, title: str = "Model Residuals") -> Dict:
    """
    Analyze model residuals for diagnostic purposes with comprehensive tests.
    
    Args:
        fitted_model: Fitted ARIMA/SARIMA model
        title: Title for analysis
        
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
    
    # Additional normality test using scipy
    jb_test = jarque_bera(residuals.dropna())
    residual_stats['jarque_bera_scipy_stat'] = jb_test[0]
    residual_stats['jarque_bera_scipy_pvalue'] = jb_test[1]
    
    print(f"Residual Statistics:")
    print(f"  Mean: {residual_stats['mean']:.4f}")
    print(f"  Std: {residual_stats['std']:.4f}")
    print(f"  Skewness: {residual_stats['skewness']:.4f}")
    print(f"  Kurtosis: {residual_stats['kurtosis']:.4f}")
    if residual_stats['jarque_bera_pvalue'] is not None:
        print(f"  Jarque-Bera p-value (statsmodels): {residual_stats['jarque_bera_pvalue']:.4f}")
    print(f"  Jarque-Bera p-value (scipy): {residual_stats['jarque_bera_scipy_pvalue']:.4f}")
    
    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    print(f"\nLjung-Box test for residual autocorrelation:")
    print(f"  p-value: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
    
    # Fit Student-t distribution if residuals are non-normal
    if residual_stats['jarque_bera_scipy_pvalue'] < 0.05:
        try:
            df, loc, scale = t.fit(residuals.dropna())
            residual_stats['student_t_df'] = df
            residual_stats['student_t_loc'] = loc
            residual_stats['student_t_scale'] = scale
            print(f"  Fitted Student-t parameters: df={df:.2f}, loc={loc:.2f}, scale={scale:.2f}")
        except Exception as e:
            print(f"  Student-t fitting failed: {e}")
    
    return {
        'residuals': residuals,
        'statistics': residual_stats,
        'ljung_box': lb_test
    }

def forecast_with_arima(fitted_model, steps: int, alpha: float = 0.05) -> Dict:
    """
    Generate forecasts using fitted ARIMA/SARIMA model.
    
    Args:
        fitted_model: Fitted ARIMA/SARIMA model
        steps: Number of steps to forecast
        alpha: Significance level for confidence intervals
        
    Returns:
        Dictionary with forecast results
    """
    print(f"Generating {steps}-step forecast...")
    
    try:
        # Generate forecast
        forecast_result = fitted_model.forecast(steps=steps, alpha=alpha)
        
        # Handle different forecast result structures
        if hasattr(forecast_result, 'predicted_mean'):
            # Standard statsmodels forecast result
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
        elif isinstance(forecast_result, pd.Series):
            # Direct series result
            forecast_values = forecast_result
            # Try to get confidence intervals if available
            try:
                conf_int = fitted_model.get_forecast(steps=steps, alpha=alpha).conf_int()
            except:
                # Create simple confidence intervals based on model residuals
                residuals_std = fitted_model.resid.std()
                conf_int = pd.DataFrame({
                    'lower': forecast_values - 1.96 * residuals_std,
                    'upper': forecast_values + 1.96 * residuals_std
                })
        else:
            # Fallback: try to extract values directly
            forecast_values = forecast_result
            conf_int = None
        
        print(f"Forecast completed successfully")
        print(f"Forecast range: {forecast_values.min():.2f} to {forecast_values.max():.2f}")
        
        return {
            'forecast': forecast_values,
            'conf_int': conf_int,
            'steps': steps,
            'alpha': alpha
        }
        
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return None

def evaluate_forecast_accuracy(actual: pd.Series, forecast: pd.Series) -> Dict:
    """
    Evaluate forecast accuracy using multiple metrics.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast series must have the same length")
    
    # Calculate errors
    errors = actual - forecast
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Calculate metrics
    mae = np.mean(abs_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(errors / actual)) * 100
    
    # Calculate directional accuracy
    actual_direction = np.diff(actual) > 0
    forecast_direction = np.diff(forecast) > 0
    directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'mean_actual': actual.mean(),
        'mean_forecast': forecast.mean(),
        'std_actual': actual.std(),
        'std_forecast': forecast.std()
    }
    
    print(f"Forecast Accuracy Metrics:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"  Mean Actual: {metrics['mean_actual']:.2f}")
    print(f"  Mean Forecast: {metrics['mean_forecast']:.2f}")
    
    return metrics

def arima_predictions(df: pd.DataFrame, column: str, forecast_steps: int) -> Tuple[pd.DataFrame, object, pd.Series]:
    """
    Alternative ARIMA implementation with automatic parameter selection.
    
    Args:
        df: DataFrame containing the time series data
        column: Column name to model
        forecast_steps: Number of steps to forecast
        
    Returns:
        Tuple of (forecast_summary, model_summary, predictions)
    """
    # MA Lags
    acf_coef = acf(df[column], alpha=0.05)
    sig_acf = []
    for i in range(1, len(acf_coef[0])):
        if acf_coef[0][i] > (acf_coef[1][i][1] - acf_coef[0][i]):
            sig_acf.append(i)
        elif acf_coef[0][i] < (acf_coef[1][i][0] - acf_coef[0][i]):
            sig_acf.append(i)

    # AR Lags
    pacf_coef = pacf(df[column], alpha=0.05)
    sig_pacf = []
    for i in range(1, len(pacf_coef[0])):
        if pacf_coef[0][i] > (pacf_coef[1][i][1] - pacf_coef[0][i]):
            sig_pacf.append(i)
        elif pacf_coef[0][i] < (pacf_coef[1][i][0] - pacf_coef[0][i]):
            sig_pacf.append(i)

    # Order of integration (difference)
    adf = adfuller(x=df[column], autolag='BIC')
    order = 0

    if adf[0] < adf[4]['5%']:
        order = 0  # d
    else:
        order = 1  # d

    # Trend indication
    if order == 1:
        trend = 't'
    else:
        trend = 'c'

    # ARIMA(p,d,q) Model
    model = ARIMA(endog=df[column],
                  order=(sig_pacf, order, sig_acf),
                  trend=trend).fit()

    forecast = model.get_forecast(forecast_steps)
    predict = model.predict()

    return forecast.summary_frame(), model.summary(), predict

def save_and_download(data, file_path: str, save_format: str = 'pickle'):
    """
    Save and download data (models, forecasts, etc.).
    Supports 'pickle' or 'csv' formats.
    """
    if save_format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif save_format == 'csv':
        data.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")

def log_model_results(model_number: int, model_name: str, model_results: Dict, 
                     mae: float, mape: float, book: str, dataset: str):
    """
    Log model results into a dictionary and store them in a list.
    """
    # Ensure necessary metrics are provided
    if mae is None or mape is None:
        raise ValueError(f"Missing MAE or MAPE for model {model_name}")

    # Create a dictionary to store the model information
    model_info = {
        'Model Number': model_number,
        'Book': book,
        'Dataset': dataset,
        'Model Name': model_name,
        'Model Config': model_results,
        'MAE': mae,
        'MAPE': mape
    }
    model_results_list.append(model_info)

def run_complete_arima_analysis(data: pd.DataFrame, isbn: Union[str, int], 
                               forecast_horizon: int = 32,
                               use_auto_arima: bool = True,
                               seasonal: bool = True,
                               title: Optional[str] = None) -> Dict:
    """
    Run complete ARIMA analysis for a single book.
    
    Args:
        data: DataFrame with time series data
        isbn: ISBN of the book to analyze
        forecast_horizon: Number of weeks to forecast
        use_auto_arima: Whether to use Auto ARIMA for parameter selection
        seasonal: Whether to use seasonal ARIMA
        title: Title for the book (optional)
        
    Returns:
        Dictionary with complete analysis results
    """
    if title is None:
        title = f"Book {isbn}"
    
    print(f"\n{'='*80}")
    print(f"ARIMA ANALYSIS FOR {title} ({isbn})")
    print(f"{'='*80}")
    
    # Prepare data
    train_data, test_data = prepare_data_for_arima(data, isbn, forecast_horizon)
    
    print(f"Data prepared:")
    print(f"  Training data: {len(train_data)} observations")
    print(f"  Test data: {len(test_data)} observations")
    print(f"  Date range: {train_data.index.min()} to {test_data.index.max()}")
    
    results = {
        'isbn': isbn,
        'title': title,
        'train_data': train_data,
        'test_data': test_data,
        'forecast_horizon': forecast_horizon
    }
    
    # Auto ARIMA parameter selection
    if use_auto_arima:
        print(f"\n1. AUTO ARIMA PARAMETER SELECTION")
        print("-" * 40)
        auto_results = run_auto_arima(train_data, seasonal=seasonal)
        if auto_results:
            results['auto_arima'] = auto_results
            order = auto_results['order']
            seasonal_order = auto_results['seasonal_order']
        else:
            print("Auto ARIMA failed, using default parameters")
            order = (1, 0, 1)
            seasonal_order = None
    else:
        # Use default parameters
        order = (1, 0, 1)
        seasonal_order = None
    
    # Fit model
    print(f"\n2. MODEL FITTING")
    print("-" * 40)
    model_results = fit_arima_model(train_data, order, seasonal_order)
    if model_results:
        results['model'] = model_results
        fitted_model = model_results['fitted_model']
    else:
        print("Model fitting failed")
        return results
    
    # Residual analysis
    print(f"\n3. RESIDUAL ANALYSIS")
    print("-" * 40)
    residual_results = analyze_residuals(fitted_model, title)
    results['residuals'] = residual_results
    
    # Generate forecast
    print(f"\n4. FORECASTING")
    print("-" * 40)
    forecast_results = forecast_with_arima(fitted_model, forecast_horizon)
    if forecast_results:
        results['forecast'] = forecast_results
        
        # Evaluate accuracy
        print(f"\n5. FORECAST ACCURACY EVALUATION")
        print("-" * 40)
        accuracy_metrics = evaluate_forecast_accuracy(test_data, forecast_results['forecast'])
        results['accuracy'] = accuracy_metrics
    
    print(f"\n{'='*80}")
    print(f"ARIMA ANALYSIS COMPLETED FOR {title}")
    print(f"{'='*80}")
    
    return results

def compare_arima_models(data: pd.DataFrame, isbn: Union[str, int], 
                        models_to_test: List[Tuple] = None,
                        forecast_horizon: int = 32) -> Dict:
    """
    Compare multiple ARIMA models for a single book.
    
    Args:
        data: DataFrame with time series data
        isbn: ISBN of the book to analyze
        models_to_test: List of (p, d, q) tuples to test
        forecast_horizon: Number of weeks to forecast
        
    Returns:
        Dictionary with comparison results
    """
    if models_to_test is None:
        models_to_test = [
            (1, 0, 1),  # ARIMA(1,0,1)
            (1, 0, 0),  # ARIMA(1,0,0) - AR(1)
            (0, 0, 1),  # ARIMA(0,0,1) - MA(1)
            (2, 0, 2),  # ARIMA(2,0,2)
            (1, 1, 1),  # ARIMA(1,1,1) - with differencing
            (0, 1, 1),  # ARIMA(0,1,1) - simple differenced MA
        ]
    
    print(f"\n{'='*80}")
    print(f"ARIMA MODEL COMPARISON FOR BOOK {isbn}")
    print(f"{'='*80}")
    
    # Prepare data
    train_data, test_data = prepare_data_for_arima(data, isbn, forecast_horizon)
    
    comparison_results = {
        'isbn': isbn,
        'models_tested': models_to_test,
        'results': {}
    }
    
    for i, order in enumerate(models_to_test):
        print(f"\nTesting model {i+1}/{len(models_to_test)}: ARIMA{order}")
        print("-" * 50)
        
        try:
            # Fit model
            model_results = fit_arima_model(train_data, order)
            if model_results is None:
                continue
            
            # Generate forecast
            forecast_results = forecast_with_arima(model_results['fitted_model'], forecast_horizon)
            if forecast_results is None:
                continue
            
            # Evaluate accuracy
            accuracy_metrics = evaluate_forecast_accuracy(test_data, forecast_results['forecast'])
            
            # Store results
            comparison_results['results'][order] = {
                'model': model_results,
                'forecast': forecast_results,
                'accuracy': accuracy_metrics
            }
            
        except Exception as e:
            print(f"Model ARIMA{order} failed: {e}")
            continue
    
    # Find best model
    if comparison_results['results']:
        best_model = min(comparison_results['results'].keys(), 
                        key=lambda x: comparison_results['results'][x]['accuracy']['rmse'])
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: ARIMA{best_model}")
        print(f"{'='*80}")
        print(f"RMSE: {comparison_results['results'][best_model]['accuracy']['rmse']:.2f}")
        print(f"MAPE: {comparison_results['results'][best_model]['accuracy']['mape']:.2f}%")
        
        comparison_results['best_model'] = best_model
    
    return comparison_results

if __name__ == "__main__":
    print("ARIMA Modeling Module")
    print("This module provides core ARIMA modeling functionality for book sales forecasting.")
    print("\nUsage examples:")
    print("  from steps._04_arima import *")
    print("  # Complete analysis for a single book:")
    print("  results = run_complete_arima_analysis(data, '9780722532935')")
    print("  # Compare multiple models:")
    print("  comparison = compare_arima_models(data, '9780722532935')")
    print("  # Manual model fitting:")
    print("  model = fit_arima_model(train_data, (1, 0, 1))")
    print("\nFor visualization and plotting:")
    print("  from steps._04_arima_plots import *")
    print("  plot_forecast_results(train_data, test_data, forecast_result, 'Book Title')")