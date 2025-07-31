# ARIMA Module Organization

## Overview

The ARIMA modeling functionality has been organized into two separate modules for better maintainability and production deployment:

1. **`steps/_04_arima.py`** - Core ARIMA modeling functionality
2. **`steps/_04_arima_plots.py`** - Visualization and diagnostic functions

## Module Structure

### Core ARIMA Module (`steps/_04_arima.py`)

This module contains all the core ARIMA modeling functions:

#### Data Preparation Functions
- `prepare_data_for_arima()` - Prepare data for ARIMA modeling by ISBN
- `prepare_data_after_2012()` - Prepare data after 2012-01-01 with split

#### Model Selection and Fitting
- `run_auto_arima()` - Auto ARIMA with optimized parameters (prevents crashes)
- `fit_arima_model()` - Fit ARIMA/SARIMA model with specified parameters
- `arima_predictions()` - Alternative ARIMA implementation with automatic parameter selection

#### Analysis and Evaluation
- `analyze_residuals()` - Comprehensive residual analysis with statistical tests
- `forecast_with_arima()` - Generate forecasts with confidence intervals
- `evaluate_forecast_accuracy()` - Calculate accuracy metrics (MAE, MAPE, RMSE, etc.)

#### Utility Functions
- `save_and_download()` - Save models and results
- `log_model_results()` - Track model performance
- `run_complete_arima_analysis()` - Complete end-to-end analysis
- `compare_arima_models()` - Compare multiple ARIMA models

### Plotting Module (`steps/_04_arima_plots.py`)

This module contains all visualization and diagnostic functions:

#### Core Plotting Functions
- `plot_prediction()` - Main forecast plot with training, test, and forecast data
- `plot_residuals_with_tests()` - Comprehensive residual analysis plots
- `plot_qq_residuals()` - Q-Q plots for normality analysis
- `plot_forecast_with_components()` - Detailed forecast with all components

#### Diagnostic Functions
- `plot_residuals_analysis()` - Residual time series, histogram, ACF, PACF
- `plot_forecast_results()` - Forecast visualization with confidence intervals
- `plot_forecast_accuracy()` - Accuracy analysis plots
- `plot_model_comparison()` - Model comparison bar charts

#### Report Generation
- `create_diagnostic_report()` - Complete diagnostic report with all plots

## Key Improvements Made

### 1. Optimized Auto ARIMA Parameters
The `run_auto_arima()` function now uses optimized parameters to prevent crashes:
- `max_p=2, max_d=0, max_q=2` (reduced from 5 to prevent crashes)
- `max_P=2, max_D=0, max_Q=2` (reduced from 5 to prevent crashes)
- `stationary=True` (prevents session crashes due to RAM usage)
- `maxiter=30, n_jobs=1` (limits iterations and prevents memory issues)

### 2. Enhanced Residual Analysis
The `analyze_residuals()` function now includes:
- Both statsmodels and scipy Jarque-Bera tests
- Student-t distribution fitting for non-normal residuals
- Comprehensive statistical reporting

### 3. Comprehensive Plotting Functions
Added new plotting functions:
- `plot_residuals_with_tests()` - Combines residual plots with statistical tests
- `plot_qq_residuals()` - Q-Q plots for normality analysis
- `plot_forecast_with_components()` - Detailed forecast visualization

## Usage Examples

### Basic ARIMA Analysis

```python
from steps._04_arima import *
from steps._04_arima_plots import *

# Load and prepare data
data = load_your_data()  # Your data loading function
isbn = '9780722532935'

# Run complete analysis
results = run_complete_arima_analysis(
    data=data,
    isbn=isbn,
    forecast_horizon=32,
    use_auto_arima=True,
    seasonal=True,
    title="The Alchemist"
)

# Create diagnostic plots
if results.get('forecast'):
    plot_prediction(
        series_train=results['train_data'],
        series_test=results['test_data'],
        forecast=results['forecast']['forecast'],
        forecast_int=results['forecast']['conf_int'],
        title="The Alchemist - ARIMA Forecast"
    )
```

### Using Alternative Data Preparation

```python
# Prepare data after 2012
train_data, test_data = prepare_data_after_2012(
    book_data=your_book_data,
    column_name='Volume',
    split_size=32
)

# Run Auto ARIMA with custom parameters
auto_results = run_auto_arima(
    train_data=train_data,
    seasonal=True,
    m=52,
    information_criterion='aic',  # or 'bic'
    out_of_sample_size=20  # for validation
)
```

### Residual Analysis

```python
# Get fitted model
fitted_model = auto_results['model']

# Analyze residuals
residual_results = analyze_residuals(fitted_model, "Model Residuals")

# Plot residuals with tests
plot_residuals_with_tests(residual_results['residuals'], "Model Residuals")

# Q-Q plots for normality
plot_qq_residuals(residual_results['residuals'], "Q-Q Analysis")
```

### Model Comparison

```python
# Compare multiple models
comparison = compare_arima_models(
    data=data,
    isbn=isbn,
    models_to_test=[
        (1, 0, 1),  # ARIMA(1,0,1)
        (1, 0, 0),  # ARIMA(1,0,0)
        (0, 0, 1),  # ARIMA(0,0,1)
        (2, 0, 2),  # ARIMA(2,0,2)
    ]
)

# Plot comparison results
plot_model_comparison(comparison, metric='rmse')
```

### Complete Diagnostic Report

```python
# Create comprehensive diagnostic report
report = create_diagnostic_report(
    fitted_model=fitted_model,
    train_data=train_data,
    test_data=test_data,
    forecast_result=forecast_results,
    title="Complete ARIMA Analysis"
)
```

## Important Notes

### Auto ARIMA Parameter Limits
- **max_p, max_q ≤ 2**: Higher values cause crashes
- **max_d, max_D = 0**: Higher values cause crashes
- **max_P, max_Q ≤ 2**: Higher values cause crashes
- **stationary=True**: Prevents RAM issues
- **n_jobs=1**: Prevents memory problems

### Model Performance Insights
Based on the analysis:
- Models perform well on training data but struggle with unseen data
- Residuals show non-normality and heteroskedasticity
- BIC criterion with validation set may perform worse than AIC
- Seasonal components (m=52) are important for weekly data

### Best Practices
1. Use `seasonal=True` for weekly data with yearly patterns
2. Start with conservative parameter limits
3. Always analyze residuals for model adequacy
4. Compare multiple models using `compare_arima_models()`
5. Use the plotting functions for comprehensive diagnostics

## File Dependencies

### Required Imports for Core Module
```python
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, jarque_bera, acf, pacf
from scipy import stats
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
```

### Required Imports for Plotting Module
```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
```

This organization provides a clean separation between core modeling functionality and visualization, making the code more maintainable and suitable for production deployment. 