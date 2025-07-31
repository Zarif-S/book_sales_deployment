# Plot Control Guide for Time Series Diagnostics

## Problem Solved

The original diagnostics script was generating too many plots due to duplicate analysis calls and potentially stale data. This guide explains the improvements made to control plot generation.

## Issues Identified

### 1. **Duplicate Plot Generation**
- The original script called `run_complete_diagnostics()` twice (once for default books, once for all books)
- This resulted in double the plots for each book

### 2. **Excessive Plot Types**
- STL decomposition plots were always shown regardless of limits
- ACF/PACF analysis generated both standard (40 lags) and extended (104 lags) plots
- Decomposition analysis included 3 separate plots per book (original, additive, multiplicative)

### 3. **Potential Incorrect Plots**
- Multiplicative decomposition could be misleading for books with many zero sales weeks
- No validation of data suitability for different decomposition methods

## Solutions Implemented

### 1. **Eliminated Duplicate Analysis**
- Modified `scripts/run_diagnostics.py` to only run diagnostics once
- Removed redundant calls to `run_complete_diagnostics()`

### 2. **Added Plot Control Parameters**
- `show_plots`: Boolean to enable/disable all plots
- `show_plot`: Parameter for individual functions

### 3. **Improved Plot Logic**
- STL decomposition plots now respect the `show_plots` parameter
- Removed redundant extended ACF/PACF plots (104 lags)
- Added validation for multiplicative decomposition

### 4. **Created New Control Functions**
- `run_analysis_with_plot_control()`: Control over plot generation
- `run_custom_analysis()`: Custom book selection with plot control
- `run_diagnostics_with_plots()`: Standard diagnostics with plots

## Usage Examples

### Basic Usage (With Plots)
```python
# Run with plots (recommended)
python scripts/run_diagnostics_minimal.py
```

### Control Plot Generation
```python
from scripts.run_diagnostics import run_analysis_with_plot_control

# No plots at all
results = run_analysis_with_plot_control(show_plots=False)

# Analyze specific books
results = run_analysis_with_plot_control(
    show_plots=True, 
    books=[9780722532935, 9780241003008]
)
```

### Custom Analysis
```python
from scripts.run_diagnostics import run_custom_analysis

# Analyze specific books with plot control
results = run_custom_analysis(
    book_isbns=[9780722532935], 
    show_plots=True
)
```

## Plot Types and Controls

### 1. **Sales Data Plots**
- **Function**: `plot_sales_data()`, `plot_combined_sales_data()`
- **Control**: Controlled by `show_plots` parameter
- **Purpose**: Show raw sales data over time

### 2. **Decomposition Plots**
- **Function**: `perform_stl_decomposition()`, `decompose_and_plot()`
- **Control**: STL plots respect `show_plots`, traditional decomposition controlled by `show_plots`
- **Purpose**: Show trend, seasonal, and residual components

### 3. **ACF/PACF Plots**
- **Function**: `plot_acf_pacf()`
- **Control**: Only standard lags (40) shown, extended lags (104) disabled
- **Purpose**: Show autocorrelation patterns for ARIMA model selection

### 4. **Stationarity Test Results**
- **Function**: Various test functions (`adf_test()`, `kpss_test()`, etc.)
- **Control**: No plots, only text output
- **Purpose**: Statistical tests for stationarity

## Recommended Settings

### For Quick Analysis
```python
# Standard plots, essential analysis
run_analysis_with_plot_control(
    show_plots=True
)
```

### For Detailed Analysis
```python
# All plots, comprehensive analysis
run_analysis_with_plot_control(
    show_plots=True
)
```

### For Batch Processing
```python
# No plots, pure analysis
run_analysis_with_plot_control(
    show_plots=False
)
```

## Files Modified

1. **`scripts/run_diagnostics.py`**
   - Eliminated duplicate analysis calls
   - Added plot control functions
   - Improved user guidance

2. **`steps/_03__time_series_diagnostics.py`**
   - Added `show_plot` parameter to `perform_stl_decomposition()`
   - Improved `decompose_and_plot()` with validation
   - Removed redundant ACF/PACF plots
   - Better plot counting logic

3. **`scripts/run_diagnostics_minimal.py`** (New)
   - Interactive diagnostics script
   - Choice between plots or no plots
   - Analysis-only option

## Benefits

1. **Eliminated Duplicate Plots**: No more double analysis calls
2. **Better Performance**: No redundant plot generation
3. **Improved Accuracy**: Validation prevents misleading plots
4. **Flexible Usage**: Simple on/off control for plots
5. **Clear Documentation**: Users understand what each option does

## Migration Guide

### From Old Script
```python
# Old way (generates many plots)
from steps._03__time_series_diagnostics import run_complete_diagnostics
results = run_complete_diagnostics()
```

### To New Script
```python
# New way (controlled plots)
from scripts.run_diagnostics import run_analysis_with_plot_control
results = run_analysis_with_plot_control(show_plots=True)
```

This approach gives you the same analysis results but without duplicate plot generation. 