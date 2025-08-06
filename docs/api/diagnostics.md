# Time Series Diagnostics for Book Sales Data

This repository contains a cleaned and organized codebase for performing comprehensive time series diagnostics on book sales data. The analysis focuses on two books: "The Alchemist" and "The Very Hungry Caterpillar".

## 📁 Project Structure

```
book_sales_deployment/
├── steps/
│   └── _03__time_series_diagnostics.py    # Main diagnostics module
├── scripts/
│   ├── comprehensive_diagnostics.py       # Complete automated diagnostics
│   ├── interactive_diagnostics.py         # Interactive user-guided diagnostics
│   └── arima_analysis.ipynb              # ARIMA analysis notebook
└── README_TIME_SERIES_DIAGNOSTICS.md     # This file
```

## 🎯 Features

### Time Series Diagnostics Module (`steps/_03__time_series_diagnostics.py`)

The main module provides comprehensive time series analysis including:

**Default Books:**
- **9780722532935**: The Alchemist (Paperback)
- **9780241003008**: The Very Hungry Caterpillar (Hardback)

The module automatically analyzes these default books unless specified otherwise.

#### 1. **Decomposition Analysis**
- STL (Seasonal and Trend decomposition using Loess) decomposition
- Additive and multiplicative seasonal decomposition
- Trend, seasonal, and residual component analysis
- Visualization of decomposition components

#### 2. **ACF and PACF Analysis**
- Autocorrelation Function (ACF) analysis
- Partial Autocorrelation Function (PACF) analysis
- Confidence intervals for significance testing
- Extended lag analysis (104 lags for seasonal patterns)

#### 3. **Stationarity Tests**
- **Augmented Dickey-Fuller (ADF) Test**: Tests for unit root non-stationarity
- **KPSS Test**: Tests for trend stationarity
- **Phillips-Perron Test**: Robust to heteroscedasticity and serial correlation
- **Zivot-Andrews Test**: Tests for structural breaks

#### 4. **Ljung-Box Test**
- Tests for residual autocorrelation
- Multiple lag analysis (1-10 lags)
- Statistical significance testing

#### 5. **COVID-19 Analysis**
- Identification of zero-sales weeks during lockdowns
- Documentation of UK lockdown periods
- Analysis of COVID-19 impact on sales patterns

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib plotly statsmodels scipy
```

For ARIMA analysis:
```bash
pip install pmdarima scikit-learn
```

### 2. Run Diagnostics

```python
# Import the diagnostics module
from steps._03__time_series_diagnostics import *

# Set your data
set_data(sales_dataframe, selected_books_dataframe)

# Run complete diagnostics (uses default books)
results = run_complete_diagnostics()

# Or for specific books
results = run_diagnostics_for_books(['9780722532935', '9780241003008'])

# Or for all available books
results = run_complete_diagnostics(get_all_books())
```

### 3. Run the Demo Script

```bash
python scripts/comprehensive_diagnostics.py
```

### 4. ARIMA Analysis

Open the Jupyter notebook for ARIMA modeling:

```bash
jupyter notebook scripts/arima_analysis.ipynb
```

## 📊 Analysis Components

### Decomposition Analysis

The module performs both STL and traditional seasonal decomposition:

```python
# STL Decomposition
stl_results = perform_stl_decomposition(data, title, period=52)

# Traditional decomposition
decompose_and_plot(dataframe, title, period=52)
```

**Key Findings:**
- Both books show seasonal variation around Christmas
- The Very Hungry Caterpillar shows additional peaks in March
- Additive decomposition suitable for The Alchemist
- Multiplicative decomposition suitable for The Very Hungry Caterpillar

### ACF/PACF Analysis

```python
# Standard analysis
plot_acf_pacf(data, title, lags=40)

# Extended analysis for seasonal patterns
plot_acf_pacf(data, title, lags=104)
```

**Key Findings:**
- Strong autocorrelation that decays slowly
- No sharp cutoff in ACF (MA order not easily determined)
- PACF shows strong correlation at lag 1
- Seasonal patterns present, especially for children's books

### Stationarity Tests

```python
# Multiple stationarity tests
adf_result = adf_test(data, label)
kpss_result = kpss_test(data, label)
pp_result = pp_test(data, label)
za_result = za_test(data, label)
```

**Key Findings:**
- ADF tests confirm both time series are stationary
- ARIMA/SARIMA models can be implemented without differencing
- Structural breaks may be present (COVID-19 impact)

## 🔧 API Reference

### Main Functions

#### `set_data(sales_dataframe, selected_books_dataframe)`
Set the sales data for analysis.

#### `run_complete_diagnostics()`
Run all diagnostic tests in sequence for default books.

#### `perform_decomposition_analysis()`
Perform comprehensive decomposition analysis for default books.

#### `perform_acf_pacf_analysis()`
Perform ACF, PACF, and Ljung-Box analysis for default books.

#### `perform_stationarity_analysis()`
Perform multiple stationarity tests for default books.

### Utility Functions

#### `plot_sales_data(data, isbn, title, color)`
Plot sales data using Plotly.

#### `analyze_covid_weeks()`
Analyze COVID-19 lockdown periods.

#### `plot_acf_pacf(data, title, lags=40, alpha=0.05)`
Plot ACF and PACF with confidence intervals.

## 📈 ARIMA Analysis Notebook

The `scripts/arima_analysis.ipynb` notebook provides:

1. **Auto ARIMA Model Selection**
   - Automatic parameter optimization
   - Seasonal and non-seasonal models
   - AIC-based model selection

2. **Model Diagnostics**
   - Residuals analysis
   - Normality tests
   - Autocorrelation tests

3. **Forecasting**
   - In-sample and out-of-sample forecasts
   - Confidence intervals
   - Performance metrics (MSE, RMSE, MAE, MAPE)

4. **Model Comparison**
   - Performance comparison between books
   - Statistical significance testing
   - Recommendations for model selection

## 🎯 Key Insights

### Seasonal Patterns
- **Christmas Season**: Both books show increased sales
- **Children's Book Week**: The Very Hungry Caterpillar shows additional peaks
- **March Peak**: Children's book shows spring sales increase

### COVID-19 Impact
- Zero sales during UK lockdown periods
- Clear structural breaks in the time series
- Recovery patterns post-lockdown

### Model Recommendations
- **SARIMA models** recommended due to strong seasonality
- **Seasonal period**: 52 weeks (weekly data)
- **Parameter ranges**: p, q ≤ 5, P, Q ≤ 2
- **Differencing**: Not required (data is stationary)

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install statsmodels plotly scipy
   ```

2. **Data Format Issues**
   - Ensure data has datetime index
   - Check for missing values
   - Verify 'Volume' column exists

3. **Memory Issues**
   - Reduce number of lags in ACF/PACF analysis
   - Use smaller datasets for testing

### Performance Tips

1. **Large Datasets**
   - Use sampling for initial exploration
   - Reduce plot resolution for faster rendering

2. **Multiple Tests**
   - Run tests individually for debugging
   - Use `warnings.filterwarnings('ignore')` for cleaner output

## 📝 Example Usage

```python
import pandas as pd
from steps._03__time_series_diagnostics import *

# Load your data
sales_data = pd.read_csv('sales_data.csv', index_col=0, parse_dates=True)

# Set data
set_data(sales_data)

# Run specific analyses (uses default books)
decomposition_results = perform_decomposition_analysis()
acf_pacf_results = perform_acf_pacf_analysis()
stationarity_results = perform_stationarity_analysis()

# Or run everything (uses default books)
complete_results = run_complete_diagnostics()

# Or for specific books
specific_results = run_diagnostics_for_books(['9780722532935', '9780241003008'])

# Or for all available books
all_results = run_complete_diagnostics(get_all_books())
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UK Government lockdown data for COVID-19 analysis
- Children's Book Week information from Publishers Weekly
- Statsmodels library for time series analysis tools
- Plotly for interactive visualizations

---

**Note**: This codebase has been cleaned and organized from the original analysis. All syntax errors have been fixed, and the code is now modular and well-documented for easy use and maintenance. 