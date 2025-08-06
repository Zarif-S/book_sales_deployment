# ARIMA Pipeline Integration

This document describes the integration of ARIMA modeling into the book sales data pipeline.

## Overview

The ARIMA pipeline has been successfully integrated into the existing ZenML pipeline structure. The pipeline now includes:

1. **Data Loading**: ISBN and UK weekly sales data
2. **Data Preprocessing**: Cleaning and merging of datasets
3. **Data Quality Analysis**: Comprehensive quality metrics
4. **Data Preparation**: Train/test splitting for selected books
5. **ARIMA Modeling**: Time series modeling with Optuna optimization
6. **Results Storage**: Comprehensive results in structured format

## Files Added/Modified

### New Files
- `pipelines/zenml_pipeline_with_arima.py` - Complete ARIMA pipeline
- `scripts/run_arima_pipeline.py` - Script to run the ARIMA pipeline
- `README_ARIMA_PIPELINE.md` - This documentation

### Modified Files
- `pipelines/zenml_pipeline_with_modelling_prep.py` - Added ARIMA step import and integration

## Pipeline Components

### ARIMA Step (`steps/_04_arima_zenml_mlflow_optuna.py`)

The ARIMA step performs the following operations:

1. **Time Series Creation**: Aggregates volume across all books per date
2. **Train/Test Split**: Splits data into training (all but last 32 periods) and test (last 32 periods)
3. **Optuna Optimization**: Runs 30 trials to find optimal ARIMA parameters
4. **Model Training**: Trains final model on full dataset
5. **Results Extraction**: Extracts residuals, fitted values, and forecasts
6. **MLflow Integration**: Logs parameters and metrics to MLflow
7. **Structured Output**: Returns comprehensive results in DataFrame format

### Key Functions

- `create_time_series_from_df()`: Converts DataFrame to time series
- `split_time_series()`: Splits into train/test sets
- `objective()`: Optuna objective function for hyperparameter optimization
- `run_optuna_optimization()`: Manages Optuna study
- `train_final_arima_model()`: Trains final model with best parameters
- `evaluate_forecast()`: Computes MAE, RMSE, and MAPE metrics

## Usage

### Running the Complete ARIMA Pipeline

```bash
# From the project root
python scripts/run_arima_pipeline.py
```

### Running from Python

```python
from pipelines.zenml_pipeline_with_arima import book_sales_arima_pipeline

# Run the pipeline
results = book_sales_arima_pipeline(
    output_dir='data/processed',
    selected_isbns=['9780722532935', '9780241003008'],
    column_name='Volume',
    split_size=32
)
```

### Pipeline Parameters

- `output_dir`: Directory to save processed data
- `selected_isbns`: List of ISBNs to model (defaults to The Alchemist and The Very Hungry Caterpillar)
- `column_name`: Column to use for time series analysis (default: 'Volume')
- `split_size`: Number of periods for test set (default: 32 weeks)

## Output Structure

The pipeline returns a dictionary with the following keys:

- `df_merged`: Merged and processed DataFrame
- `quality_report`: Data quality metrics
- `processed_data_path`: Path to saved processed data
- `modelling_data`: Prepared data for modeling
- `arima_results`: Comprehensive ARIMA results DataFrame

### ARIMA Results DataFrame Structure

The `arima_results` DataFrame contains the following result types:

1. **Model Configuration** (`result_type='model_config'`)
   - ARIMA order parameters (p, d, q)
   - Seasonal order parameters (P, D, Q, s)

2. **Evaluation Metrics** (`result_type='evaluation'`)
   - MAE, RMSE, MAPE on test set

3. **Forecast Results** (`result_type='forecast'`)
   - Actual vs predicted values for test periods

4. **Model Diagnostics** (`result_type='diagnostics'`)
   - Residuals, fitted values, and actual values

5. **Model Artifacts** (`result_type='model_artifact'`)
   - Serialized trained model

6. **Summary Statistics** (`result_type='summary'`)
   - Training data summary

## ARIMA Model Details

### Model Type
- **SARIMAX**: Seasonal ARIMA with exogenous variables
- **Seasonality**: Weekly (s=52) based on data frequency
- **Optimization**: Optuna with 30 trials, 30-minute timeout

### Hyperparameter Search Space
- `p`: [0, 3] - AR order
- `d`: [0, 2] - Differencing order
- `q`: [0, 3] - MA order
- `P`: [0, 2] - Seasonal AR order
- `D`: [0, 1] - Seasonal differencing order
- `Q`: [0, 2] - Seasonal MA order

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error (handles zero values)

## MLflow Integration

The ARIMA step automatically logs to MLflow when available:
- Model parameters (best ARIMA order)
- Evaluation metrics
- Data statistics
- Training metadata

## Error Handling

The pipeline includes comprehensive error handling:
- Graceful failure with error DataFrames
- Detailed logging at each step
- Validation of input data structure
- Timeout handling for optimization

## Dependencies

The ARIMA pipeline requires the following additional dependencies:
- `statsmodels` - For SARIMAX models
- `optuna` - For hyperparameter optimization
- `mlflow` - For experiment tracking (optional)

## Monitoring and Logging

- All steps include detailed logging
- Metadata is added to ZenML artifacts
- MLflow experiment tracking (when available)
- Comprehensive error reporting

## Next Steps

Potential enhancements for the ARIMA pipeline:
1. Individual book modeling (separate models per book)
2. Additional time series models (LSTM, Prophet)
3. Ensemble methods
4. Automated model selection
5. Real-time forecasting capabilities 