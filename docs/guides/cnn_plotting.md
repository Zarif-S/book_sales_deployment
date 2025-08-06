# CNN Model with Comprehensive Plotting Guide

This guide explains how to use the CNN model (`_04_cnn.py`) with the comprehensive plotting functionality from `hybrid_plotting.py`.

## Overview

The CNN model has been enhanced to include comprehensive plotting capabilities using the `comprehensive_model_evaluation` function. This allows you to generate interactive plots and detailed analysis of your CNN model's performance.

## What's New

### Enhanced `_04_cnn.py`
- Added comprehensive plotting functionality at the end of the `train_cnn_step` function
- Integrated with `hybrid_plotting.py` for consistent visualization
- Generates interactive HTML plots and static PNG images
- Saves detailed comparison data to CSV files

### Generated Outputs
When you run the CNN model with plotting, you'll get:

1. **Interactive HTML Plots**: 
   - Main forecast comparison plot
   - Residuals analysis
   - Model performance metrics

2. **Static PNG Images**: 
   - High-resolution plots for reports/presentations

3. **CSV Data Files**: 
   - Detailed comparison data
   - Model predictions and residuals
   - Performance metrics

## How to Use

### Option 1: Quick Test with Sample Data

Run the test script to see the plotting functionality in action:

```bash
python test_cnn_with_plotting.py
```

This will:
- Create synthetic time series data
- Train a CNN model with 5 optimization trials
- Generate comprehensive plots
- Save outputs to `test_outputs/` directory

### Option 2: Use with Your Real Data

Run the main script with your project data:

```bash
python run_cnn_with_plotting.py
```

This will:
- Load data from `data/processed/` or `data/raw/` directories
- Train a CNN model with 20 optimization trials
- Generate comprehensive plots
- Save outputs to `outputs/cnn_plots/` directory

### Option 3: Use in Your Own Code

Import and use the CNN module directly:

```python
from steps._04_cnn import train_cnn_step

# Your data preparation
train_data = ...  # Your training DataFrame
test_data = ...   # Your test DataFrame

# Run CNN with plotting
results = train_cnn_step(
    train_data=train_data,
    test_data=test_data,
    output_dir="my_outputs",
    n_trials=20,
    sequence_length=12,
    forecast_horizon=32,
    study_name="my_cnn_optimization"
)

# Unpack results
results_df, best_hyperparameters_json, final_model, \
residuals_df, test_predictions_df, forecast_comparison_df = results
```

## Output Files

After running the CNN model with plotting, you'll find these files in your output directory:

### Plot Files
- `CNN_Book_Sales_Forecast_*.html` - Interactive Plotly plots
- `CNN_Book_Sales_Forecast_*.png` - Static image plots

### Data Files
- `CNN_Book_Sales_Forecast_*_comparison_data.csv` - Detailed comparison data
- `cnn_residuals.csv` - Training residuals
- `cnn_forecasts.csv` - Test predictions
- `cnn_forecast_comparison.csv` - Forecast comparison data

## Plot Features

The generated plots include:

### Main Forecast Plot
- Training data (blue line)
- Test data (orange line)
- CNN predictions (red line)
- Confidence intervals (if available)
- Performance metrics display

### Residuals Analysis
- Residuals over time
- Residuals distribution
- Autocorrelation analysis
- Model diagnostics

### Performance Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Model comparison statistics

## Configuration Options

You can customize the CNN training and plotting by adjusting these parameters:

### CNN Training Parameters
- `n_trials`: Number of Optuna optimization trials (default: 20)
- `sequence_length`: Input sequence length (default: 12)
- `forecast_horizon`: Prediction horizon (default: 32)
- `study_name`: Optuna study name for persistent storage

### Plotting Parameters
- `save_plots`: Whether to save plots (default: True)
- `output_dir`: Directory to save outputs (default: specified in function call)
- `model_signature`: Custom model identifier for file naming

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Data Format**: Ensure your data has 'volume' or 'Volume' column and date index
3. **Memory Issues**: Reduce `n_trials` for faster execution with less memory usage
4. **Plot Generation Fails**: Check that `hybrid_plotting.py` is available in the `steps/` directory

### Error Messages

- `"Plotting module not available"`: The `hybrid_plotting.py` file is missing
- `"No data files found"`: No CSV files in data directories
- `"Training failed"`: Check data format and model parameters

## Integration with Existing Workflow

The CNN plotting functionality is designed to work seamlessly with your existing pipeline:

1. **ZenML Compatibility**: Returns the same data structures as other models
2. **File Naming**: Uses consistent naming conventions
3. **Output Format**: Matches existing output formats
4. **Error Handling**: Graceful fallback if plotting fails

## Next Steps

After running the CNN model with plotting:

1. **Review Plots**: Open HTML files in your browser for interactive analysis
2. **Analyze Metrics**: Check CSV files for detailed performance data
3. **Compare Models**: Use the same plotting function for other models (LSTM, ARIMA)
4. **Customize**: Modify the plotting parameters for your specific needs

## Example Output

```
🚀 Starting CNN with comprehensive plotting...
============================================================
📊 Loading project data...
📁 Loading data from: data/processed/book_sales_data.csv
✅ Loaded data with 500 rows and columns: ['volume', 'date']
📊 Splitting data for CNN training...
✅ Training set: 400 points
✅ Test set: 100 points

🔧 Running CNN training with comprehensive plotting...
============================================================
Starting CNN training with Optuna optimization...
✅ CNN training completed successfully!
📋 Step 6: Creating comprehensive evaluation plots...
✅ Comprehensive plotting completed!
📊 Plotting results: ['metrics', 'comparison_df', 'fig_main']

✅ CNN training completed successfully!
============================================================

📊 Results Summary:
• Model signature: CNN_Book_Sales_Forecast_filters64_kernel3_layers2
• Training residuals: 388 points
• Test predictions: 100 points
• Test MAE: 15.23
• Test RMSE: 18.45
• Test MAPE: 8.67%

📁 Generated files:
  • CNN_Book_Sales_Forecast_filters64_kernel3_layers2_forecast_comparison.html (245.3 KB)
  • CNN_Book_Sales_Forecast_filters64_kernel3_layers2_forecast_comparison.png (156.7 KB)
  • CNN_Book_Sales_Forecast_filters64_kernel3_layers2_comparison_data.csv (12.1 KB)

🎉 CNN with plotting completed successfully!
📁 Check the 'outputs/cnn_plots' directory for generated plots and data files.
📊 Open the HTML files in your browser to view interactive plots.
```

This enhancement makes it easy to visualize and analyze your CNN model's performance with professional-quality plots and detailed metrics. 