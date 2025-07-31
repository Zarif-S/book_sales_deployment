# Persistent Optuna Storage and ZenML Caching Implementation

## Overview

This document describes the modifications made to the ZenML pipeline to implement:
1. **Persistent Optuna storage** with SQLite and `load_if_exists=True`
2. **ZenML step caching** to skip re-runs when inputs haven't changed
3. **Proper metadata handling** by converting dictionaries to strings
4. **Structured outputs** returning both results DataFrame and hyperparameters

## Key Changes Made

### 1. Persistent Optuna Storage (`steps/_04_arima_zenml_mlflow_optuna.py`)

#### Modified `run_optuna_optimization()` function:
```python
def run_optuna_optimization(train_series: pd.Series, test_series: pd.Series, 
                          n_trials: int = 30, study_name: str = "arima_optimization") -> Dict[str, Any]:
    # Create storage directory for Optuna studies
    storage_dir = os.path.join(os.getcwd(), "optuna_storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create SQLite storage URL for persistent storage
    storage_url = f"sqlite:///{os.path.join(storage_dir, f'{study_name}.db')}"
    
    # Create study with persistent storage and load_if_exists=True
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,  # Key feature: resume from previous runs
        direction="minimize"
    )
```

**Why this change:**
- `load_if_exists=True` allows Optuna to resume optimization from where it left off
- SQLite storage persists trials across pipeline runs
- Each study gets its own database file for isolation

### 2. ZenML Step Caching and Structured Outputs

#### Modified step decorator and return type:
```python
@step(
    enable_cache=True,  # Enable ZenML caching to skip re-runs when inputs haven't changed
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    output_materializers=PandasMaterializer
)
def train_arima_optuna_step(
    modelling_data: pd.DataFrame,
    n_trials: int = 30,
    study_name: str = "arima_optimization"
) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="arima_results")],
    Annotated[Dict, ArtifactConfig(name="best_hyperparameters")]
]:
```

**Why this change:**
- `enable_cache=True` allows ZenML to skip the step if inputs and code haven't changed
- Returns both results DataFrame and hyperparameters as separate artifacts
- Structured outputs enable better caching granularity

### 3. Metadata Handling (Dict to String Conversion)

#### All dictionary metadata converted to strings:
```python
# Add metadata to step context - CONVERT ALL DICTS TO STRINGS
metadata_dict = {
    "best_params_str": str(best_params),  # Convert dict to string
    "eval_metrics_str": str(eval_metrics),  # Convert dict to string
    "optimization_results_str": str(optimization_results),  # Convert dict to string
    "training_periods": len(time_series),
    "books_count": df_work['book_name'].nunique(),
    "optuna_study_name": study_name,
    "optuna_trials_completed": optimization_results["n_trials"],
    "best_rmse_value": float(optimization_results["best_value"]),
    "storage_url": optimization_results["storage_url"]
}
```

**Why this change:**
- ZenML metadata must be hashable (no dictionaries)
- Converting to strings prevents "unhashable type: 'dict'" errors
- Maintains all information while being compatible with ZenML

### 4. Enhanced MLflow Logging

#### Added Optuna-specific metrics to MLflow:
```python
mlflow.log_metric("optuna_trials", optimization_results["n_trials"])
mlflow.log_metric("best_rmse", optimization_results["best_value"])
```

**Why this change:**
- Tracks optimization progress in MLflow
- Enables monitoring of hyperparameter tuning across runs
- Maintains existing MLflow functionality

### 5. Pipeline Integration (`pipelines/zenml_pipeline_with_arima.py`)

#### Updated pipeline to handle new return structure:
```python
# Train ARIMA models with Optuna optimization
arima_results, best_hyperparameters = train_arima_optuna_step(
    modelling_data=modelling_data,
    n_trials=30,
    study_name="book_sales_arima_optimization"
)

return {
    "df_merged": df_merged,
    "quality_report": quality_report,
    "processed_data_path": processed_data_path,
    "modelling_data": modelling_data,
    "arima_results": arima_results,
    "best_hyperparameters": best_hyperparameters,  # New output
}
```

## Benefits of These Changes

### 1. Persistent Optuna Storage
- **Resume optimization**: Trials accumulate across pipeline runs
- **Faster convergence**: More trials lead to better hyperparameters
- **Resource efficiency**: No need to restart from scratch
- **Study isolation**: Each study has its own database file

### 2. ZenML Caching
- **Skip unnecessary runs**: Step is skipped when inputs haven't changed
- **Faster development**: No waiting for repeated optimization
- **Cost savings**: Reduces computational resources
- **Reproducibility**: Ensures consistent results

### 3. Structured Outputs
- **Better organization**: Separate artifacts for results and hyperparameters
- **Easier access**: Direct access to best hyperparameters
- **Improved caching**: More granular cache invalidation
- **Enhanced metadata**: Rich metadata for each output

### 4. Proper Metadata Handling
- **No errors**: Eliminates "unhashable type" errors
- **Complete information**: All data preserved as strings
- **ZenML compatibility**: Works with ZenML's metadata system
- **Debugging support**: Rich metadata for troubleshooting

## Usage Examples

### Running the Pipeline
```python
# The pipeline will now use persistent storage and caching
book_sales_arima_pipeline(
    output_dir=output_dir,
    selected_isbns=default_selected_isbns,
    column_name='Volume',
    split_size=32
)
```

### Accessing Results
```python
# Results now include both DataFrame and hyperparameters
results = book_sales_arima_pipeline(...)
arima_results_df = results["arima_results"]
best_hyperparameters = results["best_hyperparameters"]

# Access best hyperparameters directly
best_params = best_hyperparameters["best_params"]
optimization_summary = best_hyperparameters["optimization_results"]
```

### Monitoring Progress
- Check `optuna_storage/` directory for SQLite database files
- Monitor MLflow for optimization metrics
- Use ZenML dashboard to see cached vs. executed steps

## Testing

A comprehensive test script (`test_persistent_optuna.py`) is provided to verify:
1. Persistent Optuna storage functionality
2. Metadata conversion (dict to string)
3. ZenML step structure
4. MLflow integration

Run the test:
```bash
python test_persistent_optuna.py
```

## File Structure

```
book_sales_deployment/
├── steps/
│   └── _04_arima_zenml_mlflow_optuna.py  # Modified with persistent storage
├── pipelines/
│   └── zenml_pipeline_with_arima.py      # Updated to handle new outputs
├── optuna_storage/                       # Created automatically
│   └── *.db                             # SQLite database files
├── test_persistent_optuna.py            # Test script
└── PERSISTENT_OPTUNA_CHANGES.md         # This documentation
```

## Migration Notes

### For Existing Users
1. **No breaking changes**: Existing pipeline calls work the same
2. **Automatic storage**: Optuna storage is created automatically
3. **Backward compatibility**: All existing functionality preserved
4. **Enhanced outputs**: Additional hyperparameters output available

### For New Users
1. **Immediate benefits**: Caching and persistent storage work out of the box
2. **Better monitoring**: Enhanced MLflow logging
3. **Structured access**: Easy access to optimization results
4. **Error prevention**: No more metadata errors

## Troubleshooting

### Common Issues

1. **Storage directory not created**
   - Ensure write permissions in project directory
   - Check if `optuna_storage/` directory exists

2. **Caching not working**
   - Verify `enable_cache=True` in step decorator
   - Check if inputs have actually changed

3. **Metadata errors**
   - All dictionaries are now converted to strings
   - Check for any remaining dict objects in metadata

4. **MLflow logging issues**
   - Ensure MLflow experiment tracker is configured
   - Check network connectivity for MLflow server

### Debug Commands

```python
# Check Optuna storage
import os
print(os.listdir("optuna_storage/"))

# Check step caching
from zenml import get_step_context
context = get_step_context()
print(context.get_output_metadata())

# Verify metadata types
metadata = context.get_output_metadata()
for key, value in metadata.items():
    print(f"{key}: {type(value)}")
```

## Future Enhancements

1. **Distributed optimization**: Use Redis storage for multi-node optimization
2. **Advanced caching**: Implement custom cache keys for more granular control
3. **Hyperparameter tracking**: Enhanced tracking of hyperparameter evolution
4. **Automated cleanup**: Automatic cleanup of old Optuna studies
5. **Performance monitoring**: Real-time optimization progress monitoring 