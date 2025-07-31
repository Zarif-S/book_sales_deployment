# Book Sales Analysis - Cleaned Codebase

This repository contains a cleaned and reorganized version of the book sales analysis code, with proper separation of concerns, modular functions, and clear documentation.

## 🏗️ Project Structure

```
book_sales_deployment/
├── steps/
│   ├── __init__.py
│   ├── _01_load_data.py          # Data loading functions
│   └── _02_preprocessing.py      # Data preprocessing functions (CLEANED)
├── utils/
│   ├── __init__.py
│   └── plotting.py               # All plotting functions (NEW)
├── scripts/
│   ├── run_analysis.py           # Main analysis script (NEW)
│   ├── run_classification.py
│   └── run_time_series.py
├── outputs/                      # Generated plots and results
├── pyproject.toml               # Poetry dependencies and project config
├── poetry.lock                  # Poetry lock file for reproducible builds
└── README_CLEANED.md            # This file
```

## 🧹 What Was Cleaned

### Before (Issues):
- ❌ Mixed processing and plotting code in one file
- ❌ No function organization
- ❌ Hardcoded variables scattered throughout
- ❌ No proper error handling
- ❌ No logging
- ❌ Difficult to maintain and extend
- ❌ Linter errors due to malformed text at the beginning

### After (Improvements):
- ✅ Separated processing and plotting into different modules
- ✅ Organized code into logical functions with clear responsibilities
- ✅ Added proper type hints and documentation
- ✅ Implemented logging for better debugging
- ✅ Created reusable functions with parameters
- ✅ Added error handling and validation
- ✅ Clean, maintainable, and extensible code structure

## 📦 Key Components

### 1. Data Preprocessing (`steps/_02_preprocessing.py`)

**Main Functions:**
- `convert_data_types()` - Convert ISBNs to strings and dates to datetime
- `prepare_time_series_data()` - Set date as index and sort chronologically
- `fill_missing_weeks()` - Fill missing weeks with 0 sales
- `filter_data_by_date()` - Filter data by date ranges
- `select_specific_books()` - Select books by ISBN for analysis
- `preprocess_sales_data()` - Main pipeline orchestrator

**Usage:**
```python
from steps._02_preprocessing import preprocess_sales_data

# Run complete preprocessing pipeline
df_processed, df_filtered, selected_books = preprocess_sales_data(df_raw)
```

### 2. Plotting Utilities (`utils/plotting.py`)

**Main Functions:**
- `plot_weekly_volume_by_isbn()` - Plot weekly sales volume
- `plot_yearly_volume_by_isbn()` - Plot yearly aggregated sales
- `plot_selected_books_weekly()` - Plot specific books weekly data
- `plot_selected_books_yearly()` - Plot specific books yearly data
- `plot_sales_comparison()` - Compare sales between time periods
- `create_summary_dashboard()` - Create comprehensive dashboard
- `save_plot()` - Save plots to various formats

**Usage:**
```python
from utils.plotting import plot_weekly_volume_by_isbn, save_plot

# Create and save a plot
fig = plot_weekly_volume_by_isbn(df, "Weekly Sales Volume")
save_plot(fig, "outputs/weekly_sales.html")
```

### 3. Main Analysis Script (`scripts/run_analysis.py`)

**Features:**
- Complete analysis pipeline
- Command-line interface
- Multiple analysis types
- Automatic output generation

**Usage:**
```bash
# Run complete analysis
python scripts/run_analysis.py --analysis-type complete

# Run specific analysis
python scripts/run_analysis.py --analysis-type preprocessing
python scripts/run_analysis.py --analysis-type plotting
python scripts/run_analysis.py --analysis-type comparison
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Using Poetry (recommended)
poetry install

# Or if you prefer pip (not recommended)
pip install -e .
```

### 2. Run Analysis
```bash
# Run complete analysis with real data
python scripts/run_analysis.py

# Run specific analysis type
python scripts/run_analysis.py --analysis-type plotting
```

### 3. Use Individual Functions
```python
import pandas as pd
from steps._02_preprocessing import preprocess_sales_data, get_isbn_to_title_mapping
from utils.plotting import plot_selected_books_weekly, save_plot

# Load your data
df_raw = pd.read_csv('your_data.csv')

# Preprocess
df_processed, _, selected_books = preprocess_sales_data(df_raw)

# Create plots
isbn_to_title = get_isbn_to_title_mapping()
fig = plot_selected_books_weekly(selected_books, isbn_to_title)
save_plot(fig, "my_analysis.html")
```

## 📊 Analysis Features

### Data Preprocessing
- ✅ Convert data types (ISBN to string, dates to datetime)
- ✅ Fill missing weeks with 0 sales for continuous time series
- ✅ Filter data by date ranges
- ✅ Select specific books for detailed analysis
- ✅ Aggregate data to different time frequencies

### Visualization
- ✅ Weekly sales volume plots
- ✅ Yearly aggregated sales plots
- ✅ Comparison plots between time periods
- ✅ Sales trends analysis
- ✅ Comprehensive dashboard with multiple subplots
- ✅ Export plots to HTML, PNG, PDF formats

### Analysis Types
- ✅ Complete pipeline analysis
- ✅ Preprocessing-focused analysis
- ✅ Plotting-focused analysis
- ✅ Comparison analysis between time periods

## 🔧 Configuration

### Logging
The code uses Python's built-in logging module. Configure logging levels in each module:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Output Directory
Plots are automatically saved to the `outputs/` directory. Create it if it doesn't exist:
```python
import os
os.makedirs('outputs', exist_ok=True)
```

## 📝 Code Quality

### Type Hints
All functions include proper type hints for better IDE support and code documentation:
```python
def filter_data_by_date(df: pd.DataFrame, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
```

### Documentation
All functions include comprehensive docstrings with:
- Function description
- Parameter descriptions
- Return value descriptions
- Usage examples

### Error Handling
Functions include proper error handling and validation:
```python
if format not in ['html', 'png', 'jpg', 'svg', 'pdf']:
    raise ValueError(f"Unsupported format: {format}")
```

## 🧪 Testing

The code is structured to be easily testable. You can test individual functions:
```python
# Test preprocessing
from steps._02_preprocessing import convert_data_types
df_test = convert_data_types(sample_df)
assert df_test['ISBN'].dtype == 'object'

# Test plotting
from utils.plotting import plot_weekly_volume_by_isbn
fig = plot_weekly_volume_by_isbn(df_test)
assert fig is not None
```

## 🔄 Extending the Code

### Adding New Plot Types
1. Add new function to `utils/plotting.py`
2. Follow the existing pattern with proper type hints and documentation
3. Use the `save_plot()` and `display_plot()` utilities

### Adding New Preprocessing Steps
1. Add new function to `steps/_02_preprocessing.py`
2. Update the main `preprocess_sales_data()` function if needed
3. Add proper logging and error handling

### Adding New Analysis Types
1. Add new function to `scripts/run_analysis.py`
2. Update the argument parser
3. Add the new analysis type to the main function

## 📈 Performance Considerations

- The code uses efficient pandas operations for data manipulation
- Plotting functions are optimized for large datasets
- Memory usage is monitored through the `get_data_info()` function
- Large datasets can be processed in chunks if needed

## 🤝 Contributing

When contributing to this codebase:
1. Follow the existing code structure and patterns
2. Add proper type hints and documentation
3. Include logging for debugging
4. Add error handling where appropriate
5. Test your changes thoroughly

## 📄 License

This code is part of the book sales analysis project. Please refer to the main project documentation for licensing information. 