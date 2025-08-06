# Book Sales Time Series Analysis & MLOps Pipeline

A comprehensive time series forecasting project for book sales data using multiple machine learning approaches with MLOps best practices.

## 🎯 Project Overview

This project implements a complete MLOps pipeline for book sales forecasting using:
- **Multiple ML Models**: ARIMA, CNN, LSTM, and hybrid approaches
- **ZenML Pipeline Orchestration**: Reproducible ML workflows
- **Experiment Tracking**: MLflow integration with Optuna optimization
- **Comprehensive Analysis**: Time series diagnostics and model comparison
- **Production Ready**: FastAPI serving with Docker deployment

## 🏗️ Project Structure

```
book_sales_deployment/
├── README.md                    # This file
├── pyproject.toml              # Poetry dependencies and project config
├── poetry.lock                 # Reproducible dependency versions
│
├── steps/                      # ZenML pipeline steps
│   ├── _01_load_data.py        # Data loading functions
│   ├── _02_preprocessing.py    # Data preprocessing and cleaning
│   ├── _03__time_series_diagnostics.py  # Time series analysis
│   ├── _04_arima_standalone.py # ARIMA modeling
│   ├── _04_cnn_standalone.py   # CNN modeling
│   ├── _04_lstm_standalone.py  # LSTM modeling
│   └── _05_lstm_*_residuals.py # Hybrid modeling approaches
│
├── pipelines/                  # ZenML pipeline definitions
│   ├── zenml_pipeline_arima_lstm.py
│   ├── zenml_pipeline_cnn_lstm.py
│   └── zenml_pipeline.py
│
├── utils/                      # Utility functions
│   └── plotting.py            # Comprehensive plotting utilities
│
├── scripts/                    # Execution scripts
│   ├── run_analysis.py        # Main analysis runner
│   ├── comprehensive_diagnostics.py  # Complete automated diagnostics
│   └── interactive_diagnostics.py    # Interactive user-guided diagnostics
│
├── data/                       # Data storage
│   ├── raw/                   # Original datasets
│   └── processed/             # Cleaned and processed data
│
├── outputs/                    # Organized model outputs
│   ├── models/                # Model-specific results
│   ├── plots/                 # Visualization outputs
│   └── data/                  # Generated datasets
│
├── experiments/                # Experiment trials organized by model type
│   ├── lstm/standalone/        # Pure LSTM optimization trials
│   ├── hybrid/arima_residuals/ # LSTM+ARIMA hybrid experiments  
│   ├── hybrid/cnn_residuals/   # LSTM+CNN hybrid experiments
│   └── optuna_storage/         # Optuna optimization databases
├── docs/                       # Comprehensive documentation
└── tests/                      # Test suite
```

## 🚀 Quick Start

### 1. Environment Setup

Using Poetry (recommended) [[memory:5369862]]:
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 2. Data Preparation

```bash
# Load and preprocess data
poetry run python scripts/run_analysis.py --analysis-type preprocessing
```

### 3. Run Time Series Analysis

```bash
# Complete time series diagnostics
poetry run python scripts/comprehensive_diagnostics.py

# Run specific model
poetry run python -m steps._04_arima_standalone
poetry run python -m steps._04_cnn_standalone  
poetry run python -m steps._04_lstm_standalone
```

### 4. Run Complete Pipeline

```bash
# ARIMA + LSTM hybrid pipeline
poetry run python pipelines/zenml_pipeline_arima_lstm.py

# CNN + LSTM hybrid pipeline  
poetry run python pipelines/zenml_pipeline_cnn_lstm.py
```

## 📊 Available Models

### ARIMA (Statistical)
- Seasonal ARIMA with Optuna optimization
- Comprehensive residual analysis
- 30 optimization trials with timeout handling
- **Best for**: Baseline forecasting, interpretable results

### CNN (Deep Learning)
- 1D Convolutional Neural Networks for time series
- Hyperparameter optimization with Optuna
- Sequence-to-sequence forecasting
- **Best for**: Pattern recognition in time series

### LSTM (Recurrent Neural Networks)
- Long Short-Term Memory networks
- Handles long-term dependencies
- Multiple architecture variants
- **Best for**: Complex temporal patterns

### Hybrid Models
- **ARIMA + LSTM**: Statistical + Deep learning combination
- **CNN + LSTM**: Pattern recognition + temporal modeling
- **Residual Modeling**: Multi-stage forecasting approach

## 📈 Key Features

### Time Series Diagnostics
- **Stationarity Tests**: ADF, KPSS, Phillips-Perron, Zivot-Andrews
- **Decomposition Analysis**: STL and traditional seasonal decomposition
- **ACF/PACF Analysis**: Autocorrelation patterns for model selection
- **COVID-19 Impact Analysis**: Structural break detection

### Model Optimization
- **Optuna Integration**: Automated hyperparameter tuning
- **MLflow Tracking**: Experiment logging and comparison
- **Cross-validation**: Time series aware validation
- **Performance Metrics**: MAE, RMSE, MAPE with statistical significance

### Visualization
- **Interactive Plots**: Plotly-based dashboards
- **Model Comparison**: Side-by-side performance analysis
- **Residual Analysis**: Comprehensive diagnostic plots
- **Forecast Visualization**: Confidence intervals and uncertainty

## 🎛️ Configuration

### Default Books Analyzed
- **9780722532935**: The Alchemist (Paperback)
- **9780241003008**: The Very Hungry Caterpillar (Hardback)

### Key Parameters
- **Forecast Horizon**: 32 weeks (configurable)
- **Training Split**: 80/20 train/test split
- **Optimization Trials**: 20-30 per model
- **Seasonal Period**: 52 weeks (yearly seasonality)

## 📁 Output Organization

All results are saved to organized directories [[memory:5369851]]:

```
outputs/
├── models/
│   ├── arima/forecasts/        # ARIMA predictions and diagnostics
│   ├── cnn/forecasts/          # CNN model outputs
│   └── lstm/forecasts/         # LSTM model results
├── plots/
│   ├── interactive/            # HTML interactive plots
│   └── static/                 # PNG static images
└── data/
    ├── comparisons/            # Model comparison data
    └── residuals/              # Residual analysis results
```

## 🔧 Development

### Running Tests
```bash
poetry run python -m pytest tests/
```

### Code Quality
- Type hints throughout codebase
- Comprehensive docstrings
- Logging for debugging
- Error handling and validation

### Adding New Models
1. Create new step in `steps/_04_your_model.py`
2. Follow existing patterns for consistency
3. Add comprehensive plotting integration
4. Include in pipeline definitions

## 📚 Documentation

Comprehensive documentation available in `docs/`:
- **API Documentation**: Detailed function references
- **User Guides**: Step-by-step tutorials
- **Development Notes**: Architecture and design decisions

## 🎯 Key Insights

### Seasonal Patterns
- **Christmas Season**: Both books show increased sales
- **Children's Book Week**: Additional peaks for children's books
- **Spring Sales**: March increases for seasonal titles

### Model Performance
- **ARIMA**: Good baseline, interpretable coefficients
- **CNN**: Excellent pattern recognition, handles seasonality
- **LSTM**: Best for complex temporal dependencies
- **Hybrid**: Combines strengths, often best overall performance

### COVID-19 Impact
- Clear structural breaks during UK lockdowns
- Zero sales weeks during restrictions
- Recovery patterns vary by book type

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow existing code patterns and documentation standards
4. Add tests for new functionality
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ZenML**: Pipeline orchestration framework
- **MLflow**: Experiment tracking and model management
- **Optuna**: Hyperparameter optimization
- **Plotly**: Interactive visualization capabilities

---

**Quick Commands Reference:**
```bash
# Complete analysis pipeline
poetry run python scripts/run_analysis.py

# Time series diagnostics  
poetry run python scripts/comprehensive_diagnostics.py

# Individual models
poetry run python -m steps._04_arima_standalone
poetry run python -m steps._04_cnn_standalone
poetry run python -m steps._04_lstm_standalone

# Hybrid pipelines
poetry run python pipelines/zenml_pipeline_arima_lstm.py
```

For detailed API documentation and guides, see the `docs/` directory.