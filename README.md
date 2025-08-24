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
│   ├── zenml_pipeline.py            # Main ZenML pipeline
│   └── zenml_pipeline_cnn_lstm_testing.py  # CNN+LSTM testing pipeline
│
├── utils/                      # Utility functions
│   └── plotting.py            # Comprehensive plotting utilities
│
├── scripts/                    # Analysis and utility scripts
│   ├── comprehensive_diagnostics.py  # Complete automated diagnostics
│   ├── interactive_diagnostics.py    # Interactive user-guided diagnostics
│   ├── arima_model_diagnostics.py    # ARIMA-specific diagnostics
│   ├── seasonality_business_insights.py  # Business insights analysis
│   └── seasonality_deployment_analysis.py # Deployment readiness analysis
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
├── mlruns/                     # MLflow experiment tracking data
│   ├── models/                 # Registered model versions
│   └── [experiment_ids]/       # Individual experiment runs
├── docs/                       # Comprehensive documentation
├── dev/                        # Development files and archives
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

### 2. ZenML Setup (if using ZenML pipelines)

```bash
# Install ZenML integrations
zenml integration install mlflow

# Set up experiment tracker (optional)
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
```

### 3. Data Preparation

```bash
# Load and preprocess data
poetry run python -m steps._01_load_data
poetry run python -m steps._02_preprocessing
```

### 4. Run Time Series Analysis

```bash
# Complete time series diagnostics
poetry run python scripts/comprehensive_diagnostics.py

# Run specific model
poetry run python -m steps._04_arima_standalone
poetry run python -m steps._04_cnn_standalone  
poetry run python -m steps._04_lstm_standalone
```

### 5. Run Complete Pipeline

```bash
# Main ZenML pipeline
poetry run python pipelines/zenml_pipeline.py

# CNN + LSTM testing pipeline  
poetry run python pipelines/zenml_pipeline_cnn_lstm_testing.py
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

All results are saved to organized directories:

```
outputs/
├── models/
│   ├── arima/forecasts/        # ARIMA predictions and diagnostics
│   ├── cnn/forecasts/          # CNN model outputs
│   └── lstm/forecasts/         # LSTM model results
├── plots/
│   ├── interactive/            # HTML interactive plots
│   └── static/                 # PNG static images
├── data/
│   ├── comparisons/            # Model comparison data
│   └── residuals/              # Residual analysis results
├── seasonality_analysis/       # Business insights and deployment analysis
└── value_seasonality_analysis/ # Value vs seasonality correlation analysis
```

## 🔬 MLflow Integration

The project includes comprehensive experiment tracking:

```bash
# View MLflow UI (after running experiments)
mlflow ui

# Access at http://localhost:5000
```

**Tracked Metrics:**
- Model performance (MAE, RMSE, MAPE)
- Hyperparameter optimization results
- Model artifacts and plots
- Individual book model comparisons

## ☁️ Production MLOps Architecture

**Current Architecture:** Hybrid orchestration (local) + Cloud storage + Remote MLflow + Vertex AI deployment

### Quick Deployment Workflow
```bash
# 1. Train models (local orchestration with cloud storage)
python pipelines/zenml_pipeline.py

# 2. Upload models to GCS for Vertex AI
python deploy/02_upload_models_to_gcs.py --upload-all

# 3. Deploy to Vertex AI endpoints
python deploy/03_deploy_to_vertex_endpoints.py --deploy-all
```

For detailed setup instructions and architecture overview, see **`deploy/DEPLOYMENT_README.md`** and **`PIPELINE_README.md`**.

## 🔧 Development

### Running Tests
```bash
# Run all tests
poetry run python -m pytest tests/

# Run specific test files
poetry run python -m pytest tests/test_01_load_data.py -v
poetry run python -m pytest tests/test_02_preprocessing.py -v
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
- **API Documentation**: Detailed function references (`docs/api/`)
- **User Guides**: Step-by-step tutorials (`docs/guides/`)
- **ZenML Pipeline Guide**: Optimization and best practices
- **Development Notes**: Architecture and design decisions (`dev/`)

## 🎯 Key Insights

### Seasonal Patterns
- **Christmas Season**: Both books show increased sales
- **Children's Book Week**: Additional peaks for children's books
- **Spring Sales**: March increases for seasonal titles

### Model Performance
- **ARIMA**: Good baseline, interpretable coefficients
- **CNN**: Excellent pattern recognition, handles seasonality
- **LSTM**: Best for complex temporal dependencies
- **Hybrid**: Combines different approaches, performance varies by use case

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
# Time series diagnostics  
poetry run python scripts/comprehensive_diagnostics.py

# Individual models
poetry run python -m steps._04_arima_standalone
poetry run python -m steps._04_cnn_standalone
poetry run python -m steps._04_lstm_standalone

# Business insights and seasonality analysis
poetry run python scripts/seasonality_business_insights.py
poetry run python scripts/seasonality_deployment_analysis.py

# Main ZenML pipeline
poetry run python pipelines/zenml_pipeline.py

# CNN+LSTM testing pipeline
poetry run python pipelines/zenml_pipeline_cnn_lstm_testing.py

# View MLflow experiments
mlflow ui
```

For detailed API documentation and guides, see the `docs/` directory.