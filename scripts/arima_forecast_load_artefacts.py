from zenml.client import Client
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
import numpy as np

# ------------------------------------------------------------------ #
# 1.  Grab the most recent run of the pipeline and load artefacts
# ------------------------------------------------------------------ #
def load_latest_run_outputs(pipeline_name: str):
    """
    Return a dict  {step_name -> {output_name -> python_object}}
    Compatible with zenml 0.84.0 (outputs are lists).
    """
    client = Client()

    # Most-recent run comes first
    runs = client.list_pipeline_runs(pipeline_name=pipeline_name, size=1)
    if not runs:
        raise ValueError(f"No runs found for pipeline '{pipeline_name}'")

    run = client.get_pipeline_run(runs[0].id)
    artefacts = {}

    for step_name, step_view in run.steps.items():
        artefacts[step_name] = {}
        for output_name, output_list in step_view.outputs.items():
            if not output_list:
                continue                       # empty output
            artefacts[step_name][output_name] = output_list[0].load()

    return artefacts, run


def list_available_artifacts(artefacts):
    """Helper function to list all available artifacts for debugging."""
    print("Available artifacts:")
    for step_name, step_artifacts in artefacts.items():
        print(f"  Step: {step_name}")
        for artifact_name, artifact_obj in step_artifacts.items():
            obj_type = type(artifact_obj).__name__
            if hasattr(artifact_obj, 'shape'):
                print(f"    - {artifact_name}: {obj_type} {artifact_obj.shape}")
            else:
                print(f"    - {artifact_name}: {obj_type}")


# ------------------------------------------------------------------ #
# 2.  Pull the objects we need (model, train/test, predictions)
# Updated for separate train_data and test_data artifacts
# ------------------------------------------------------------------ #
def extract_objects(artefacts, book_name: str | None = None):
    """Return train_series, test_series and the fitted SARIMA model."""
    # Find train_data and test_data DataFrames (separate artifacts now)
    train_df = None
    test_df = None
    
    for d in artefacts.values():
        for name, obj in d.items():
            if "train_data" in name and isinstance(obj, pd.DataFrame):
                train_df = obj
            elif "test_data" in name and isinstance(obj, pd.DataFrame):
                test_df = obj
    
    if train_df is None:
        raise RuntimeError("Could not locate 'train_data' artefact")
    if test_df is None:
        raise RuntimeError("Could not locate 'test_data' artefact")

    # choose book
    if not book_name:
        book_name = train_df["book_name"].unique()[0]

    # Filter by book and extract volume series
    book_train_df = train_df[train_df.book_name == book_name]
    book_test_df = test_df[test_df.book_name == book_name]
    
    if book_train_df.empty:
        raise RuntimeError(f"No training data found for book: {book_name}")
    if book_test_df.empty:
        raise RuntimeError(f"No test data found for book: {book_name}")
    
    # Create time series (data is already indexed by date from our pipeline)
    # If 'volume' column exists, use it; otherwise use 'Volume'
    volume_col = 'volume' if 'volume' in book_train_df.columns else 'Volume'
    
    train = book_train_df[volume_col].sort_index()
    test = book_test_df[volume_col].sort_index()

    # SARIMA model (output 'trained_model' of train_arima_optuna_step)
    model = None
    for d in artefacts.values():
        for name, obj in d.items():
            if "trained_model" in name:
                model = obj
                break
        else:
            continue
        break
    
    if model is None:
        raise RuntimeError("Could not find a trained model artefact")

    return train, test, model, book_name


# ------------------------------------------------------------------ #
# 2.5 Load ARIMA results from CSV files as fallback
# ------------------------------------------------------------------ #

def load_arima_results_from_csv(csv_dir="arima_standalone_outputs"):
    """Load ARIMA results from CSV files if ZenML artifacts are not available."""
    try:
        # Load CSV files
        forecast_comparison_path = os.path.join(csv_dir, "arima_forecast_comparison.csv")
        predictions_path = os.path.join(csv_dir, "arima_predictions.csv")
        residuals_path = os.path.join(csv_dir, "arima_residuals.csv")
        
        if not os.path.exists(forecast_comparison_path):
            raise FileNotFoundError(f"ARIMA forecast comparison CSV not found: {forecast_comparison_path}")
        
        forecast_df = pd.read_csv(forecast_comparison_path)
        predictions_df = pd.read_csv(predictions_path)
        
        # Convert date columns
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        print(f"✅ Loaded ARIMA results from CSV files:")
        print(f"   • Forecast comparison: {forecast_df.shape}")
        print(f"   • Predictions: {predictions_df.shape}")
        
        return forecast_df, predictions_df
        
    except Exception as e:
        print(f"❌ Failed to load ARIMA CSV files: {e}")
        return None, None

# ------------------------------------------------------------------ #
# 3.  Simple plot that uses the stored test-set predictions
# ------------------------------------------------------------------ #

def plot_test_predictions(artefacts, title="SARIMA test-set performance", save_path="outputs"):
    # load test_predictions DataFrame
    for d in artefacts.values():
        for name, obj in d.items():
            if "test_predictions" in name and isinstance(obj, pd.DataFrame):
                preds = obj
                break
        else:
            continue
        break
    else:
        raise RuntimeError("No 'test_predictions' artefact found")

    # Calculate metrics
    mae  = preds.absolute_error.mean()
    mape = (preds.absolute_error / preds.actual).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds.date, y=preds.actual,
                             mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=preds.date, y=preds.predicted,
                             mode="lines", name="Predicted", line=dict(color="red")))

    # Add MAE/MAPE as text annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"MAE: {mae:,.2f}<br>MAPE: {mape:,.2f}%",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Volume",
                      template="plotly_white", width=900, height=500)

    print(f"MAE:  {mae:,.2f}")
    print(f"MAPE: {mape:,.2f}%")

    # Save the plot
    filename = "test_predictions.html"
    filepath = os.path.join(save_path, filename)
    fig.write_html(filepath)
    print(f"Plot saved to: {filepath}")

    # fig.show()  # Commented out to prevent massive HTML output
    return fig

def plot_arima_from_csv(csv_dir="arima_standalone_outputs", save_path="outputs", 
                       title="ARIMA Standalone Forecast"):
    """Plot ARIMA results loaded from CSV files."""
    forecast_df, predictions_df = load_arima_results_from_csv(csv_dir)
    
    if forecast_df is None or predictions_df is None:
        print("❌ Could not load ARIMA CSV data for plotting")
        return None
    
    # Calculate metrics
    mae = forecast_df['absolute_error'].mean()
    mape = forecast_df['percentage_error'].mean()
    rmse = np.sqrt(forecast_df['squared_error'].mean())
    
    # Load training data for context
    train_path = "data/processed/combined_train_data.csv"
    if os.path.exists(train_path):
        train_data = pd.read_csv(train_path)
        train_data['End Date'] = pd.to_datetime(train_data['End Date'])
        train_dates = train_data['End Date']
        train_volumes = train_data['Volume']
    else:
        # Create placeholder training data
        train_dates = pd.date_range('2020-01-01', periods=500, freq='W-SAT')
        train_volumes = [500] * len(train_dates)
    
    # Create plot
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_volumes,
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    # Add actual test data
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['actual_volume'],
        mode='lines+markers',
        name='Actual Test Data',
        line=dict(color='black', width=3),
        marker=dict(size=5)
    ))
    
    # Add ARIMA predictions
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_volume'],
        mode='lines+markers',
        name='ARIMA Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Update layout with metrics
    title_text = f'{title}<br><sub>MAE: {mae:.2f} | MAPE: {mape:.2f}% | RMSE: {rmse:.2f}</sub>'
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        width=1200,
        height=500,
        showlegend=True
    )
    
    # Save files
    os.makedirs(save_path, exist_ok=True)
    html_filename = os.path.join(save_path, "arima_standalone_forecast.html")
    png_filename = os.path.join(save_path, "arima_standalone_forecast.png")
    
    fig.write_html(html_filename)
    fig.write_image(png_filename)
    
    print(f"📊 ARIMA CSV Results:")
    print(f"   • MAE: {mae:.2f}")
    print(f"   • MAPE: {mape:.2f}%")
    print(f"   • RMSE: {rmse:.2f}")
    print(f"📁 ARIMA plots saved:")
    print(f"   • HTML: {html_filename}")
    print(f"   • PNG: {png_filename}")
    
    return fig, {"mae": mae, "mape": mape, "rmse": rmse}

# ------------------------------------------------------------------ #
# 5.  CLI entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    PIPELINE_NAME = "book_sales_arima_pipeline"        # <- your pipeline
    BOOK          = None                               # or e.g. "The Alchemist"
    PLOTS_FOLDER  = "outputs"                          # outputs folder
    CSV_FOLDER    = "arima_standalone_outputs"         # CSV fallback folder

    # Ensure plots folder exists
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    print("🚀 Starting ARIMA forecast visualization...")
    print("=" * 60)

    # --------  Option 1: Try to load from CSV files first (most recent)  -------- #
    print("\n📋 Attempting to load ARIMA results from CSV files...")
    try:
        if os.path.exists(CSV_FOLDER):
            fig, metrics = plot_arima_from_csv(
                csv_dir=CSV_FOLDER, 
                save_path=PLOTS_FOLDER,
                title="ARIMA Standalone Forecast"
            )
            if fig is not None:
                print("✅ Successfully plotted ARIMA results from CSV files!")
                print("=" * 60)
                exit(0)  # Success, no need to try artifacts
        else:
            print(f"❌ CSV folder not found: {CSV_FOLDER}")
    except Exception as e:
        print(f"❌ Failed to plot from CSV: {e}")
    
    # --------  Option 2: Fallback to ZenML artifacts  -------- #
    print("\n📋 Falling back to ZenML artifacts...")
    try:
        artefacts, run = load_latest_run_outputs(PIPELINE_NAME)
        print(f"✅ Loaded artefacts from run: {run.name}")
        
        # List available artifacts for debugging
        list_available_artifacts(artefacts)

        # --------  Option A: use stored test-set predictions  -------- #
        try:
            plot_test_predictions(artefacts,
                title="SARIMA – test-set predictions (ZenML artifacts)",
                save_path=PLOTS_FOLDER)
            print("✅ Successfully plotted from ZenML artifacts!")
        except Exception as e:
            print(f"❌ Error plotting test predictions: {e}")
            print("This might be due to changes in artifact structure.")
            
        # --------  Option B: extract train/test data manually  -------- #
        try:
            print("\nExtracting train/test data from separate artifacts...")
            train, test, model, book_name = extract_objects(artefacts, BOOK)
            print(f"Successfully extracted data for book: {book_name}")
            print(f"Train data shape: {train.shape}")
            print(f"Test data shape: {test.shape}")
            print(f"Model type: {type(model).__name__}")
        except Exception as e:
            print(f"Error extracting objects: {e}")
            print("Please check that the pipeline has been run with the updated structure.")
            
    except Exception as e:
        print(f"❌ Failed to load ZenML artifacts: {e}")
        print("Please ensure the pipeline has been run or CSV files are available.")
    
    print("\n" + "=" * 60)
