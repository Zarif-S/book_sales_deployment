from zenml.client import Client
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
import numpy as np
import warnings
import mlflow
import mlflow.statsmodels

# Suppress specific statsmodels warnings about unsupported index
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels.tsa.base.tsa_model', message='No supported index is available.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels.tsa.base.tsa_model', message='No supported index is available.*')

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
def extract_objects(artefacts, book_isbn: str | None = None):
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

    # choose book by ISBN
    if not book_isbn:
        book_isbn = train_df["ISBN"].unique()[0]

    # Filter by ISBN and extract volume series
    book_train_df = train_df[train_df.ISBN == book_isbn]
    book_test_df = test_df[test_df.ISBN == book_isbn]

    if book_train_df.empty:
        raise RuntimeError(f"No training data found for ISBN: {book_isbn}")
    if book_test_df.empty:
        raise RuntimeError(f"No test data found for ISBN: {book_isbn}")

    # Create time series (data is already indexed by date from our pipeline)
    # Use Volume column from consolidated artifacts
    volume_col = 'Volume'
    if volume_col not in book_train_df.columns:
        raise RuntimeError(f"Volume column not found. Available columns: {list(book_train_df.columns)}")

    # Ensure we have proper datetime index for both train and test
    if 'End Date' in book_train_df.columns:
        book_train_df['End Date'] = pd.to_datetime(book_train_df['End Date'])
        book_train_df = book_train_df.set_index('End Date')
        book_test_df['End Date'] = pd.to_datetime(book_test_df['End Date'])
        book_test_df = book_test_df.set_index('End Date')
    
    train = book_train_df[volume_col].sort_index()
    test = book_test_df[volume_col].sort_index()
    
    # Get book title for display
    book_title = book_train_df["Title"].iloc[0] if "Title" in book_train_df.columns else f"Book_{book_isbn}"

    # SARIMA model - load from saved file since models are saved to disk, not as artifacts
    model = None
    
    # First, get the training results to find the model path
    arima_results = None
    for d in artefacts.values():
        for name, obj in d.items():
            if "arima_training_results" in name and isinstance(obj, dict):
                arima_results = obj
                break
    
    if arima_results and 'book_results' in arima_results:
        book_results = arima_results['book_results']
        if book_isbn in book_results:
            book_result = book_results[book_isbn]
            if 'model_path' in book_result and 'error' not in book_result:
                model_path = book_result['model_path']
                try:
                    # Try loading as MLflow model first
                    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'MLmodel')):
                        model = mlflow.statsmodels.load_model(model_path)
                        print(f"✅ Loaded MLflow model from: {model_path}")
                        
                        # Load model metadata to restore datetime context
                        model_datetime_context = None
                        try:
                            import mlflow.models
                            model_metadata = mlflow.models.Model.load(model_path).metadata
                            if model_metadata:
                                train_freq = model_metadata.get('train_freq')
                                train_start = model_metadata.get('train_start') 
                                train_end = model_metadata.get('train_end')
                                train_length = model_metadata.get('training_data_length')
                                
                                if train_start and train_end and train_length:
                                    # Store datetime context for prediction
                                    model_datetime_context = {
                                        'train_start': pd.to_datetime(train_start),
                                        'train_end': pd.to_datetime(train_end),
                                        'train_length': train_length,
                                        'train_freq': train_freq
                                    }
                                    print(f"✅ Restored model datetime context: {train_start} to {train_end} ({train_length} points)")
                        except Exception as meta_e:
                            print(f"⚠️ Could not load model metadata: {meta_e}")
                            
                        # Store the datetime context in the model object for later use
                        if model_datetime_context:
                            model._mlflow_datetime_context = model_datetime_context
                            
                            # Try to reconstruct the model's datetime context from orig_endog or metadata
                            try:
                                # Check if orig_endog already has datetime index (best case)
                                if hasattr(model, 'data') and hasattr(model.data, 'orig_endog') and model.data.orig_endog is not None:
                                    orig_endog = model.data.orig_endog
                                    if hasattr(orig_endog, 'index') and pd.api.types.is_datetime64_any_dtype(orig_endog.index):
                                        # Use the existing datetime index from orig_endog
                                        model.data.dates = orig_endog.index
                                        model.data.freq = orig_endog.index.freq
                                        print(f"✅ Restored datetime context from orig_endog: {orig_endog.index[0]} to {orig_endog.index[-1]}")
                                        print(f"  Frequency: {orig_endog.index.freq}")
                                    else:
                                        # Fallback: reconstruct from metadata
                                        train_length = model_datetime_context['train_length']
                                        train_start = model_datetime_context['train_start']
                                        train_end = model_datetime_context['train_end']
                                        datetime_index = pd.date_range(start=train_start, end=train_end, periods=train_length)
                                        model.data.dates = datetime_index
                                        model.data.freq = datetime_index.freq
                                        print(f"✅ Reconstructed datetime context from metadata: {datetime_index[0]} to {datetime_index[-1]}")
                                        
                            except Exception as idx_e:
                                print(f"⚠️ Could not reconstruct model datetime context: {idx_e}")
                            
                    else:
                        # Fallback to pickle for compatibility
                        if model_path.endswith('.pkl'):
                            import pickle
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            print(f"✅ Loaded pickle model from: {model_path}")
                        else:
                            raise FileNotFoundError(f"Model file not found or unrecognized format: {model_path}")
                            
                except Exception as e:
                    print(f"❌ Failed to load model from {model_path}: {e}")
                    raise RuntimeError(f"Could not load model from {model_path}: {e}")
            else:
                raise RuntimeError(f"Model training failed for ISBN {book_isbn}: {book_result.get('error', 'Unknown error')}")
        else:
            raise RuntimeError(f"No training results found for ISBN {book_isbn}")
    else:
        raise RuntimeError("Could not find ARIMA training results in artifacts")

    if model is None:
        raise RuntimeError("Could not load trained model")

    return train, test, model, book_isbn, book_title


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

def get_selected_isbns_from_artifacts(artefacts):
    """Extract the selected ISBNs that were used in the pipeline."""
    # First try to find selected_isbns artifact
    for d in artefacts.values():
        for name, obj in d.items():
            if "selected_isbns" in name and isinstance(obj, list):
                return obj
    
    # If not found, extract from arima_training_results
    for d in artefacts.values():
        for name, obj in d.items():
            if "arima_training_results" in name and isinstance(obj, dict):
                book_results = obj.get('book_results', {})
                if book_results:
                    return list(book_results.keys())
    
    # Fallback: extract unique ISBNs from train_data
    for d in artefacts.values():
        for name, obj in d.items():
            if "train_data" in name and hasattr(obj, 'columns') and 'ISBN' in obj.columns:
                return obj['ISBN'].unique().tolist()
    
    return None

def plot_test_predictions(artefacts, book_isbn=None, title="SARIMA test-set performance", save_path="outputs/plots/interactive"):
    # Extract train/test data and model for the specific book
    try:
        train, test, model, isbn, book_title = extract_objects(artefacts, book_isbn)
        
        # Generate predictions using the model with proper datetime context
        if hasattr(model, '_mlflow_datetime_context') and model._mlflow_datetime_context:
            # Use MLflow metadata to create proper datetime context for predictions
            datetime_ctx = model._mlflow_datetime_context
            train_start = datetime_ctx['train_start']
            train_length = datetime_ctx['train_length']
            
            # Create proper datetime index for predictions
            # The test period starts right after training period
            test_start_idx = train_length
            test_end_idx = test_start_idx + len(test) - 1
            
            # For SARIMA with datetime context, we can use the proper indices
            print(f"🕐 Using datetime context: train_length={train_length}, test_period={test_start_idx}-{test_end_idx}")
            
            try:
                # Use get_prediction with the proper indices based on training length
                forecast_result = model.get_prediction(start=test_start_idx, end=test_end_idx, dynamic=False)
                test_predictions = forecast_result.predicted_mean
                print(f"✅ Predictions made with datetime context")
            except Exception as e:
                print(f"⚠️ Datetime context prediction failed: {e}, falling back to forecast()")
                # Fallback to forecast method which is more robust
                test_predictions = model.forecast(steps=len(test))
        else:
            # Fallback to original logic if no MLflow context available
            print(f"⚠️ No MLflow datetime context available, using fallback prediction")
            combined_series = pd.concat([train, test])
            
            # Use the datetime index from the combined series for predictions
            start_date = test.index[0] if hasattr(test, 'index') and len(test) > 0 else len(train)
            end_date = test.index[-1] if hasattr(test, 'index') and len(test) > 0 else len(train) + len(test) - 1
            
            # Use get_prediction with proper datetime indexing
            try:
                forecast_result = model.get_prediction(start=start_date, end=end_date, dynamic=False)
                test_predictions = forecast_result.predicted_mean
            except (KeyError, ValueError) as e:
                # Fallback to integer indexing if datetime indexing fails
                start_idx = len(train)
                end_idx = start_idx + len(test) - 1
                forecast_result = model.get_prediction(start=start_idx, end=end_idx, dynamic=False)
                test_predictions = forecast_result.predicted_mean
        
        # Create a predictions DataFrame with proper date indexing
        if hasattr(test, 'index') and pd.api.types.is_datetime64_any_dtype(test.index):
            # Use the test data's date index
            pred_index = test.index
        else:
            # Create a simple integer index if no proper date index exists
            pred_index = range(len(test))
        
        preds = pd.DataFrame({
            'date': pred_index,
            'actual': test.values,
            'predicted': test_predictions.values
        })
        preds['absolute_error'] = abs(preds['actual'] - preds['predicted'])
        
        # Calculate metrics
        mae  = preds.absolute_error.mean()
        mape = (preds.absolute_error / preds.actual).mean() * 100
        
        # Update title with book info
        title = f"SARIMA test-set performance - {book_title} (ISBN: {isbn})"
        
    except Exception as e:
        print(f"Error extracting model data: {e}")
        # Fallback: load test_predictions DataFrame if available
        for d in artefacts.values():
            for name, obj in d.items():
                if "test_predictions" in name and isinstance(obj, pd.DataFrame):
                    preds = obj
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("No 'test_predictions' artefact found and could not generate predictions")

        # Calculate metrics from existing predictions
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
    os.makedirs(save_path, exist_ok=True)
    # Use book-specific filename if ISBN is available
    if book_isbn:
        filename = f"arima_test_predictions_{book_isbn}.html"
    else:
        filename = "arima_test_predictions.html"
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
    PIPELINE_NAME = "book_sales_arima_modeling_pipeline"  # <- your pipeline
    PLOTS_FOLDER  = "outputs/plots/interactive"        # interactive plots folder
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
        # Use the latest run (which should have the datetime index fixes)
        artefacts, run = load_latest_run_outputs(PIPELINE_NAME)
        print(f"✅ Loaded artefacts from run: {run.name}")
        print(f"📅 Run executed: {run.created}")

        # List available artifacts for debugging
        list_available_artifacts(artefacts)

        # --------  Get selected ISBNs from pipeline  -------- #
        selected_isbns = get_selected_isbns_from_artifacts(artefacts)
        if selected_isbns:
            print(f"\n📚 Found {len(selected_isbns)} books from pipeline: {selected_isbns}")
            
            # Generate predictions for each book
            for i, isbn in enumerate(selected_isbns):
                print(f"\n--- Processing book {i+1}/{len(selected_isbns)}: ISBN {isbn} ---")
                
                try:
                    # Extract data for this specific book
                    train, test, model, book_isbn, book_title = extract_objects(artefacts, isbn)
                    print(f"📖 Book: {book_title} (ISBN: {book_isbn})")
                    print(f"📊 Train data shape: {train.shape}")
                    print(f"📊 Test data shape: {test.shape}")
                    print(f"🤖 Model type: {type(model).__name__}")
                    
                    # Plot predictions for this book
                    plot_test_predictions(artefacts, 
                        book_isbn=isbn,
                        title=f"SARIMA – {book_title} predictions (ZenML artifacts)",
                        save_path=PLOTS_FOLDER)
                    print(f"✅ Successfully plotted predictions for {book_title}!")
                    
                except Exception as e:
                    print(f"❌ Error processing ISBN {isbn}: {e}")
                    continue
                    
        else:
            print("⚠️  Could not find selected ISBNs from pipeline artifacts")
            # Fallback: try to extract any available book data
            try:
                train, test, model, book_isbn, book_title = extract_objects(artefacts)
                print(f"📖 Fallback: Found data for {book_title} (ISBN: {book_isbn})")
                plot_test_predictions(artefacts,
                    book_isbn=book_isbn,
                    title=f"SARIMA – {book_title} predictions (ZenML artifacts)",
                    save_path=PLOTS_FOLDER)
                print("✅ Successfully plotted fallback predictions!")
            except Exception as e:
                print(f"❌ Error with fallback extraction: {e}")

    except Exception as e:
        print(f"❌ Failed to load ZenML artifacts: {e}")
        print("Please ensure the pipeline has been run or CSV files are available.")

    print("\n" + "=" * 60)
