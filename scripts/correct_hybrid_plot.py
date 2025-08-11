#!/usr/bin/env python3
"""
Correct CNN+LSTM hybrid plot using same data source as ARIMA script
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from zenml.client import Client

def load_latest_run_outputs(pipeline_name: str):
    """Load artifacts from latest pipeline run - same as ARIMA script"""
    client = Client()
    runs = client.list_pipeline_runs(pipeline_name=pipeline_name, size=1)
    if not runs:
        raise ValueError(f"No runs found for pipeline '{pipeline_name}'")

    run = client.get_pipeline_run(runs[0].id)
    artefacts = {}

    for step_name, step_view in run.steps.items():
        artefacts[step_name] = {}
        for output_name, output_list in step_view.outputs.items():
            if not output_list:
                continue
            artefacts[step_name][output_name] = output_list[0].load()

    return artefacts, run

def extract_train_test_data(artefacts, book_isbn: str = "9780722532935"):
    """Extract train/test data exactly like ARIMA script does"""
    # Find train_data and test_data DataFrames
    train_df = None
    test_df = None

    for d in artefacts.values():
        for name, obj in d.items():
            if "train_data" in name and isinstance(obj, pd.DataFrame):
                train_df = obj
            elif "test_data" in name and isinstance(obj, pd.DataFrame):
                test_df = obj

    if train_df is None or test_df is None:
        raise RuntimeError("Could not locate train_data or test_data artifacts")

    # Filter by ISBN and extract volume series
    book_train_df = train_df[train_df.ISBN == book_isbn]
    book_test_df = test_df[test_df.ISBN == book_isbn]

    if book_train_df.empty or book_test_df.empty:
        raise RuntimeError(f"No data found for ISBN: {book_isbn}")

    # Set up datetime index
    if 'End Date' in book_train_df.columns:
        book_train_df['End Date'] = pd.to_datetime(book_train_df['End Date'])
        book_train_df = book_train_df.set_index('End Date')
        book_test_df['End Date'] = pd.to_datetime(book_test_df['End Date'])
        book_test_df = book_test_df.set_index('End Date')
    
    train_series = book_train_df['Volume'].sort_index()
    test_series = book_test_df['Volume'].sort_index()
    
    book_title = book_train_df["Title"].iloc[0] if "Title" in book_train_df.columns else f"Book_{book_isbn}"
    
    return train_series, test_series, book_title

def create_correct_hybrid_plot():
    print("🚀 Creating correct CNN+LSTM hybrid plot using ZenML artifacts...")
    
    try:
        # Load artifacts from latest pipeline run
        PIPELINE_NAME = "book_sales_arima_modeling_pipeline" 
        artefacts, run = load_latest_run_outputs(PIPELINE_NAME)
        print(f"✅ Loaded artifacts from run: {run.name}")
        
        # Extract train/test data using same method as ARIMA script
        train_series, test_series, book_title = extract_train_test_data(artefacts)
        print(f"📖 Book: {book_title}")
        print(f"📊 Train data: {len(train_series)} points ({train_series.index[0]} to {train_series.index[-1]})")
        print(f"📊 Test data: {len(test_series)} points ({test_series.index[0]} to {test_series.index[-1]})")
        
    except Exception as e:
        print(f"⚠️  Could not load from ZenML artifacts: {e}")
        print("📋 Falling back to CSV files...")
        
        # Fallback to CSV files
        train_data_df = pd.read_csv("data/processed/combined_train_data.csv")
        train_data_df['End Date'] = pd.to_datetime(train_data_df['End Date'])
        train_data_df = train_data_df.set_index('End Date')
        train_series = train_data_df['Volume'].copy()
        
        test_data_df = pd.read_csv("data/processed/combined_test_data.csv")
        test_data_df['End Date'] = pd.to_datetime(test_data_df['End Date'])
        test_data_df = test_data_df.set_index('End Date')
        test_series = test_data_df['Volume'].copy()
        
        book_title = "The Alchemist"
        print(f"📖 Book: {book_title} (fallback)")
        print(f"📊 Train data: {len(train_series)} points")
        print(f"📊 Test data: {len(test_series)} points")

    # Load hybrid results
    hybrid_data = pd.read_csv("outputs/data/comparisons/cnn_lstm_hybrid_comparison.csv", index_col=0)
    
    # Align lengths 
    min_length = min(len(test_series), len(hybrid_data))
    hybrid_forecast = hybrid_data['Hybrid_Model'].values[:min_length]
    first_model_forecast = hybrid_data['First_Model_Only'].values[:min_length]
    test_series_aligned = test_series.iloc[:min_length]
    
    print(f"📊 Using {min_length} aligned data points for comparison")
    
    # Calculate metrics
    hybrid_mae = mean_absolute_error(test_series_aligned.values, hybrid_forecast)
    hybrid_mape = np.mean(np.abs((test_series_aligned.values - hybrid_forecast) / test_series_aligned.values)) * 100
    hybrid_rmse = np.sqrt(mean_squared_error(test_series_aligned.values, hybrid_forecast))
    
    # Create the plot using same structure as ARIMA script
    fig = go.Figure()
    
    # Add training data (full series from artifacts)
    fig.add_trace(go.Scatter(
        x=train_series.index, 
        y=train_series.values,
        mode='lines', 
        name='Training Data',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    # Add actual test data  
    fig.add_trace(go.Scatter(
        x=test_series_aligned.index, 
        y=test_series_aligned.values,
        mode='lines+markers', 
        name='Actual Test Data',
        line=dict(color='black', width=3),
        marker=dict(size=5)
    ))
    
    # Add hybrid predictions
    fig.add_trace(go.Scatter(
        x=test_series_aligned.index, 
        y=hybrid_forecast,
        mode='lines+markers', 
        name='Hybrid Model',
        line=dict(color='green', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Add first model predictions  
    fig.add_trace(go.Scatter(
        x=test_series_aligned.index, 
        y=first_model_forecast,
        mode='lines', 
        name='First Model Only',
        line=dict(color='orange', width=2),
        opacity=0.8
    ))
    
    # Layout with metrics - same style as ARIMA script
    title_text = f'Book Sales Forecast - CNN + LSTM Hybrid Book Sales Forecast<br><sub>MAE: {hybrid_mae:.2f} | MAPE: {hybrid_mape:.2f}% | RMSE: {hybrid_rmse:.2f}</sub>'
    
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
    
    # Save using same paths as other plots
    fig.write_html("outputs/plots/interactive/CNN_plus_LSTM_Hybrid_Book_Sales_Forecast_forecast_comparison.html")
    
    print("✅ Correct hybrid plot saved!")
    print(f"📊 Metrics - MAE: {hybrid_mae:.2f}, MAPE: {hybrid_mape:.2f}%, RMSE: {hybrid_rmse:.2f}")
    print(f"📁 Saved to: outputs/plots/interactive/CNN_plus_LSTM_Hybrid_Book_Sales_Forecast_forecast_comparison.html")
    
    return fig

if __name__ == "__main__":
    create_correct_hybrid_plot()