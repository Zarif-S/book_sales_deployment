"""
Simple Plotting Functions for Streamlit App

Self-contained Plotly functions adapted from the main codebase plotting utilities.
These are simplified versions focused on the core forecasting visualization needs.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_historical_sales(df: pd.DataFrame, isbn: str = None, title: str = None) -> go.Figure:
    """
    Plot historical sales data for a specific book or all books.
    
    Adapted from utils/plotting.py plot_weekly_volume_by_isbn function.
    
    Args:
        df: DataFrame with sales data (columns: End Date, ISBN, Volume, Title)
        isbn: Specific ISBN to plot (if None, plots all)
        title: Custom plot title
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating historical sales plot for ISBN: {isbn}")
    
    # Ensure End Date is datetime and set as index
    df = df.copy()
    date_column = None
    
    # Check for different possible date column names
    for col in ['End Date', 'End_Date', 'date', 'Date']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    else:
        logger.warning(f"No date column found in data. Available columns: {list(df.columns)}")
        return go.Figure().add_annotation(text="No date column found", x=0.5, y=0.5)
    
    fig = go.Figure()
    
    if isbn:
        # Plot specific book
        if 'ISBN' in df.columns:
            # Check for exact matches and similar matches
            exact_matches = df[df['ISBN'] == isbn]
            
            # Check for string matches (in case of type mismatch)
            str_matches = df[df['ISBN'].astype(str) == str(isbn)]
            
            book_data = exact_matches if not exact_matches.empty else str_matches
        else:
            book_data = df
        
        if not book_data.empty:
            # Check if Volume column exists
            if 'Volume' not in book_data.columns:
                logger.error(f"Volume column not found. Available columns: {list(book_data.columns)}")
                return go.Figure().add_annotation(text=f"Volume column missing from data", x=0.5, y=0.5)
            
            # Get book title for display
            book_title = book_data['Title'].iloc[0] if 'Title' in book_data.columns else f"ISBN {isbn}"
            
            fig.add_trace(go.Scatter(
                x=book_data.index,
                y=book_data['Volume'],
                mode='lines+markers',
                name=book_title,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        else:
            return go.Figure().add_annotation(text=f"No data found for ISBN {isbn}", x=0.5, y=0.5)
    else:
        # Plot all books
        if 'ISBN' in df.columns:
            for current_isbn in df['ISBN'].unique():
                book_data = df[df['ISBN'] == current_isbn]
                if not book_data.empty:
                    # Get book title for legend
                    book_title = book_data['Title'].iloc[0] if 'Title' in book_data.columns else f"ISBN {current_isbn}"
                    
                    fig.add_trace(go.Scatter(
                        x=book_data.index,
                        y=book_data['Volume'],
                        mode='lines+markers',
                        name=book_title,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
    
    # Update layout
    plot_title = title or f"Historical Sales - {book_title if isbn else 'All Books'}"
    fig.update_layout(
        title=dict(text=plot_title, x=0.5),
        xaxis_title='Date',
        yaxis_title='Weekly Sales Volume',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Format x-axis
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    return fig


def plot_forecast_with_historical(historical_df: pd.DataFrame, predictions: List[Dict], 
                                isbn: str, title: str = None) -> go.Figure:
    """
    Plot historical data with forecast predictions.
    
    Args:
        historical_df: Historical sales data
        predictions: List of prediction dictionaries with 'target_date' and 'predicted_sales'
        isbn: Book ISBN
        title: Custom plot title
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating forecast plot for ISBN: {isbn}")
    
    # Prepare historical data
    historical_df = historical_df.copy()
    date_column = None
    
    # Check for different possible date column names
    for col in ['End Date', 'End_Date', 'date', 'Date']:
        if col in historical_df.columns:
            date_column = col
            break
    
    if date_column:
        historical_df[date_column] = pd.to_datetime(historical_df[date_column])
        historical_df = historical_df.set_index(date_column)
    
    # Filter for specific book
    if 'ISBN' in historical_df.columns:
        book_data = historical_df[historical_df['ISBN'] == isbn]
    else:
        book_data = historical_df
    
    # Get book title
    book_title = "Unknown Book"
    if not book_data.empty and 'Title' in book_data.columns:
        book_title = book_data['Title'].iloc[0]
    
    fig = go.Figure()
    
    # Plot historical data
    if not book_data.empty:
        fig.add_trace(go.Scatter(
            x=book_data.index,
            y=book_data['Volume'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='blue', width=2),
            marker=dict(size=4, color='blue')
        ))
    
    # Plot predictions
    if predictions:
        pred_dates = [pd.to_datetime(p['target_date']) for p in predictions]
        pred_values = [p['predicted_sales'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6, color='red')
        ))
        
        # Add confidence intervals if available (mock for now)
        confidence_upper = [v * 1.2 for v in pred_values]  # 20% confidence band
        confidence_lower = [v * 0.8 for v in pred_values]
        
        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1],
            y=confidence_upper + confidence_lower[::-1],
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            hoverinfo='skip'
        ))
    
    # Update layout
    plot_title = title or f"Sales Forecast - {book_title}"
    fig.update_layout(
        title=dict(text=plot_title, x=0.5),
        xaxis_title='Date',
        yaxis_title='Weekly Sales Volume',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add vertical line at forecast start
    if predictions and not book_data.empty:
        last_historical = book_data.index.max()
        first_forecast = pd.to_datetime(predictions[0]['target_date'])
        
        fig.add_vline(
            x=last_historical,
            line_width=2,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start"
        )
    
    # Format axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig


def plot_single_prediction(prediction: Dict, title: str = None) -> go.Figure:
    """
    Create a simple display for a single prediction result.
    
    Args:
        prediction: Prediction dictionary
        title: Custom plot title
        
    Returns:
        Plotly figure object
    """
    logger.info("Creating single prediction display")
    
    # Create a simple gauge/indicator chart
    fig = go.Figure()
    
    predicted_value = prediction.get('predicted_sales', 0)
    book_title = prediction.get('title', 'Unknown Book')
    target_date = prediction.get('target_date', 'Unknown Date')
    
    # Create a bar chart showing the prediction
    fig.add_trace(go.Bar(
        x=[f"{book_title}<br>{target_date}"],
        y=[predicted_value],
        text=[f"{predicted_value:.1f} units"],
        textposition='auto',
        marker_color='lightblue',
        name='Predicted Sales'
    ))
    
    plot_title = title or f"Sales Prediction - {book_title}"
    fig.update_layout(
        title=dict(text=plot_title, x=0.5),
        xaxis_title='',
        yaxis_title='Predicted Sales Volume',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    # Format y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig


def plot_metrics_comparison(metrics: Dict[str, float], title: str = "Model Performance Metrics") -> go.Figure:
    """
    Create a simple metrics visualization.
    
    Args:
        metrics: Dictionary of metrics (e.g., {'MAE': 45.2, 'RMSE': 67.8, 'MAPE': 12.5})
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    logger.info("Creating metrics comparison plot")
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        text=[f"{v:.2f}" for v in metric_values],
        textposition='auto',
        marker_color=['lightcoral', 'lightblue', 'lightgreen'][:len(metrics)],
        name='Metrics'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Metrics',
        yaxis_title='Value',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig


def create_summary_table(predictions: List[Dict]) -> pd.DataFrame:
    """
    Create a summary table for multiple predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        DataFrame formatted for display
    """
    if not predictions:
        return pd.DataFrame()
    
    summary_data = []
    for pred in predictions:
        summary_data.append({
            'Date': pred.get('target_date', 'N/A'),
            'Book': pred.get('title', 'Unknown'),
            'Predicted Sales': f"{pred.get('predicted_sales', 0):.1f}",
            'Weeks Ahead': pred.get('forecast_steps', 'N/A'),
            'Type': pred.get('prediction_type', 'unknown')
        })
    
    return pd.DataFrame(summary_data)


# Utility functions for data formatting
def format_prediction_text(prediction: Dict) -> str:
    """Format prediction for text display."""
    book_title = prediction.get('title', 'Unknown Book')
    predicted_value = prediction.get('predicted_sales', 0)
    target_date = prediction.get('target_date', 'Unknown Date')
    pred_type = prediction.get('prediction_type', 'unknown')
    
    confidence = prediction.get('confidence_level', 'Standard confidence')
    
    text = f"""
    📚 **{book_title}**
    📅 **Target Date:** {target_date}
    📊 **Predicted Sales:** {predicted_value:.1f} units
    🔮 **Prediction Type:** {pred_type.title()}
    📈 **Confidence:** {confidence}
    """
    
    if pred_type == 'mock':
        text += "\n⚠️ *Note: This is a mock prediction for development purposes*"
    
    return text.strip()


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'End Date': pd.date_range('2023-01-01', periods=52, freq='W'),
        'ISBN': '9780722532935',
        'Title': 'The Alchemist',
        'Volume': np.random.randint(300, 600, 52)
    })
    
    # Test historical plot
    fig1 = plot_historical_sales(sample_data, '9780722532935')
    print("Historical plot created successfully")
    
    # Test prediction
    sample_prediction = {
        'title': 'The Alchemist',
        'target_date': '2024-06-01',
        'predicted_sales': 450.5,
        'prediction_type': 'mock'
    }
    
    fig2 = plot_single_prediction(sample_prediction)
    print("Prediction plot created successfully")