"""
Plotting utilities for book sales analysis.

This module contains all plotting functions for visualizing book sales data,
separated from the data processing logic for better code organization.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_weekly_volume_by_isbn(df: pd.DataFrame, title: str = "Weekly Volume by ISBN", 
                              width: int = 900) -> go.Figure:
    """
    Plot weekly volume data for each ISBN.
    
    Args:
        df: DataFrame with sales data indexed by date
        title: Plot title
        width: Plot width in pixels
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating weekly volume plot for {df['ISBN'].nunique()} ISBNs")
    
    fig = go.Figure()
    
    for isbn in df['ISBN'].unique():
        current_data = df[df['ISBN'] == isbn]
        if not current_data.empty:
            fig.add_trace(go.Scatter(
                x=current_data.index,
                y=current_data['Volume'],
                mode='lines+markers',
                name=str(isbn)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='End Date',
        yaxis_title='Volume',
        legend_title='ISBN',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        width=width
    )
    
    return fig


def plot_yearly_volume_by_isbn(df: pd.DataFrame, title: str = "Yearly Volume by ISBN",
                              width: int = 900) -> go.Figure:
    """
    Plot yearly aggregated volume data for each ISBN.
    
    Args:
        df: DataFrame with sales data indexed by date
        title: Plot title
        width: Plot width in pixels
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating yearly volume plot for {df['ISBN'].nunique()} ISBNs")
    
    # Aggregate to yearly data
    yearly_data = df.groupby([df.index.year, 'ISBN'])['Volume'].sum().reset_index()
    yearly_data.rename(columns={'End Date': 'Year'}, inplace=True)
    
    fig = go.Figure()
    
    for isbn in yearly_data['ISBN'].unique():
        isbn_data = yearly_data[yearly_data['ISBN'] == isbn]
        fig.add_trace(go.Scatter(
            x=isbn_data['Year'],
            y=isbn_data['Volume'],
            mode='lines+markers',
            name=str(isbn)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Volume',
        yaxis=dict(tickformat=','),
        legend_title='ISBN',
        xaxis_tickangle=45,
        template='plotly_white',
        width=width
    )
    
    return fig


def plot_selected_books_weekly(df: pd.DataFrame, isbn_to_title: Dict[str, str],
                              title: str = "Weekly Sales Data for Selected Books",
                              width: int = 1500) -> go.Figure:
    """
    Plot weekly sales data for selected books.
    
    Args:
        df: DataFrame with sales data for selected books
        isbn_to_title: Dictionary mapping ISBNs to book titles
        title: Plot title
        width: Plot width in pixels
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating weekly plot for {len(isbn_to_title)} selected books")
    
    fig = go.Figure()
    
    for isbn in df['ISBN'].unique():
        isbn_data = df[df['ISBN'] == isbn]
        
        fig.add_trace(go.Scatter(
            x=isbn_data.index,
            y=isbn_data['Volume'],
            mode='lines+markers',
            name=f"{str(isbn)} - {isbn_to_title.get(str(isbn), 'Unknown')}"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales Volume',
        yaxis_tickformat=',',
        legend_title='ISBN - Book Title',
        template='plotly_white',
        width=width
    )
    
    return fig


def plot_selected_books_yearly(df: pd.DataFrame, isbn_to_title: Dict[str, str],
                              title: str = "Yearly Sales Data for Selected Books",
                              width: int = 1500) -> go.Figure:
    """
    Plot yearly aggregated sales data for selected books.
    
    Args:
        df: DataFrame with sales data for selected books
        isbn_to_title: Dictionary mapping ISBNs to book titles
        title: Plot title
        width: Plot width in pixels
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating yearly plot for {len(isbn_to_title)} selected books")
    
    # Aggregate to yearly data
    yearly_volume = df.groupby([df.index.year, 'ISBN'])['Volume'].sum().reset_index()
    
    fig = go.Figure()
    
    for isbn in yearly_volume['ISBN'].unique():
        isbn_data = yearly_volume[yearly_volume['ISBN'] == isbn]
        fig.add_trace(go.Scatter(
            x=isbn_data['End Date'],
            y=isbn_data['Volume'],
            mode='lines+markers',
            name=f"{str(isbn)} - {isbn_to_title.get(str(isbn), 'Unknown')}"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Sales Volume',
        yaxis_tickformat=',',
        legend_title='ISBN - Book Title',
        template='plotly_white',
        width=width
    )
    
    return fig


def plot_sales_comparison(df: pd.DataFrame, period1_start: str, period1_end: str,
                         period2_start: str, period2_end: str,
                         title: str = "Sales Comparison Between Periods") -> Tuple[go.Figure, go.Figure]:
    """
    Create comparison plots for two different time periods.
    
    Args:
        df: DataFrame with sales data
        period1_start: Start date for first period
        period1_end: End date for first period
        period2_start: Start date for second period
        period2_end: End date for second period
        title: Base title for plots
        
    Returns:
        Tuple of (period1_figure, period2_figure)
    """
    logger.info("Creating sales comparison plots for two periods")
    
    # Filter data for each period
    period1_data = df[(df.index >= period1_start) & (df.index <= period1_end)]
    period2_data = df[(df.index >= period2_start) & (df.index <= period2_end)]
    
    # Create yearly aggregated data for each period
    period1_yearly = period1_data.groupby([period1_data.index.year, 'ISBN'])['Volume'].sum().reset_index()
    period2_yearly = period2_data.groupby([period2_data.index.year, 'ISBN'])['Volume'].sum().reset_index()
    
    # Create plots
    fig1 = plot_yearly_volume_by_isbn(
        period1_data, 
        f"{title} - Period 1 ({period1_start} to {period1_end})"
    )
    
    fig2 = plot_yearly_volume_by_isbn(
        period2_data, 
        f"{title} - Period 2 ({period2_start} to {period2_end})"
    )
    
    return fig1, fig2


def plot_sales_trends(df: pd.DataFrame, isbn_list: List[str], 
                     isbn_to_title: Dict[str, str],
                     title: str = "Sales Trends Analysis") -> go.Figure:
    """
    Create a comprehensive sales trends plot for multiple books.
    
    Args:
        df: DataFrame with sales data
        isbn_list: List of ISBNs to plot
        isbn_to_title: Dictionary mapping ISBNs to book titles
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating sales trends plot for {len(isbn_list)} books")
    
    # Ensure ISBN column is string for robust comparison
    df = df.copy()
    df['ISBN'] = df['ISBN'].astype(str)
    
    fig = go.Figure()
    
    for isbn in isbn_list:
        isbn_str = str(isbn)
        if isbn_str in df['ISBN'].values:
            isbn_data = df[df['ISBN'] == isbn_str]
            
            # Add weekly data
            fig.add_trace(go.Scatter(
                x=isbn_data.index,
                y=isbn_data['Volume'],
                mode='lines+markers',
                name=f"{isbn_str} - {isbn_to_title.get(isbn_str, 'Unknown')} (Weekly)",
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sales Volume',
        yaxis_tickformat=',',
        legend_title='Book',
        template='plotly_white',
        width=1200,
        height=600
    )
    
    return fig


def create_summary_dashboard(df: pd.DataFrame, isbn_to_title: Dict[str, str], max_books: int = 6) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple subplots.
    
    Args:
        df: DataFrame with sales data
        isbn_to_title: Dictionary mapping ISBNs to book titles
        max_books: Maximum number of books to display (default: 6)
        
    Returns:
        Plotly figure with subplots
    """
    logger.info("Creating summary dashboard")
    
    from plotly.subplots import make_subplots
    
    # Use all available ISBNs (no filtering)
    all_isbns = df['ISBN'].unique()
    selected_isbns = list(all_isbns)
    
    logger.info(f"Displaying dashboard for all {len(selected_isbns)} books")
    
    # Use the full dataframe (no filtering needed)
    df_filtered = df.copy()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weekly Sales Volume (All Books)', 'Yearly Total Sales Volume', 
                       'Top 10 Books by Total Sales', 'Sales Distribution (Volume > 0)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Subplot 1: Weekly sales volume (no legend for readability)
    for isbn in selected_isbns:
        isbn_data = df_filtered[df_filtered['ISBN'] == isbn]
        isbn_str = str(isbn)
        title = isbn_to_title.get(isbn_str, 'Unknown')
        if not isbn_data.empty:
            fig.add_trace(
                go.Scatter(x=isbn_data.index, y=isbn_data['Volume'],
                          mode='lines', name=f"{isbn_str} - {title}", 
                          showlegend=False),  # Hide legend for line chart
                row=1, col=1
            )
    
    # Subplot 2: Yearly total sales volume (aggregated across all books)
    yearly_total = df_filtered.groupby(df_filtered.index.year)['Volume'].sum().reset_index()
    yearly_total.rename(columns={'End Date': 'Year'}, inplace=True)
    
    fig.add_trace(
        go.Bar(x=yearly_total['Year'], y=yearly_total['Volume'],
               name='Total Sales Volume', showlegend=False),
        row=1, col=2
    )
    
    # Subplot 3: Sales by book (pie chart) - Show only top 10 books
    total_sales_by_book = df_filtered.groupby('ISBN')['Volume'].sum().sort_values(ascending=False)
    
    # Get top 10 books and group the rest as "Others"
    top_n = 10
    top_books = total_sales_by_book.head(top_n)
    others_sum = total_sales_by_book.iloc[top_n:].sum()
    
    labels = []
    values = []
    
    # Add top books
    for isbn, volume in top_books.items():
        isbn_str = str(isbn)
        title = isbn_to_title.get(isbn_str, 'Unknown')
        labels.append(f"{isbn_str} - {title}")
        values.append(volume)
    
    # Add "Others" category if there are more than top_n books
    if others_sum > 0:
        labels.append(f"Others ({len(total_sales_by_book) - top_n} books)")
        values.append(others_sum)
    
    if values:
        fig.add_trace(
            go.Pie(labels=labels, values=values),
            row=2, col=1
        )
    
    # Subplot 4: Sales distribution (histogram) - Filter out zeros and negatives for better visualization
    volume_positive = df_filtered[df_filtered['Volume'] > 0]['Volume']
    
    # Use log scale for better distribution visualization since data is highly skewed
    fig.add_trace(
        go.Histogram(x=volume_positive, nbinsx=50, name='Sales Distribution (>0)', 
                     showlegend=False),  # Hide legend for histogram
        row=2, col=2
    )
    
    # Update y-axis to log scale for better visualization of skewed data
    fig.update_yaxes(type="log", row=2, col=2)
    
    fig.update_layout(
        title_text=f"Book Sales Analysis Dashboard ({len(selected_isbns)} Books)",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0,
            xanchor="left",
            x=1.02,
            font=dict(size=10)  # Smaller font for pie chart legend
        ),
        height=800,
        width=1200,  # Reduced width since only pie chart has legend
        margin=dict(r=200)  # Smaller right margin
    )
    
    return fig


def save_plot(fig: go.Figure, filename: str, format: str = 'html') -> None:
    """
    Save a plot to file.
    
    Args:
        fig: Plotly figure object
        filename: Output filename
        format: Output format ('html', 'png', 'jpg', 'svg', 'pdf')
    """
    logger.info(f"Saving plot to {filename}")
    
    if format == 'html':
        fig.write_html(filename)
    elif format in ['png', 'jpg', 'svg', 'pdf']:
        fig.write_image(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def display_plot(fig: go.Figure, show: bool = True) -> None:
    """
    Display a plot.
    
    Args:
        fig: Plotly figure object
        show: Whether to show the plot
    """
    if show:
        fig.show()
    else:
        return fig


if __name__ == "__main__":
    logger.info("Plotting utilities module loaded successfully") 