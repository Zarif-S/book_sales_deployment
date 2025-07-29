"""
Main analysis script for book sales data.

This script demonstrates how to use the cleaned preprocessing and plotting functions
to analyze book sales data in a structured and organized way.
"""

import sys
import os
import pandas as pd
import logging
from typing import Dict, List
import plotly.graph_objects as go

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps._01_load_data import get_csv_data
from steps._02_preprocessing import (
    preprocess_sales_data,
    get_isbn_to_title_mapping,
    get_data_info,
    filter_data_by_date,
    get_isbns_beyond_date,
    load_processed_data
)

from utils.plotting import (
    plot_weekly_volume_by_isbn,
    plot_yearly_volume_by_isbn,
    plot_selected_books_weekly,
    plot_selected_books_yearly,
    plot_sales_comparison,
    plot_sales_trends,
    create_summary_dashboard,
    save_plot,
    display_plot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_real_data() -> pd.DataFrame:
    """
    Load real data from CSV files or processed data.
    First tries to load processed data, then falls back to raw CSV data.
    
    Returns:
        DataFrame with book sales data
    """
    logger.info("Loading real data...")
    
    # First, try to load processed data
    processed_data = load_processed_data("processed_sales_data_filled")
    if processed_data is not None:
        logger.info(f"Loaded processed data with shape: {processed_data.shape}")
        # Mark this as already processed so we don't preprocess it again
        processed_data.attrs['already_processed'] = True
        return processed_data
    
    # If processed data not available, load from raw CSV files
    logger.info("Processed data not found, loading from raw CSV files...")
    data_dict = get_csv_data()
    
    if data_dict is not None and 'df_uk_weekly' in data_dict:
        df_raw = data_dict['df_uk_weekly']
        logger.info(f"Loaded raw data with shape: {df_raw.shape}")
        return df_raw
    else:
        logger.error("Could not load data from CSV files. Please ensure data files exist in data/raw/")
        raise FileNotFoundError("No data files found")


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for demonstration purposes.
    This is kept as a fallback option.
    
    Returns:
        Sample DataFrame with book sales data
    """
    logger.info("Loading sample data...")
    
    # Create sample data structure similar to the original
    # This is a placeholder - replace with your actual data loading logic
    sample_data = {
        'End Date': pd.date_range('2001-01-01', '2024-12-31', freq='W'),
        'ISBN': ['9780722532935', '9780241003008', '9780140500875'] * 1000,
        'Value': [10.0, 15.0, 8.0] * 1000,
        'ASP': [9.5, 14.0, 7.5] * 1000,
        'RRP': [12.0, 18.0, 10.0] * 1000,
        'Volume': [100, 150, 80] * 1000,
        'Title': ['Alchemist, The', 'Very Hungry Caterpillar, The', 'Very Hungry Caterpillar, The'] * 1000,
        'Author': ['Paulo Coelho', 'Eric Carle', 'Eric Carle'] * 1000,
        'Binding': ['Paperback', 'Hardback', 'Paperback'] * 1000,
        'Imprint': ['HarperCollins', 'Penguin', 'Penguin'] * 1000,
        'Publisher Group': ['HarperCollins', 'Penguin Random House', 'Penguin Random House'] * 1000,
        'Product Class': ['Fiction', 'Children', 'Children'] * 1000,
        'Source': ['Nielsen', 'Nielsen', 'Nielsen'] * 1000
    }
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Sample data loaded with shape: {df.shape}")
    
    return df


def run_complete_analysis():
    """
    Run the complete book sales analysis pipeline.
    """
    logger.info("Starting complete book sales analysis...")
    
    # Step 1: Load data
    try:
        df_raw = load_real_data()
    except FileNotFoundError:
        logger.warning("Real data not available, using sample data...")
        df_raw = load_sample_data()
    
    # Step 2: Check if data is already processed
    if hasattr(df_raw, 'attrs') and df_raw.attrs.get('already_processed', False):
        logger.info("Data is already processed, skipping preprocessing...")
        df_processed = df_raw
        
        # Load selected books data separately
        selected_books_data = load_processed_data("selected_books_data")
        if selected_books_data is None:
            logger.warning("Selected books data not found, will create empty DataFrame")
            selected_books_data = pd.DataFrame()
    else:
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        df_processed, df_filtered, selected_books_data = preprocess_sales_data(df_raw)
    
    # Step 3: Get data information
    logger.info("Step 3: Getting data information...")
    data_info = get_data_info(df_processed)
    logger.info(f"Dataset info: {data_info}")
    
    # Step 4: Get ISBN to title mapping
    isbn_to_title = get_isbn_to_title_mapping()
    
    # Step 5: Create various plots
    logger.info("Step 5: Creating visualizations...")
    
    # Plot 1: Weekly volume for all ISBNs beyond 2024-07-01
    df_beyond_2024 = filter_data_by_date(df_processed, '2024-07-01')
    if not df_beyond_2024.empty:
        fig1 = plot_weekly_volume_by_isbn(
            df_beyond_2024,
            "Weekly Volume for Each ISBN > 2024-07-01"
        )
        display_plot(fig1)
        save_plot(fig1, "outputs/weekly_volume_beyond_2024.html")
    
    # Plot 2: Yearly comparison between first 12 years and last 12 years
    fig2_1, fig2_2 = plot_sales_comparison(
        df_processed,
        period1_start='2001-01-01', period1_end='2012-12-31',
        period2_start='2013-01-01', period2_end='2024-12-31',
        title="Sales Comparison: First 12 Years vs Last 12 Years"
    )
    display_plot(fig2_1)
    display_plot(fig2_2)
    save_plot(fig2_1, "outputs/first_12_years_sales.html")
    save_plot(fig2_2, "outputs/last_12_years_sales.html")
    
    # Plot 3: Selected books weekly data
    if not selected_books_data.empty:
        fig3 = plot_selected_books_weekly(
            selected_books_data,
            isbn_to_title,
            "Weekly Sales Data for Selected Books (2012 Onwards)"
        )
        display_plot(fig3)
        save_plot(fig3, "outputs/selected_books_weekly.html")
        
        # Plot 4: Selected books yearly data
        fig4 = plot_selected_books_yearly(
            selected_books_data,
            isbn_to_title,
            "Yearly Sales Data for Selected Books (From 2012 Onward)"
        )
        display_plot(fig4)
        save_plot(fig4, "outputs/selected_books_yearly.html")
        
        # Plot 5: Sales trends analysis
        selected_isbns = list(isbn_to_title.keys())
        fig5 = plot_sales_trends(
            selected_books_data,
            selected_isbns,
            isbn_to_title,
            "Sales Trends Analysis for Selected Books"
        )
        display_plot(fig5)
        save_plot(fig5, "outputs/sales_trends_analysis.html")
        
        # Plot 6: Summary dashboard
        fig6 = create_summary_dashboard(selected_books_data, isbn_to_title)
        display_plot(fig6)
        save_plot(fig6, "outputs/summary_dashboard.html")
    
    logger.info("Analysis completed successfully!")


def run_specific_analysis(analysis_type: str):
    """
    Run a specific type of analysis.
    
    Args:
        analysis_type: Type of analysis to run ('preprocessing', 'plotting', 'comparison')
    """
    logger.info(f"Running {analysis_type} analysis...")
    
    # Load and preprocess data
    try:
        df_raw = load_real_data()
    except FileNotFoundError:
        logger.warning("Real data not available, using sample data...")
        df_raw = load_sample_data()
    
    # Check if data is already processed
    if hasattr(df_raw, 'attrs') and df_raw.attrs.get('already_processed', False):
        logger.info("Data is already processed, skipping preprocessing...")
        df_processed = df_raw
        
        # Load selected books data separately
        selected_books_data = load_processed_data("selected_books_data")
        if selected_books_data is None:
            logger.warning("Selected books data not found, will create empty DataFrame")
            selected_books_data = pd.DataFrame()
    else:
        df_processed, _, selected_books_data = preprocess_sales_data(df_raw)
    
    isbn_to_title = get_isbn_to_title_mapping()
    
    if analysis_type == 'preprocessing':
        # Focus on data preprocessing
        logger.info("Running preprocessing analysis...")
        
        # Get ISBNs beyond 2024-07-01
        isbns_beyond_2024 = get_isbns_beyond_date(df_processed, '2024-07-01')
        logger.info(f"ISBNs with data beyond 2024-07-01: {isbns_beyond_2024}")
        
        # Get data info
        data_info = get_data_info(df_processed)
        logger.info(f"Dataset information: {data_info}")
        
    elif analysis_type == 'plotting':
        # Focus on plotting
        logger.info("Running plotting analysis...")
        
        # Create various plots
        fig1 = plot_weekly_volume_by_isbn(df_processed, "All Books Weekly Volume")
        fig2 = plot_yearly_volume_by_isbn(df_processed, "All Books Yearly Volume")
        
        display_plot(fig1)
        display_plot(fig2)
        
    elif analysis_type == 'comparison':
        # Focus on comparison analysis
        logger.info("Running comparison analysis...")
        
        # Compare different time periods
        fig1, fig2 = plot_sales_comparison(
            df_processed,
            period1_start='2001-01-01', period1_end='2012-12-31',
            period2_start='2013-01-01', period2_end='2024-12-31'
        )
        
        display_plot(fig1)
        display_plot(fig2)
    
    else:
        logger.error(f"Unknown analysis type: {analysis_type}")


def main():
    """
    Main function to run the analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Book Sales Analysis')
    parser.add_argument('--analysis-type', 
                       choices=['complete', 'preprocessing', 'plotting', 'comparison'],
                       default='complete',
                       help='Type of analysis to run')
    
    args = parser.parse_args()
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    if args.analysis_type == 'complete':
        run_complete_analysis()
    else:
        run_specific_analysis(args.analysis_type)


if __name__ == "__main__":
    main() 