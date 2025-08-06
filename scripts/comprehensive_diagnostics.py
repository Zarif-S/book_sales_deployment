#!/usr/bin/env python3
"""
Script to run time series diagnostics for book sales data.

This script demonstrates how to use the cleaned diagnostics module
to perform comprehensive time series analysis on any books in the dataset.
"""

import sys
import os
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from steps._03__time_series_diagnostics import (
    set_data, 
    run_complete_diagnostics,
    run_diagnostics_for_books,
    run_diagnostics_for_book,
    get_all_books,
    perform_decomposition_analysis,
    perform_acf_pacf_analysis,
    perform_stationarity_analysis,
    DEFAULT_BOOKS
)

def load_processed_data():
    """
    Load processed data from the data/processed directory.
    This is the preferred method for real usage.
    """
    try:
        from steps._02_preprocessing import load_processed_data
        
        # Try to load processed data
        processed_data = load_processed_data("selected_books_data")
        
        if processed_data is not None:
            print("Loaded processed data from data/processed/selected_books_data.csv")
            return processed_data
        else:
            print("No processed data found. Please ensure you have run the preprocessing pipeline first.")
            raise FileNotFoundError("No processed data available")
            
    except Exception as e:
        print(f"Error loading processed data: {e}")
        print("Please ensure you have run the preprocessing pipeline to generate the required data.")
        raise

def main():
    """
    Main function to run diagnostics.
    """
    print("=" * 60)
    print("TIME SERIES DIAGNOSTICS FOR BOOK SALES DATA")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    sales_data = load_processed_data()
    
    # Set data in the diagnostics module
    set_data(sales_data)
    
    print(f"Data loaded successfully:")
    print(f"- Total records: {len(sales_data)}")
    print(f"- Unique books: {sales_data['ISBN'].nunique()}")
    print(f"- Date range: {sales_data.index.min()} to {sales_data.index.max()}")
    
    # Get all available books
    all_books = get_all_books()
    print(f"- Available books: {all_books}")
    
    # Run diagnostics for default books only (avoid duplicate plots)
    print("\n" + "="*40)
    print("RUNNING DIAGNOSTICS FOR DEFAULT BOOKS")
    print("="*40)
    
    print("\nRunning complete diagnostics for default books...")
    default_results = run_complete_diagnostics(show_plots=True)
    
    # Example 3: Run diagnostics for a single book (optional, with plots disabled)
    print("\n" + "="*40)
    print("EXAMPLE: DIAGNOSTICS FOR SINGLE BOOK (PLOTS DISABLED)")
    print("="*40)
    
    if all_books:
        single_book = all_books[0]
        print(f"\nRunning diagnostics for single book: {single_book} (plots disabled)")
        single_results = run_diagnostics_for_book(single_book)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Print summary
    print("\nSummary of Results:")
    print("- Decomposition analysis completed")
    print("- ACF/PACF analysis completed")
    print("- Stationarity tests completed")
    print("- COVID-19 analysis completed")
    print(f"- Default books analyzed: {len(DEFAULT_BOOKS)}")
    print(f"- Total available books: {len(all_books)}")
    
    print("\nNext steps:")
    print("1. Review the generated plots and analysis")
    print("2. Use the results to inform ARIMA model selection")
    print("3. Run the ARIMA analysis notebook for forecasting")
    print("4. Customize the analysis for your specific books")
    
    return {
        'default_results': default_results,
        'available_books': all_books,
        'data_shape': sales_data.shape
    }

def run_custom_analysis(book_isbns: list, show_plots: bool = True):
    """
    Run diagnostics for custom list of books.
    
    Args:
        book_isbns: List of ISBNs to analyze
        show_plots: Whether to display plots
    """
    print(f"\nRunning custom analysis for books: {book_isbns}")
    print(f"Plot generation: {'Enabled' if show_plots else 'Disabled'}")
    
    # Load data
    sales_data = load_processed_data()
    set_data(sales_data)
    
    # Run diagnostics for specified books
    results = run_diagnostics_for_books(book_isbns)
    
    print(f"Custom analysis completed for {len(book_isbns)} books")
    return results

def run_analysis_with_plot_control(show_plots: bool = True, books: Optional[list] = None):
    """
    Run diagnostics with plot control.
    
    Args:
        show_plots: Whether to display plots
        books: List of ISBNs to analyze (if None, uses default books)
    """
    print("=" * 60)
    print("TIME SERIES DIAGNOSTICS WITH PLOT CONTROL")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    sales_data = load_processed_data()
    set_data(sales_data)
    
    if books is None:
        books = DEFAULT_BOOKS
    
    print(f"Analyzing books: {books}")
    print(f"Plot generation: {'Enabled' if show_plots else 'Disabled'}")
    
    # Run diagnostics
    results = run_complete_diagnostics(
        isbns=books,
        show_plots=show_plots
    )
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\nScript completed successfully!")
        
        print("\n" + "="*50)
        print("PLOT CONTROL OPTIONS")
        print("="*50)
        print("To control plot generation, you can:")
        print("1. Disable all plots:")
        print("   run_analysis_with_plot_control(show_plots=False)")
        print("2. Analyze specific books:")
        print("   run_custom_analysis([9780722532935], show_plots=True)")
        print("3. Run analysis without plots for all books:")
        print("   run_analysis_with_plot_control(show_plots=False, books=get_all_books())")
        
        print("\nDefault books (The Alchemist and The Very Hungry Caterpillar) are analyzed automatically.")
        
    except Exception as e:
        print(f"\nError running diagnostics: {e}")
        print("Please check your data and dependencies.")
        sys.exit(1) 