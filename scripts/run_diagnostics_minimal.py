#!/usr/bin/env python3
"""
Minimal diagnostics script with controlled plot generation.

This script demonstrates how to run diagnostics with minimal plots
to avoid plot overload while still getting the essential analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from steps._03__time_series_diagnostics import (
    set_data, 
    run_complete_diagnostics,
    get_all_books,
    DEFAULT_BOOKS
)

def load_processed_data():
    """
    Load processed data from the data/processed directory.
    """
    try:
        from steps._02_preprocessing import load_processed_data
        
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

def run_diagnostics_with_plots():
    """
    Run diagnostics with plots - standard visualizations.
    """
    print("=" * 60)
    print("MINIMAL TIME SERIES DIAGNOSTICS")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    sales_data = load_processed_data()
    set_data(sales_data)
    
    print(f"Data loaded successfully:")
    print(f"- Total records: {len(sales_data)}")
    print(f"- Unique books: {sales_data['ISBN'].nunique()}")
    print(f"- Date range: {sales_data.index.min()} to {sales_data.index.max()}")
    
    # Run diagnostics with plots
    print("\nRunning diagnostics with plots...")
    print("- All standard plots will be shown")
    print("- Focus on essential analysis")
    
    results = run_complete_diagnostics(
        isbns=DEFAULT_BOOKS,
        show_plots=True
    )
    
    print("\n" + "=" * 60)
    print("MINIMAL DIAGNOSTICS COMPLETED")
    print("=" * 60)
    
    return results

def run_no_plots_diagnostics():
    """
    Run diagnostics without any plots - pure analysis only.
    """
    print("=" * 60)
    print("ANALYSIS-ONLY DIAGNOSTICS (NO PLOTS)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    sales_data = load_processed_data()
    set_data(sales_data)
    
    # Run diagnostics without plots
    print("\nRunning diagnostics without plots...")
    print("- All analysis performed, no visualizations")
    print("- Results available in returned data structure")
    
    results = run_complete_diagnostics(
        isbns=DEFAULT_BOOKS,
        show_plots=False  # No plots at all
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS-ONLY DIAGNOSTICS COMPLETED")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    try:
        print("Choose your diagnostic mode:")
        print("1. With plots (recommended)")
        print("2. No plots (analysis only)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            results = run_diagnostics_with_plots()
            print("\nDiagnostics completed successfully!")
            print("You can now review the plots and analysis.")
            
        elif choice == "2":
            results = run_no_plots_diagnostics()
            print("\nAnalysis-only diagnostics completed successfully!")
            print("All analysis performed without generating plots.")
            
        elif choice == "3":
            print("Exiting...")
            sys.exit(0)
            
        else:
            print("Invalid choice. Running diagnostics with plots by default...")
            results = run_diagnostics_with_plots()
            
    except Exception as e:
        print(f"\nError running diagnostics: {e}")
        print("Please check your data and dependencies.")
        sys.exit(1) 