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

def run_diagnostics_with_plots(analyze_all_books=False):
    """
    Run diagnostics with plots - standard visualizations.
    
    Args:
        analyze_all_books: If True, analyze all available books. If False, analyze only The Alchemist.
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

    # Determine which books to analyze
    if analyze_all_books:
        all_books = get_all_books()
        books_to_analyze = all_books
        print(f"- Available books: {all_books}")
        print(f"\nAnalyzing ALL {len(books_to_analyze)} books")
    else:
        books_to_analyze = DEFAULT_BOOKS
        print(f"\nAnalyzing DEFAULT book: The Alchemist ({DEFAULT_BOOKS[0]})")

    # Run diagnostics with plots
    print("\nRunning diagnostics with plots...")
    print("- All standard plots will be shown")
    print("- Focus on essential analysis")

    results = run_complete_diagnostics(
        isbns=books_to_analyze,
        show_plots=True
    )

    print("\n" + "=" * 60)
    print("MINIMAL DIAGNOSTICS COMPLETED")
    print("=" * 60)

    return results

def run_no_plots_diagnostics(analyze_all_books=False):
    """
    Run diagnostics without any plots - pure analysis only.
    
    Args:
        analyze_all_books: If True, analyze all available books. If False, analyze only The Alchemist.
    """
    print("=" * 60)
    print("ANALYSIS-ONLY DIAGNOSTICS (NO PLOTS)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    sales_data = load_processed_data()
    set_data(sales_data)

    print(f"Data loaded successfully:")
    print(f"- Total records: {len(sales_data)}")
    print(f"- Unique books: {sales_data['ISBN'].nunique()}")
    print(f"- Date range: {sales_data.index.min()} to {sales_data.index.max()}")

    # Determine which books to analyze
    if analyze_all_books:
        all_books = get_all_books()
        books_to_analyze = all_books
        print(f"- Available books: {all_books}")
        print(f"\nAnalyzing ALL {len(books_to_analyze)} books")
    else:
        books_to_analyze = DEFAULT_BOOKS
        print(f"\nAnalyzing DEFAULT book: The Alchemist ({DEFAULT_BOOKS[0]})")

    # Run diagnostics without plots
    print("\nRunning diagnostics without plots...")
    print("- All analysis performed, no visualizations")
    print("- Results available in returned data structure")

    results = run_complete_diagnostics(
        isbns=books_to_analyze,
        show_plots=False  # No plots at all
    )

    print("\n" + "=" * 60)
    print("ANALYSIS-ONLY DIAGNOSTICS COMPLETED")
    print("=" * 60)

    return results

def get_book_choice():
    """
    Ask user which books they want to analyze.
    
    Returns:
        bool: True if user wants to analyze all books, False for default book only
    """
    print("\nWhich books would you like to analyze?")
    print("1. The Alchemist only (default - ISBN: 9780722532935)")
    print("2. All available books in the dataset")
    
    book_choice = input("\nEnter your choice (1-2, default is 1): ").strip()
    
    if book_choice == "2":
        return True  # Analyze all books
    else:
        return False  # Analyze default book only

if __name__ == "__main__":
    try:
        print("Choose your diagnostic mode:")
        print("1. With plots (recommended)")
        print("2. No plots (analysis only)")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            analyze_all = get_book_choice()
            results = run_diagnostics_with_plots(analyze_all_books=analyze_all)
            print("\nDiagnostics completed successfully!")
            print("You can now review the plots and analysis.")

        elif choice == "2":
            analyze_all = get_book_choice()
            results = run_no_plots_diagnostics(analyze_all_books=analyze_all)
            print("\nAnalysis-only diagnostics completed successfully!")
            print("All analysis performed without generating plots.")

        elif choice == "3":
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid choice. Running diagnostics with plots by default...")
            analyze_all = get_book_choice()
            results = run_diagnostics_with_plots(analyze_all_books=analyze_all)

    except Exception as e:
        print(f"\nError running diagnostics: {e}")
        print("Please check your data and dependencies.")
        sys.exit(1)
