#!/usr/bin/env python3
"""
Test Script for ARIMA Models on Book Sales Data

This script tests the ARIMA modeling functionality on:
1. The Alchemist (ISBN: 9780722532935)
2. The Very Hungry Caterpillar (ISBN: 9780241003008)

It demonstrates the complete workflow from data loading to model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our ARIMA modules
from steps._04_arima import *
from steps._04_arima_plots import *

def load_and_prepare_data():
    """Load the book sales data and prepare it for analysis."""
    print("Loading book sales data...")
    
    # Load the selected books data
    data_path = "data/processed/selected_books_data.csv"
    data = pd.read_csv(data_path)
    
    # Convert End Date to datetime and set as index
    data['End Date'] = pd.to_datetime(data['End Date'])
    data.set_index('End Date', inplace=True)
    
    # Sort by date
    data.sort_index(inplace=True)
    
    print(f"Data loaded successfully!")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Total records: {len(data)}")
    print(f"Unique books: {data['ISBN'].nunique()}")
    
    return data

def test_single_book_analysis(data, isbn, title, forecast_horizon=32):
    """Run complete ARIMA analysis for a single book."""
    print(f"\n{'='*80}")
    print(f"TESTING ARIMA MODEL FOR: {title}")
    print(f"ISBN: {isbn}")
    print(f"{'='*80}")
    
    try:
        # Run complete analysis
        results = run_complete_arima_analysis(
            data=data,
            isbn=isbn,
            forecast_horizon=forecast_horizon,
            use_auto_arima=True,
            seasonal=True,
            title=title
        )
        
        # Create diagnostic plots if forecast was successful
        if results.get('forecast') and results.get('model'):
            print(f"\nCreating diagnostic plots for {title}...")
            
            # Get fitted model for residual analysis
            fitted_model = results['model']['fitted_model']
            
            # Plot residuals with tests
            residual_results = analyze_residuals(fitted_model, title)
            plot_residuals_with_tests(residual_results['residuals'], title)
            
            # Q-Q plots for normality
            plot_qq_residuals(residual_results['residuals'], f"{title} - Q-Q Analysis")
            
            # Main forecast plot
            fig, mae, mape = plot_prediction(
                series_train=results['train_data'],
                series_test=results['test_data'],
                forecast=results['forecast']['forecast'],
                forecast_int=results['forecast']['conf_int'],
                title=f"{title} - ARIMA Forecast"
            )
            
            print(f"\nForecast Performance for {title}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            
            # Show the plot
            fig.show()
            
            return results
            
        else:
            print(f"Analysis failed for {title}")
            return None
            
    except Exception as e:
        print(f"Error analyzing {title}: {e}")
        return None

def test_model_comparison(data, isbn, title):
    """Test model comparison functionality."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON FOR: {title}")
    print(f"{'='*80}")
    
    try:
        # Define models to test
        models_to_test = [
            (1, 0, 1),  # ARIMA(1,0,1)
            (1, 0, 0),  # ARIMA(1,0,0) - AR(1)
            (0, 0, 1),  # ARIMA(0,0,1) - MA(1)
            (2, 0, 2),  # ARIMA(2,0,2)
        ]
        
        # Run comparison
        comparison = compare_arima_models(
            data=data,
            isbn=isbn,
            models_to_test=models_to_test,
            forecast_horizon=32
        )
        
        # Plot comparison results
        if comparison.get('results'):
            plot_model_comparison(comparison, metric='rmse')
            plot_model_comparison(comparison, metric='mape')
        
        return comparison
        
    except Exception as e:
        print(f"Error in model comparison for {title}: {e}")
        return None

def test_alternative_arima_implementation(data, isbn, title):
    """Test the alternative ARIMA implementation."""
    print(f"\n{'='*80}")
    print(f"ALTERNATIVE ARIMA IMPLEMENTATION FOR: {title}")
    print(f"{'='*80}")
    
    try:
        # Prepare data using the alternative method
        book_data = data[data['ISBN'] == isbn].copy()
        train_data, test_data = prepare_data_after_2012(book_data, 'Volume', 32)
        
        print(f"Training data: {len(train_data)} observations")
        print(f"Test data: {len(test_data)} observations")
        
        # Convert to DataFrame for the alternative function
        train_df = train_data.to_frame(name='Volume')
        
        # Run alternative ARIMA implementation
        forecast_summary, model_summary, predict = arima_predictions(
            df=train_df, 
            column='Volume', 
            forecast_steps=32
        )
        
        print(f"\nAlternative ARIMA Results for {title}:")
        print(f"Forecast summary shape: {forecast_summary.shape}")
        print(f"Model summary available: {model_summary is not None}")
        print(f"Predictions shape: {predict.shape}")
        
        # Calculate residuals
        residuals = predict[1:] - train_df['Volume'][1:]
        
        # Plot residuals
        plot_residuals_with_tests(residuals, f"{title} - Alternative ARIMA")
        
        return {
            'forecast_summary': forecast_summary,
            'model_summary': model_summary,
            'predict': predict,
            'residuals': residuals
        }
        
    except Exception as e:
        print(f"Error in alternative ARIMA for {title}: {e}")
        return None

def main():
    """Main function to run all tests."""
    print("ARIMA MODEL TESTING SCRIPT")
    print("=" * 50)
    
    # Load data
    data = load_and_prepare_data()
    
    # Define books to test
    books = [
        {
            'isbn': 9780722532935,  # Changed from string to integer
            'title': 'The Alchemist',
            'author': 'Paulo Coelho'
        },
        {
            'isbn': 9780241003008,  # Changed from string to integer
            'title': 'The Very Hungry Caterpillar',
            'author': 'Eric Carle'
        }
    ]
    
    # Store results
    all_results = {}
    
    # Test each book
    for book in books:
        print(f"\n{'='*80}")
        print(f"TESTING BOOK: {book['title']} by {book['author']}")
        print(f"{'='*80}")
        
        # Test 1: Complete ARIMA analysis
        print(f"\n1. COMPLETE ARIMA ANALYSIS")
        results = test_single_book_analysis(
            data=data,
            isbn=book['isbn'],
            title=book['title']
        )
        all_results[book['title']] = {'complete_analysis': results}
        
        # Test 2: Model comparison
        print(f"\n2. MODEL COMPARISON")
        comparison = test_model_comparison(
            data=data,
            isbn=book['isbn'],
            title=book['title']
        )
        if book['title'] in all_results:
            all_results[book['title']]['comparison'] = comparison
        
        # Test 3: Alternative ARIMA implementation
        print(f"\n3. ALTERNATIVE ARIMA IMPLEMENTATION")
        alt_results = test_alternative_arima_implementation(
            data=data,
            isbn=book['isbn'],
            title=book['title']
        )
        if book['title'] in all_results:
            all_results[book['title']]['alternative'] = alt_results
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETED")
    print(f"{'='*80}")
    
    for book_title, results in all_results.items():
        print(f"\n{book_title}:")
        if results.get('complete_analysis'):
            print(f"  ✅ Complete analysis: SUCCESS")
            if results['complete_analysis'].get('accuracy'):
                acc = results['complete_analysis']['accuracy']
                print(f"    RMSE: {acc['rmse']:.2f}")
                print(f"    MAPE: {acc['mape']:.2f}%")
        else:
            print(f"  ❌ Complete analysis: FAILED")
        
        if results.get('comparison'):
            print(f"  ✅ Model comparison: SUCCESS")
        else:
            print(f"  ❌ Model comparison: FAILED")
            
        if results.get('alternative'):
            print(f"  ✅ Alternative ARIMA: SUCCESS")
        else:
            print(f"  ❌ Alternative ARIMA: FAILED")
    
    print(f"\nAll tests completed! Check the plots above for detailed results.")
    
    return all_results

if __name__ == "__main__":
    # Run the tests
    results = main() 