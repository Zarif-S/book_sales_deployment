#!/usr/bin/env python3
"""
Test script for individual book ARIMA training.
Demonstrates the new architecture for training separate models per book.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steps._04_arima_standalone import train_individual_book_arima, train_multiple_books_arima

def test_single_book():
    """Test training ARIMA model for a single book."""
    print("🧪 Testing single book ARIMA training...")
    
    # Test with The Alchemist
    book_isbn = "9780722532935"
    
    # Use very few trials for quick testing (normally would use 50+)
    results = train_individual_book_arima(
        book_isbn=book_isbn,
        output_dir="outputs/arima_test",
        n_trials=5  # Small number for quick test
    )
    
    if "error" not in results:
        print(f"✅ Successfully trained ARIMA model for {book_isbn}")
        print(f"📈 Best parameters: {results['best_params']}")
        print(f"📊 Metrics: {results['evaluation_metrics']}")
    else:
        print(f"❌ Failed to train model: {results['error']}")

def test_multiple_books():
    """Test training ARIMA models for multiple books."""
    print("\n🧪 Testing multiple book ARIMA training...")
    
    # Use both of your books
    book_isbns = ["9780722532935", "9780241003008"]  # Alchemist, Caterpillar
    
    # Use very few trials for quick testing
    summary = train_multiple_books_arima(
        book_isbns=book_isbns,
        output_dir="outputs/arima_test",
        n_trials=5  # Small number for quick test
    )
    
    print(f"✅ Training completed!")
    print(f"📊 Success rate: {summary['successful_models']}/{summary['total_books']}")

if __name__ == "__main__":
    print("🚀 Individual Book ARIMA Training Test")
    print("=" * 50)
    
    # Test single book first
    test_single_book()
    
    # Test multiple books
    test_multiple_books()
    
    print("\n🎉 Testing completed! Check outputs/arima_test/ for results.")