"""
Test script for the cleaned codebase.

This script tests that all the cleaned functions can be imported and used correctly.
"""

import sys
import os
import pandas as pd
import logging

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported successfully."""
    logger.info("Testing imports...")
    
    try:
        # Test preprocessing imports
        from steps._02_preprocessing import (
            convert_data_types,
            prepare_time_series_data,
            fill_missing_weeks,
            filter_data_by_date,
            get_isbns_beyond_date,
            get_book_summary,
            select_specific_books,
            get_isbn_to_title_mapping,
            aggregate_yearly_data,
            preprocess_sales_data,
            get_data_info
        )
        logger.info("✅ All preprocessing functions imported successfully")
        
        # Test plotting imports
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
        logger.info("✅ All plotting functions imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def create_test_data():
    """Create test data for testing functions."""
    logger.info("Creating test data...")
    
    # Create a small test dataset
    test_data = {
        'End Date': pd.date_range('2020-01-01', '2020-12-31', freq='W'),
        'ISBN': ['9780722532935', '9780241003008'] * 26,  # 52 weeks / 2 books
        'Value': [10.0, 15.0] * 26,
        'ASP': [9.5, 14.0] * 26,
        'RRP': [12.0, 18.0] * 26,
        'Volume': [100, 150] * 26,
        'Title': ['Alchemist, The', 'Very Hungry Caterpillar, The'] * 26,
        'Author': ['Paulo Coelho', 'Eric Carle'] * 26,
        'Binding': ['Paperback', 'Hardback'] * 26,
        'Imprint': ['HarperCollins', 'Penguin'] * 26,
        'Publisher Group': ['HarperCollins', 'Penguin Random House'] * 26,
        'Product Class': ['Fiction', 'Children'] * 26,
        'Source': ['Nielsen', 'Nielsen'] * 26
    }
    
    df = pd.DataFrame(test_data)
    logger.info(f"Test data created with shape: {df.shape}")
    return df


def test_preprocessing_functions():
    """Test preprocessing functions with test data."""
    logger.info("Testing preprocessing functions...")
    
    try:
        from steps._02_preprocessing import (
            convert_data_types,
            prepare_time_series_data,
            fill_missing_weeks,
            filter_data_by_date,
            get_isbn_to_title_mapping
        )
        
        # Create test data
        df_test = create_test_data()
        
        # Test convert_data_types
        df_converted = convert_data_types(df_test.copy())
        assert df_converted['ISBN'].dtype == 'object'
        logger.info("✅ convert_data_types works correctly")
        
        # Test prepare_time_series_data
        df_time_series = prepare_time_series_data(df_converted.copy())
        assert isinstance(df_time_series.index, pd.DatetimeIndex)
        logger.info("✅ prepare_time_series_data works correctly")
        
        # Test fill_missing_weeks
        df_filled = fill_missing_weeks(df_time_series)
        assert len(df_filled) >= len(df_time_series)
        logger.info("✅ fill_missing_weeks works correctly")
        
        # Test filter_data_by_date
        df_filtered = filter_data_by_date(df_filled, '2020-06-01')
        assert len(df_filtered) < len(df_filled)
        logger.info("✅ filter_data_by_date works correctly")
        
        # Test get_isbn_to_title_mapping
        mapping = get_isbn_to_title_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        logger.info("✅ get_isbn_to_title_mapping works correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {e}")
        return False


def test_plotting_functions():
    """Test plotting functions with test data."""
    logger.info("Testing plotting functions...")
    
    try:
        from utils.plotting import (
            plot_weekly_volume_by_isbn,
            plot_yearly_volume_by_isbn,
            plot_selected_books_weekly,
            plot_selected_books_yearly
        )
        from steps._02_preprocessing import get_isbn_to_title_mapping
        
        # Create test data
        df_test = create_test_data()
        df_converted = convert_data_types(df_test.copy())
        df_time_series = prepare_time_series_data(df_converted.copy())
        df_filled = fill_missing_weeks(df_time_series)
        
        # Test plot_weekly_volume_by_isbn
        fig1 = plot_weekly_volume_by_isbn(df_filled, "Test Weekly Plot")
        assert fig1 is not None
        logger.info("✅ plot_weekly_volume_by_isbn works correctly")
        
        # Test plot_yearly_volume_by_isbn
        fig2 = plot_yearly_volume_by_isbn(df_filled, "Test Yearly Plot")
        assert fig2 is not None
        logger.info("✅ plot_yearly_volume_by_isbn works correctly")
        
        # Test plot_selected_books_weekly
        isbn_to_title = get_isbn_to_title_mapping()
        selected_books = df_filled[df_filled['ISBN'].isin(['9780722532935', '9780241003008'])]
        fig3 = plot_selected_books_weekly(selected_books, isbn_to_title, "Test Selected Books")
        assert fig3 is not None
        logger.info("✅ plot_selected_books_weekly works correctly")
        
        # Test plot_selected_books_yearly
        fig4 = plot_selected_books_yearly(selected_books, isbn_to_title, "Test Selected Books Yearly")
        assert fig4 is not None
        logger.info("✅ plot_selected_books_yearly works correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Plotting test failed: {e}")
        return False


def test_main_pipeline():
    """Test the main preprocessing pipeline."""
    logger.info("Testing main preprocessing pipeline...")
    
    try:
        from steps._02_preprocessing import preprocess_sales_data
        
        # Create test data
        df_test = create_test_data()
        
        # Test main pipeline
        df_processed, df_filtered, selected_books = preprocess_sales_data(df_test)
        
        assert df_processed is not None
        assert df_filtered is not None
        assert selected_books is not None
        logger.info("✅ Main preprocessing pipeline works correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Main pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting tests for cleaned codebase...")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Preprocessing Functions Test", test_preprocessing_functions),
        ("Plotting Functions Test", test_plotting_functions),
        ("Main Pipeline Test", test_main_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The cleaned codebase is working correctly.")
    else:
        logger.error("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    # Import the missing function for testing
    from steps._02_preprocessing import convert_data_types, prepare_time_series_data
    
    success = main()
    sys.exit(0 if success else 1) 

    