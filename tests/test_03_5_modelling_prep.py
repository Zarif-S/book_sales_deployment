"""
Unit tests for the modelling preparation module.

This test suite covers the data preparation functions that split time series data
into training and testing sets for ARIMA model development.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the steps directory to the path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'steps'))

from steps._03_5_modelling_prep import prepare_data_after_2012, prepare_multiple_books_data


class TestPrepareDataAfter2012:
    """Test suite for the prepare_data_after_2012 function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create a realistic time series dataset with weekly frequency
        dates = pd.date_range('2010-01-01', '2024-12-31', freq='W')
        
        # Create sample data with some seasonality and trend
        np.random.seed(42)  # For reproducible results
        base_volume = 100
        trend = np.linspace(0, 50, len(dates))  # Upward trend
        seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  # Annual seasonality
        noise = np.random.normal(0, 10, len(dates))
        
        volume = base_volume + trend + seasonality + noise
        volume = np.maximum(volume, 0)  # Ensure non-negative volumes
        
        self.test_data = pd.DataFrame({
            'Volume': volume,
            'Value': volume * 10,  # Simple relationship
            'Title': ['Test Book'] * len(dates),
            'ISBN': ['9781234567890'] * len(dates)
        }, index=dates)
        
        # Create data that starts before 2012
        dates_before_2012 = pd.date_range('2005-01-01', '2011-12-31', freq='W')
        self.data_before_2012 = pd.DataFrame({
            'Volume': np.random.randint(50, 200, len(dates_before_2012)),
            'Value': np.random.randint(500, 2000, len(dates_before_2012)),
            'Title': ['Old Book'] * len(dates_before_2012),
            'ISBN': ['9780987654321'] * len(dates_before_2012)
        }, index=dates_before_2012)
    
    def test_prepare_data_after_2012_success(self):
        """Test successful data preparation with default parameters."""
        train_data, test_data = prepare_data_after_2012(self.test_data, 'Volume')
        
        # Check that we get the expected data types
        assert isinstance(train_data, pd.Series)
        assert isinstance(test_data, pd.Series)
        
        # Check that test data has the default split size (32)
        assert len(test_data) == 32
        
        # Check that train data contains the rest
        expected_train_size = len(self.test_data[self.test_data.index >= '2012-01-01']) - 32
        assert len(train_data) == expected_train_size
        
        # Check that all data is after 2012-01-01
        assert train_data.index.min() >= pd.Timestamp('2012-01-01')
        assert test_data.index.min() >= pd.Timestamp('2012-01-01')
        
        # Check that train and test data don't overlap
        assert train_data.index.max() < test_data.index.min()
        
        # Check that the data is continuous (no gaps)
        all_data = pd.concat([train_data, test_data]).sort_index()
        expected_dates = pd.date_range(
            all_data.index.min(), 
            all_data.index.max(), 
            freq='W'
        )
        assert len(all_data) == len(expected_dates)
    
    def test_prepare_data_after_2012_custom_split_size(self):
        """Test data preparation with custom split size."""
        custom_split = 16
        train_data, test_data = prepare_data_after_2012(self.test_data, 'Volume', custom_split)
        
        assert len(test_data) == custom_split
        assert len(train_data) == len(self.test_data[self.test_data.index >= '2012-01-01']) - custom_split
    
    def test_prepare_data_after_2012_different_column(self):
        """Test data preparation with a different column."""
        train_data, test_data = prepare_data_after_2012(self.test_data, 'Value', 32)
        
        # Check that we're using the correct column
        assert train_data.name == 'Value'
        assert test_data.name == 'Value'
        
        # Check that the data matches the original
        original_value_data = self.test_data[self.test_data.index >= '2012-01-01']['Value']
        assert len(train_data) + len(test_data) == len(original_value_data)
    
    def test_prepare_data_after_2012_insufficient_data(self):
        """Test error handling when there's insufficient data."""
        # Create data with only 20 entries after 2012
        limited_data = self.test_data[self.test_data.index >= '2012-01-01'].head(20)
        
        with pytest.raises(ValueError, match="Not enough data available for the test set"):
            prepare_data_after_2012(limited_data, 'Volume', 32)
    
    def test_prepare_data_after_2012_no_data_after_2012(self):
        """Test handling when no data exists after 2012."""
        with pytest.raises(ValueError, match="Not enough data available for the test set"):
            prepare_data_after_2012(self.data_before_2012, 'Volume', 32)
    
    def test_prepare_data_after_2012_edge_case_exact_split(self):
        """Test edge case where split size equals available data."""
        # Create data with exactly 32 entries after 2012
        exact_data = self.test_data[self.test_data.index >= '2012-01-01'].head(32)
        
        train_data, test_data = prepare_data_after_2012(exact_data, 'Volume', 32)
        
        assert len(test_data) == 32
        assert len(train_data) == 0  # No training data left
    
    def test_prepare_data_after_2012_data_integrity(self):
        """Test that the original data is not modified."""
        original_data = self.test_data.copy()
        train_data, test_data = prepare_data_after_2012(self.test_data, 'Volume')
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(original_data, self.test_data)
        
        # Check that train and test data are views/subsets of original
        original_after_2012 = self.test_data[self.test_data.index >= '2012-01-01']['Volume']
        combined_data = pd.concat([train_data, test_data]).sort_index()
        
        pd.testing.assert_series_equal(combined_data, original_after_2012)


class TestPrepareMultipleBooksData:
    """Test suite for the prepare_multiple_books_data function."""
    
    def setup_method(self):
        """Set up test data for multiple books."""
        # Create data for multiple books
        dates = pd.date_range('2010-01-01', '2024-12-31', freq='W')
        
        np.random.seed(42)
        
        # Book 1: Trending upward
        volume1 = 100 + np.linspace(0, 50, len(dates)) + np.random.normal(0, 10, len(dates))
        volume1 = np.maximum(volume1, 0)
        
        # Book 2: Seasonal pattern
        volume2 = 80 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 52) + np.random.normal(0, 8, len(dates))
        volume2 = np.maximum(volume2, 0)
        
        # Book 3: Declining trend
        volume3 = 150 - np.linspace(0, 40, len(dates)) + np.random.normal(0, 12, len(dates))
        volume3 = np.maximum(volume3, 0)
        
        self.books_data = {
            'Book_1': pd.DataFrame({
                'Volume': volume1,
                'Value': volume1 * 12,
                'Title': ['Book 1'] * len(dates),
                'ISBN': ['9781111111111'] * len(dates)
            }, index=dates),
            
            'Book_2': pd.DataFrame({
                'Volume': volume2,
                'Value': volume2 * 15,
                'Title': ['Book 2'] * len(dates),
                'ISBN': ['9782222222222'] * len(dates)
            }, index=dates),
            
            'Book_3': pd.DataFrame({
                'Volume': volume3,
                'Value': volume3 * 8,
                'Title': ['Book 3'] * len(dates),
                'ISBN': ['9783333333333'] * len(dates)
            }, index=dates)
        }
    
    def test_prepare_multiple_books_data_success(self):
        """Test successful preparation of multiple books data."""
        prepared_data = prepare_multiple_books_data(self.books_data, 'Volume')
        
        # Check that all books are processed
        assert len(prepared_data) == 3
        assert all(book_name in prepared_data for book_name in self.books_data.keys())
        
        # Check that each book has train and test data
        for book_name, (train_data, test_data) in prepared_data.items():
            assert train_data is not None
            assert test_data is not None
            assert isinstance(train_data, pd.Series)
            assert isinstance(test_data, pd.Series)
            assert len(test_data) == 32  # Default split size
            assert train_data.name == 'Volume'
            assert test_data.name == 'Volume'
    
    def test_prepare_multiple_books_data_custom_column(self):
        """Test preparation with custom column."""
        prepared_data = prepare_multiple_books_data(self.books_data, 'Value', 16)
        
        for book_name, (train_data, test_data) in prepared_data.items():
            assert train_data.name == 'Value'
            assert test_data.name == 'Value'
            assert len(test_data) == 16
    
    def test_prepare_multiple_books_data_with_error(self):
        """Test handling when one book has insufficient data."""
        # Add a book with insufficient data
        limited_dates = pd.date_range('2012-01-01', '2012-06-30', freq='W')
        self.books_data['Book_4'] = pd.DataFrame({
            'Volume': np.random.randint(50, 150, len(limited_dates)),
            'Value': np.random.randint(500, 1500, len(limited_dates)),
            'Title': ['Book 4'] * len(limited_dates),
            'ISBN': ['9784444444444'] * len(limited_dates)
        }, index=limited_dates)
        
        prepared_data = prepare_multiple_books_data(self.books_data, 'Volume')
        
        # Check that successful books are processed
        assert prepared_data['Book_1'][0] is not None
        assert prepared_data['Book_2'][0] is not None
        assert prepared_data['Book_3'][0] is not None
        
        # Check that failed book has None values
        assert prepared_data['Book_4'][0] is None
        assert prepared_data['Book_4'][1] is None
    
    def test_prepare_multiple_books_data_empty_dict(self):
        """Test handling of empty dictionary."""
        prepared_data = prepare_multiple_books_data({}, 'Volume')
        assert prepared_data == {}
    
    def test_prepare_multiple_books_data_data_consistency(self):
        """Test that the prepared data maintains consistency with original."""
        prepared_data = prepare_multiple_books_data(self.books_data, 'Volume')
        
        for book_name, (train_data, test_data) in prepared_data.items():
            if train_data is not None and test_data is not None:
                original_data = self.books_data[book_name][self.books_data[book_name].index >= '2012-01-01']['Volume']
                combined_data = pd.concat([train_data, test_data]).sort_index()
                
                # Check that combined data matches original
                pd.testing.assert_series_equal(combined_data, original_data)
                
                # Check that train and test don't overlap
                assert train_data.index.max() < test_data.index.min()


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_arima_ready_data_structure(self):
        """Test that the prepared data is suitable for ARIMA models."""
        # Create realistic data
        dates = pd.date_range('2010-01-01', '2024-12-31', freq='W')
        
        # Create data with trend and seasonality (typical for ARIMA)
        np.random.seed(42)
        trend = np.linspace(100, 200, len(dates))
        seasonality = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
        noise = np.random.normal(0, 15, len(dates))
        volume = trend + seasonality + noise
        volume = np.maximum(volume, 0)
        
        test_data = pd.DataFrame({
            'Volume': volume,
            'Value': volume * 10
        }, index=dates)
        
        # Prepare data
        train_data, test_data = prepare_data_after_2012(test_data, 'Volume', 32)
        
        # Check ARIMA requirements
        assert len(train_data) > 0, "Training data should not be empty"
        assert len(test_data) > 0, "Test data should not be empty"
        assert train_data.index.freq == 'W', "Data should maintain weekly frequency"
        assert test_data.index.freq == 'W', "Data should maintain weekly frequency"
        
        # Check for stationarity indicators (basic checks)
        assert train_data.std() > 0, "Training data should have variation"
        assert test_data.std() > 0, "Test data should have variation"
        
        # Check that data is continuous
        all_data = pd.concat([train_data, test_data]).sort_index()
        expected_weeks = len(all_data)
        actual_weeks = (all_data.index.max() - all_data.index.min()).days // 7 + 1
        assert abs(expected_weeks - actual_weeks) <= 1, "Data should be continuous"
    
    def test_multiple_books_arima_preparation(self):
        """Test preparation of multiple books for ARIMA analysis."""
        # Create multiple books with different patterns
        dates = pd.date_range('2010-01-01', '2024-12-31', freq='W')
        
        books_data = {}
        for i in range(3):
            np.random.seed(42 + i)
            base_volume = 100 + i * 50
            trend = np.linspace(0, 30 + i * 10, len(dates))
            seasonality = (20 + i * 5) * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
            noise = np.random.normal(0, 10 + i * 2, len(dates))
            volume = base_volume + trend + seasonality + noise
            volume = np.maximum(volume, 0)
            
            books_data[f'Book_{i+1}'] = pd.DataFrame({
                'Volume': volume,
                'Value': volume * (10 + i)
            }, index=dates)
        
        # Prepare all books
        prepared_data = prepare_multiple_books_data(books_data, 'Volume', 32)
        
        # Verify each book is properly prepared
        for book_name, (train_data, test_data) in prepared_data.items():
            assert train_data is not None, f"Training data should exist for {book_name}"
            assert test_data is not None, f"Test data should exist for {book_name}"
            assert len(train_data) > 0, f"Training data should not be empty for {book_name}"
            assert len(test_data) == 32, f"Test data should have 32 entries for {book_name}"
            
            # Check data quality
            assert not train_data.isna().any(), f"No NaN values in training data for {book_name}"
            assert not test_data.isna().any(), f"No NaN values in test data for {book_name}"
            assert (train_data >= 0).all(), f"No negative values in training data for {book_name}"
            assert (test_data >= 0).all(), f"No negative values in test data for {book_name}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 