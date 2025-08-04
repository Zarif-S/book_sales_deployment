"""
Unit tests for the data preprocessing module.

This test suite covers the core data transformation, validation, and calculation functions
that are critical to the preprocessing pipeline's reliability.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import logging

# Import the functions to test from the preprocessing module
from steps._02_preprocessing import (
    validate_required_columns,
    validate_date_format,
    convert_data_types,
    prepare_time_series_data,
    fill_missing_weeks,
    filter_data_by_date,
    merge_and_fill_author_data,
    get_isbns_beyond_date,
    get_book_summary,
    aggregate_yearly_data,
    validate_preprocessing_inputs,
    create_resampling_aggregation_dict,
    select_specific_books,
    analyze_missing_values,
    get_data_info,
    get_isbn_to_title_mapping,
    get_project_directories,
    ensure_directory_exists,
    config
)


class TestValidationFunctions:
    """Test suite for validation functions."""
    
    def test_validate_required_columns_success(self):
        """Test validate_required_columns with all required columns present."""
        df = pd.DataFrame({
            'ISBN': ['123', '456'],
            'End Date': ['2023-01-01', '2023-01-02'],
            'Volume': [100, 200],
            'Value': [10.0, 20.0],
            'Title': ['Book A', 'Book B']
        })
        
        required_cols = ['ISBN', 'End Date', 'Volume', 'Value', 'Title']
        
        # Should not raise any exception
        validate_required_columns(df, required_cols)
    
    def test_validate_required_columns_missing_columns(self):
        """Test validate_required_columns with missing columns."""
        df = pd.DataFrame({
            'ISBN': ['123', '456'],
            'Volume': [100, 200]
        })
        
        required_cols = ['ISBN', 'End Date', 'Volume', 'Value', 'Title']
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, required_cols)
    
    def test_validate_required_columns_empty_dataframe(self):
        """Test validate_required_columns with empty DataFrame."""
        df = pd.DataFrame()
        required_cols = ['ISBN', 'End Date']
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, required_cols)
    
    def test_validate_required_columns_default_config(self):
        """Test validate_required_columns using default config."""
        df = pd.DataFrame({
            'ISBN': ['123'],
            'End Date': ['2023-01-01'],
            'Volume': [100],
            'Value': [10.0],
            'Title': ['Book A']
        })
        
        # Should not raise exception with default config
        validate_required_columns(df)
    
    def test_validate_date_format_valid_dates(self):
        """Test validate_date_format with valid date strings."""
        valid_dates = [
            '2023-01-01',
            '2023-12-31',
            '2020-02-29',  # Leap year
            '1999-01-01'
        ]
        
        for date_str in valid_dates:
            assert validate_date_format(date_str), f"Failed for valid date: {date_str}"
    
    def test_validate_date_format_invalid_dates(self):
        """Test validate_date_format with invalid date strings."""
        invalid_dates = [
            '2023-13-01',    # Invalid month
            '2023-01-32',    # Invalid day
            '2023/01/01',    # Wrong separator
            '01-01-2023',    # Wrong order
            '2023-1-1',      # Single digits
            'not-a-date',    # Not a date
            '',              # Empty string
            '2023-02-30'     # Invalid date
        ]
        
        for date_str in invalid_dates:
            assert not validate_date_format(date_str), f"Should fail for invalid date: {date_str}"


class TestDataTypeConversion:
    """Test suite for convert_data_types function."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.sample_data = pd.DataFrame({
            'ISBN': [9780722532935, 9780241003008, 9780140500875],
            'End Date': ['2023-01-01', '2023-01-08', '2023-01-15'],
            'Volume': ['100', '200', '300'],
            'Value': ['10.5', '20.0', '30.25'],
            'ASP': ['0.105', '0.10', '0.101'],
            'RRP': ['12.99', '24.99', '35.00'],
            'Title': ['Book A', 'Book B', 'Book C'],
            'Author': ['Author A', 'Author B', 'Author C']
        })
    
    def test_convert_data_types_success(self):
        """Test successful data type conversion."""
        df_converted = convert_data_types(self.sample_data)
        
        # Check ISBN conversion to string
        assert df_converted['ISBN'].dtype == 'object'
        assert df_converted['ISBN'].iloc[0] == '9780722532935'
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(df_converted['End Date'])
        
        # Check numeric conversions
        numeric_cols = ['Volume', 'Value', 'ASP', 'RRP']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df_converted[col]), f"{col} should be numeric"
    
    def test_convert_data_types_missing_required_columns(self):
        """Test convert_data_types with missing required columns."""
        df_incomplete = pd.DataFrame({
            'ISBN': ['123'],
            'Volume': [100]
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            convert_data_types(df_incomplete)
    
    def test_convert_data_types_invalid_dates(self):
        """Test convert_data_types with invalid date values."""
        df_bad_dates = self.sample_data.copy()
        df_bad_dates.loc[0, 'End Date'] = 'invalid-date'
        
        with pytest.raises(Exception):
            convert_data_types(df_bad_dates)
    
    def test_convert_data_types_missing_numeric_values(self):
        """Test convert_data_types with missing numeric values."""
        df_with_nulls = self.sample_data.copy()
        df_with_nulls.loc[0, 'Volume'] = np.nan
        df_with_nulls.loc[1, 'Value'] = None
        
        df_converted = convert_data_types(df_with_nulls)
        
        # Should handle NaN values gracefully
        assert pd.isna(df_converted.loc[0, 'Volume'])
        assert pd.isna(df_converted.loc[1, 'Value'])
    
    def test_convert_data_types_preserves_original(self):
        """Test that convert_data_types doesn't modify the original DataFrame."""
        original_dtypes = self.sample_data.dtypes.copy()
        convert_data_types(self.sample_data)
        
        # Original DataFrame should be unchanged
        pd.testing.assert_series_equal(self.sample_data.dtypes, original_dtypes)


class TestTimeSeriesPreparation:
    """Test suite for prepare_time_series_data function."""
    
    def test_prepare_time_series_data_with_date_column(self):
        """Test prepare_time_series_data when End Date is a column."""
        df = pd.DataFrame({
            'End Date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15']),
            'ISBN': ['123', '456', '789'],
            'Volume': [100, 200, 300]
        })
        
        df_ts = prepare_time_series_data(df)
        
        # Check that End Date is now the index
        assert df_ts.index.name == 'End Date'
        assert pd.api.types.is_datetime64_any_dtype(df_ts.index)
        assert df_ts.index.is_monotonic_increasing
        
        # Check that End Date column is removed
        assert 'End Date' not in df_ts.columns
    
    def test_prepare_time_series_data_already_indexed(self):
        """Test prepare_time_series_data when End Date is already the index."""
        dates = pd.to_datetime(['2023-01-15', '2023-01-01', '2023-01-08'])
        df = pd.DataFrame({
            'ISBN': ['123', '456', '789'],
            'Volume': [100, 200, 300]
        }, index=dates)
        df.index.name = 'End Date'
        
        df_ts = prepare_time_series_data(df)
        
        # Should be sorted
        assert df_ts.index.is_monotonic_increasing
        assert df_ts.index.name == 'End Date'
    
    def test_prepare_time_series_data_missing_date_column(self):
        """Test prepare_time_series_data when End Date column is missing."""
        df = pd.DataFrame({
            'ISBN': ['123', '456'],
            'Volume': [100, 200]
        })
        
        with pytest.raises(ValueError, match="'End Date' column not found"):
            prepare_time_series_data(df)


class TestFillMissingWeeks:
    """Test suite for fill_missing_weeks function."""
    
    def setup_method(self):
        """Set up test data with missing weeks."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-05'])  # Missing weeks
        self.df_with_gaps = pd.DataFrame({
            'ISBN': ['123', '123', '123'],
            'Volume': [100, 200, 300],
            'Value': [10.0, 20.0, 30.0],
            'Title': ['Book A', 'Book A', 'Book A'],
            'Author': ['Author A', 'Author A', 'Author A']
        }, index=dates)
        self.df_with_gaps.index.name = 'End Date'
    
    def test_fill_missing_weeks_single_isbn(self):
        """Test fill_missing_weeks with a single ISBN."""
        df_filled = fill_missing_weeks(self.df_with_gaps)
        isbn_data = df_filled[df_filled['ISBN'] == '123']

        start_date = self.df_with_gaps.index.min()
        end_date = self.df_with_gaps.index.max()
        expected_index = pd.date_range(start=start_date, end=end_date, freq='W-SAT')
        
        if end_date.dayofweek != 5:
            expected_index = expected_index.union([expected_index[-1] + pd.Timedelta(weeks=1)])
        
        expected_index = expected_index.tz_localize(None)
        expected_volume = pd.Series(0.0, index=expected_index, name='Volume')
        
        # Fix: Map original dates to their corresponding Saturday week-ending dates
        original_week_endings = self.df_with_gaps.index.to_series().dt.to_period('W-SAT').dt.end_time.dt.normalize().dt.tz_localize(None)
        original_volumes = self.df_with_gaps['Volume'].values
        
        # Now both indices should align properly
        expected_volume.loc[original_week_endings] = original_volumes
        
        actual_volume = isbn_data['Volume'].copy()
        actual_volume.index = actual_volume.index.tz_localize(None)
        
        # Ensure both series have the same index name
        expected_volume.index.name = 'End Date'
        
        pd.testing.assert_series_equal(actual_volume, expected_volume)
    
    def test_fill_missing_weeks_multiple_isbns(self):
        """Test fill_missing_weeks with multiple ISBNs."""
        # Add data for another ISBN
        dates2 = pd.to_datetime(['2023-01-08', '2023-01-22'])
        df_isbn2 = pd.DataFrame({
            'ISBN': ['456', '456'],
            'Volume': [150, 250],
            'Value': [15.0, 25.0],
            'Title': ['Book B', 'Book B'],
            'Author': ['Author B', 'Author B']
        }, index=dates2)
        df_isbn2.index.name = 'End Date'
        df_multi = pd.concat([self.df_with_gaps, df_isbn2])
        df_filled = fill_missing_weeks(df_multi)

        assert set(df_filled['ISBN'].unique()) == {'123', '456'}

        for isbn in ['123', '456']:
            isbn_data = df_filled[df_filled['ISBN'] == isbn]
            original_data = df_multi[df_multi['ISBN'] == isbn]

            # FIX: Correctly determine the resampled date range for each ISBN group
            start_date = original_data.index.min()
            end_date = original_data.index.max()
            expected_range = pd.date_range(start=start_date, end=end_date, freq='W-SAT')
            if end_date.dayofweek != 5: # 5 corresponds to Saturday
                expected_range = expected_range.union([expected_range[-1] + pd.Timedelta(weeks=1)])

            assert len(isbn_data) == len(expected_range)
    
    def test_fill_missing_weeks_missing_isbn_column(self):
        """Test fill_missing_weeks when ISBN column is missing."""
        df_no_isbn = self.df_with_gaps.drop(columns=['ISBN'])
        
        with pytest.raises(ValueError, match="'ISBN' column not found"):
            fill_missing_weeks(df_no_isbn)
    
    def test_fill_missing_weeks_preserves_categorical_data(self):
        """Test that categorical data is forward-filled correctly."""
        df_filled = fill_missing_weeks(self.df_with_gaps)
        
        # All rows for ISBN '123' should have the same Title and Author
        isbn_data = df_filled[df_filled['ISBN'] == '123']
        assert (isbn_data['Title'] == 'Book A').all()
        assert (isbn_data['Author'] == 'Author A').all()


class TestDateFiltering:
    """Test suite for filter_data_by_date function."""
    
    def setup_method(self):
        """Set up test data with date range."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='W-SAT')
        self.df_yearly = pd.DataFrame({
            'ISBN': ['123'] * len(dates),
            'Volume': range(len(dates)),
            'Value': [i * 10.0 for i in range(len(dates))]
        }, index=dates)
        self.df_yearly.index.name = 'End Date'
    
    def test_filter_data_by_date_start_only(self):
        """Test filtering with start date only."""
        start_date = '2023-06-01'
        df_filtered = filter_data_by_date(self.df_yearly, start_date)
        
        assert df_filtered.index.min() >= pd.to_datetime(start_date)
        assert len(df_filtered) < len(self.df_yearly)
    
    def test_filter_data_by_date_start_and_end(self):
        """Test filtering with both start and end dates."""
        start_date = '2023-03-01'
        end_date = '2023-09-30'
        
        df_filtered = filter_data_by_date(self.df_yearly, start_date, end_date)
        
        assert df_filtered.index.min() >= pd.to_datetime(start_date)
        assert df_filtered.index.max() <= pd.to_datetime(end_date)
    
    def test_filter_data_by_date_invalid_start_date(self):
        """Test filtering with invalid start date format."""
        with pytest.raises(ValueError, match="Invalid start_date format"):
            filter_data_by_date(self.df_yearly, '2023/01/01')
    
    def test_filter_data_by_date_invalid_end_date(self):
        """Test filtering with invalid end date format."""
        with pytest.raises(ValueError, match="Invalid end_date format"):
            filter_data_by_date(self.df_yearly, '2023-01-01', '2023/12/31')
    
    def test_filter_data_by_date_no_data_in_range(self):
        """Test filtering when no data exists in the specified range."""
        df_filtered = filter_data_by_date(self.df_yearly, '2024-01-01', '2024-12-31')
        assert len(df_filtered) == 0
    
    def test_filter_data_by_date_preserves_structure(self):
        """Test that filtering preserves DataFrame structure."""
        df_filtered = filter_data_by_date(self.df_yearly, '2023-06-01')
        
        # Should have same columns
        assert list(df_filtered.columns) == list(self.df_yearly.columns)
        assert df_filtered.index.name == self.df_yearly.index.name


class TestMergeAndFillAuthorData:
    """Test suite for merge_and_fill_author_data function."""
    
    def setup_method(self):
        """Set up test data for merging."""
        self.df_uk_weekly = pd.DataFrame({
            'ISBN': ['123', '456', '789', '123'],
            'Volume': [100, 200, 300, 150],
            'Value': [10.0, 20.0, 30.0, 15.0],
            'Title': ['Book A', 'Book B', 'Book C', 'Book A'],
            'Author': ['Author A', None, 'Author C', 'Author A']  # Missing author for ISBN 456
        })
        
        self.df_isbns = pd.DataFrame({
            'ISBN': ['123', '456', '789', '999'],
            'Author': ['Author A', 'Author B', 'Author C', 'Author D'],
            'Publisher': ['Pub A', 'Pub B', 'Pub C', 'Pub D']
        })
    
    def test_merge_and_fill_author_data_success(self):
        """Test successful merge and author filling."""
        df_merged = merge_and_fill_author_data(self.df_uk_weekly, self.df_isbns)
        
        # Check that missing author was filled
        isbn_456_data = df_merged[df_merged['ISBN'] == '456']
        assert (isbn_456_data['Author'] == 'Author B').all()
        
        # Check that existing authors were preserved
        isbn_123_data = df_merged[df_merged['ISBN'] == '123']
        assert (isbn_123_data['Author'] == 'Author A').all()
        
        # Check that the shape is preserved
        assert len(df_merged) == len(self.df_uk_weekly)
    
    def test_merge_and_fill_author_data_missing_isbn_column(self):
        """Test merge when ISBN column is missing."""
        df_no_isbn = self.df_uk_weekly.drop(columns=['ISBN'])
        
        with pytest.raises(ValueError, match="'ISBN' column not found"):
            merge_and_fill_author_data(df_no_isbn, self.df_isbns)
    
    def test_merge_and_fill_author_data_no_author_in_isbn_data(self):
        """Test merge when Author column is missing from ISBN data."""
        df_isbns_no_author = self.df_isbns.drop(columns=['Author'])
        
        # Should return original data with warning logged
        df_result = merge_and_fill_author_data(self.df_uk_weekly, df_isbns_no_author)
        pd.testing.assert_frame_equal(df_result, self.df_uk_weekly)
    
    def test_merge_and_fill_author_data_isbn_not_in_reference(self):
        """Test merge when some ISBNs are not in reference data."""
        df_uk_with_new_isbn = self.df_uk_weekly.copy()
        df_uk_with_new_isbn.loc[len(df_uk_with_new_isbn)] = {
            'ISBN': '999999', 'Volume': 400, 'Value': 40.0, 
            'Title': 'Book D', 'Author': None
        }
        
        df_merged = merge_and_fill_author_data(df_uk_with_new_isbn, self.df_isbns)
        
        # ISBN not in reference should still have None for Author
        new_isbn_data = df_merged[df_merged['ISBN'] == '999999']
        assert pd.isna(new_isbn_data['Author'].iloc[0])


class TestGetIsbnsBeyondDate:
    """Test suite for get_isbns_beyond_date function."""
    
    def setup_method(self):
        """Set up test data with various date ranges."""
        dates_early = pd.date_range('2023-01-01', '2023-06-30', freq='W-SAT')
        dates_late = pd.date_range('2024-01-01', '2024-06-30', freq='W-SAT')
        dates_mixed = pd.date_range('2023-06-01', '2024-12-31', freq='W-SAT')
        
        df_early = pd.DataFrame({'ISBN': ['123'] * len(dates_early)}, index=dates_early)
        df_late = pd.DataFrame({'ISBN': ['456'] * len(dates_late)}, index=dates_late)
        df_mixed = pd.DataFrame({'ISBN': ['789'] * len(dates_mixed)}, index=dates_mixed)
        
        self.df_combined = pd.concat([df_early, df_late, df_mixed])
        self.df_combined.index.name = 'End Date'
    
    def test_get_isbns_beyond_date_default_cutoff(self):
        """Test get_isbns_beyond_date with default cutoff date."""
        # Default cutoff is '2024-07-01'
        isbns_beyond = get_isbns_beyond_date(self.df_combined)
        
        # Only ISBN '789' (mixed dates) should have data beyond July 2024
        expected_isbns = ['789']
        assert set(isbns_beyond) == set(expected_isbns)
    
    def test_get_isbns_beyond_date_custom_cutoff(self):
        """Test get_isbns_beyond_date with custom cutoff date."""
        cutoff_date = '2023-12-01'
        isbns_beyond = get_isbns_beyond_date(self.df_combined, cutoff_date)
        
        # ISBNs '456' and '789' should have data beyond Dec 2023
        expected_isbns = {'456', '789'}
        assert set(isbns_beyond) == expected_isbns
    
    def test_get_isbns_beyond_date_invalid_format(self):
        """Test get_isbns_beyond_date with invalid date format."""
        with pytest.raises(ValueError, match="Invalid cutoff_date format"):
            get_isbns_beyond_date(self.df_combined, '2024/07/01')
    
    def test_get_isbns_beyond_date_no_data_beyond_cutoff(self):
        """Test when no ISBNs have data beyond cutoff."""
        cutoff_date = '2025-01-01'
        isbns_beyond = get_isbns_beyond_date(self.df_combined, cutoff_date)
        
        assert len(isbns_beyond) == 0


class TestGetBookSummary:
    """Test suite for get_book_summary function."""
    
    def setup_method(self):
        """Set up test data for book summary."""
        self.df_books = pd.DataFrame({
            'ISBN': ['123', '123', '456', '789', '789'],
            'Title': ['The Alchemist', 'The Alchemist', 'Very Hungry Caterpillar', 
                     'Different Book', 'Different Book'],
            'Volume': [100, 150, 200, 50, 75],
            'Value': [10.0, 15.0, 20.0, 5.0, 7.5],
            'ASP': [0.10, 0.10, 0.10, 0.10, 0.10],
            'Binding': ['Paperback', 'Hardcover', 'Paperback', 'Paperback', 'Paperback'],
            'RRP': [12.99, 15.99, 8.99, 6.99, 6.99]
        })
    
    def test_get_book_summary_single_title(self):
        """Test get_book_summary with single book title."""
        book_titles = ['Alchemist']
        summary = get_book_summary(self.df_books, book_titles)
        
        # Should find 'The Alchemist' (case-insensitive partial match)
        assert len(summary) == 2  # Two different bindings
        assert 'The Alchemist' in summary['Title'].values[0]
        
        # Check aggregated values
        alchemist_summary = summary[summary['Title'] == 'The Alchemist']
        expected_volume_sum = 100 + 150  # Sum of volumes for ISBN 123
        assert alchemist_summary['Volume_Sum'].sum() == expected_volume_sum
    
    def test_get_book_summary_multiple_titles(self):
        """Test get_book_summary with multiple book titles."""
        book_titles = ['Alchemist', 'Caterpillar']
        summary = get_book_summary(self.df_books, book_titles)
        
        # Should find both books
        titles_in_summary = set(summary['Title'].unique())
        assert 'The Alchemist' in str(titles_in_summary)
        assert 'Very Hungry Caterpillar' in str(titles_in_summary)
    
    def test_get_book_summary_no_matches(self):
        """Test get_book_summary when no books match."""
        book_titles = ['Nonexistent Book']
        summary = get_book_summary(self.df_books, book_titles)
        
        assert len(summary) == 0
    
    def test_get_book_summary_missing_title_column(self):
        """Test get_book_summary when Title column is missing."""
        df_no_title = self.df_books.drop(columns=['Title'])
        
        with pytest.raises(ValueError, match="'Title' column not found"):
            get_book_summary(df_no_title, ['Alchemist'])
    
    def test_get_book_summary_case_insensitive(self):
        """Test that book summary search is case-insensitive."""
        book_titles = ['ALCHEMIST', 'caterpillar']
        summary = get_book_summary(self.df_books, book_titles)
        
        assert len(summary) > 0  # Should find matches despite different cases


class TestAggregateYearlyData:
    """Test suite for aggregate_yearly_data function."""
    
    def setup_method(self):
        """Set up test data for yearly aggregation."""
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='W-SAT')
        volumes = [100 + i for i in range(len(dates))]
        
        self.df_weekly = pd.DataFrame({
            'ISBN': ['123'] * len(dates),
            'Volume': volumes
        }, index=dates)
        self.df_weekly.index.name = 'End Date'
    
    def test_aggregate_yearly_data_success(self):
        """Test successful yearly aggregation."""
        df_yearly = aggregate_yearly_data(self.df_weekly)
        
        # Check structure
        # FIX: The column is named 'Year', not 'End Date'.
        expected_columns = ['Year', 'ISBN', 'Volume']
        assert all(col in df_yearly.columns for col in expected_columns)
        
        # Should have data for 2023 and 2024
        # FIX: Check the 'Year' column.
        years = df_yearly['Year'].unique()
        assert 2023 in years
        assert 2024 in years
        
        # Check that volumes are summed correctly
        # FIX: Filter by the 'Year' column.
        volume_2023 = df_yearly[df_yearly['Year'] == 2023]['Volume'].iloc[0]
        volume_2024 = df_yearly[df_yearly['Year'] == 2024]['Volume'].iloc[0]
        
        assert volume_2023 > 0
        assert volume_2024 > 0
        assert volume_2023 != volume_2024  # Should be different sums
    
    def test_aggregate_yearly_data_missing_volume_column(self):
        """Test aggregate_yearly_data when Volume column is missing."""
        df_no_volume = self.df_weekly.drop(columns=['Volume'])
        
        with pytest.raises(ValueError, match="'Volume' column not found"):
            aggregate_yearly_data(df_no_volume)
    
    def test_aggregate_yearly_data_multiple_isbns(self):
        """Test yearly aggregation with multiple ISBNs."""
        # FIX: Define the missing DataFrame for the second ISBN.
        dates_456 = pd.date_range('2023-01-01', '2023-12-31', freq='W-SAT')
        volumes_456 = [50 + i for i in range(len(dates_456))]
        df_456 = pd.DataFrame({
            'ISBN': ['456'] * len(dates_456),
            'Volume': volumes_456
        }, index=dates_456)
        df_456.index.name = 'End Date'

        df_multi = pd.concat([self.df_weekly, df_456])
        df_yearly = aggregate_yearly_data(df_multi)

        # FIX: Use the 'Year' column for assertion.
        unique_combinations = df_yearly[['Year', 'ISBN']].drop_duplicates()

        # We expect (2023, '123'), (2024, '123'), and (2023, '456')
        assert len(unique_combinations) == 3


class TestValidatePreprocessingInputs:
    """Test suite for validate_preprocessing_inputs function."""
    
    def test_validate_preprocessing_inputs_all_defaults(self):
        """Test validation with all default values."""
        isbns, start_date, cutoff_date = validate_preprocessing_inputs()
        
        # Should return default values from config
        assert isinstance(isbns, list)
        assert len(isbns) > 0
        assert validate_date_format(start_date)
        assert validate_date_format(cutoff_date)
    
    def test_validate_preprocessing_inputs_custom_values(self):
        """Test validation with custom valid values."""
        custom_isbns = ['111', '222', '333']
        custom_start = '2020-01-01'
        custom_cutoff = '2023-01-01'
        
        isbns, start_date, cutoff_date = validate_preprocessing_inputs(
            custom_isbns, custom_start, custom_cutoff
        )
        
        assert isbns == custom_isbns
        assert start_date == custom_start
        assert cutoff_date == custom_cutoff
    
    def test_validate_preprocessing_inputs_invalid_isbn_list_empty(self):
        """Test validation with empty ISBN list."""
        with pytest.raises(ValueError, match="selected_isbns must be a non-empty list"):
            validate_preprocessing_inputs(selected_isbns=[])
    
    def test_validate_preprocessing_inputs_invalid_isbn_list_not_list(self):
        """Test validation with non-list ISBN parameter."""
        with pytest.raises(ValueError, match="selected_isbns must be a non-empty list"):
            validate_preprocessing_inputs(selected_isbns="not_a_list")
    
    def test_validate_preprocessing_inputs_invalid_start_date(self):
        """Test validation with invalid start date."""
        with pytest.raises(ValueError, match="Invalid start_date format"):
            validate_preprocessing_inputs(start_date='2023/01/01')
    
    def test_validate_preprocessing_inputs_invalid_cutoff_date(self):
        """Test validation with invalid cutoff date."""
        with pytest.raises(ValueError, match="Invalid cutoff_date format"):
            validate_preprocessing_inputs(cutoff_date='invalid-date')
    
    def test_validate_preprocessing_inputs_date_logic_warning(self, caplog):
        """Test that warning is logged when start_date >= cutoff_date."""
        with caplog.at_level(logging.WARNING):
            validate_preprocessing_inputs(
                start_date='2023-01-01', 
                cutoff_date='2023-01-01'
            )
        
        assert "start_date" in caplog.text
        assert "cutoff_date" in caplog.text


class TestCreateResamlingAggregationDict:
    """Test suite for create_resampling_aggregation_dict function."""
    
    def test_create_resampling_aggregation_dict_structure(self):
        """Test that aggregation dictionary has correct structure."""
        agg_dict = create_resampling_aggregation_dict()
        
        # Should be a dictionary
        assert isinstance(agg_dict, dict)
        
        # Should contain numeric columns with 'mean'
        numeric_cols = ['Value', 'ASP', 'RRP', 'Volume']
        for col in numeric_cols:
            assert col in agg_dict
            assert agg_dict[col] == 'mean'
        
        # Should contain categorical columns with 'first'
        categorical_cols = ['Title', 'Author', 'Binding', 'Imprint', 
                           'Publisher Group', 'Product Class', 'Source']
        for col in categorical_cols:
            assert col in agg_dict
            assert agg_dict[col] == 'first'


class TestSelectSpecificBooks:
    """Test suite for select_specific_books function."""
    
    def setup_method(self):
        """Set up test data with multiple books and date ranges."""
        dates = pd.date_range('2010-01-01', '2025-01-01', freq='W-SAT')
        
        # FIX: Use the default ISBNs from the config to match the test case
        isbns_to_use = config.DEFAULT_SELECTED_ISBNS
        
        data = []
        for i, date in enumerate(dates):
            # Cycle through the ISBNs from the config
            isbn = isbns_to_use[i % len(isbns_to_use)]
            
            data.append({
                'ISBN': isbn,
                'Volume': 100 + i,
                'Value': 10.0 + i,
                'Title': f'Book {isbn}'
            })
        
        self.df_multi_books = pd.DataFrame(data, index=dates[:len(data)])
        self.df_multi_books.index.name = 'End Date'
    
    def test_select_specific_books_default_params(self):
        """Test select_specific_books with default parameters."""
        # Should use default ISBN list and start date from config
        selected_data = select_specific_books(self.df_multi_books)
        
        # Should only contain data from default start date onwards
        assert selected_data.index.min() >= pd.to_datetime(config.DEFAULT_START_DATE)
        
        # Should only contain default ISBNs
        selected_isbns = set(selected_data['ISBN'].unique())
        # FIX: The expected set is simply the default ISBNs. No intersection needed.
        expected_isbns = set(config.DEFAULT_SELECTED_ISBNS)
        assert selected_isbns.issubset(expected_isbns)

    def test_select_specific_books_custom_params(self):
        """Test select_specific_books with custom parameters."""
        # FIX: Use ISBNs that actually exist in the test data.
        custom_isbns = [config.DEFAULT_SELECTED_ISBNS[0]]
        custom_start_date = '2015-01-01'
        
        selected_data = select_specific_books(
            self.df_multi_books,
            isbn_list=custom_isbns,
            start_date=custom_start_date
        )
        
        # Should only contain specified ISBNs
        assert set(selected_data['ISBN'].unique()) == set(custom_isbns)
        
        # Should only contain data from start date onwards
        assert selected_data.index.min() >= pd.to_datetime(custom_start_date)
    
    def test_select_specific_books_no_matching_isbns(self):
        """Test select_specific_books when no ISBNs match."""
        non_existent_isbns = ['999', '888']
        
        selected_data = select_specific_books(
            self.df_multi_books,
            isbn_list=non_existent_isbns
        )
        
        assert len(selected_data) == 0
    
    def test_select_specific_books_invalid_start_date(self):
        """Test select_specific_books with invalid start date format."""
        with pytest.raises(ValueError, match="Invalid start_date format"):
            select_specific_books(
                self.df_multi_books,
                start_date='2015/01/01'
            )
    
    def test_select_specific_books_missing_isbn_column(self):
        """Test select_specific_books when ISBN column is missing."""
        df_no_isbn = self.df_multi_books.drop(columns=['ISBN'])
        
        with pytest.raises(ValueError, match="'ISBN' column not found"):
            select_specific_books(df_no_isbn)


class TestAnalyzeMissingValues:
    """Test suite for analyze_missing_values function."""
    
    def test_analyze_missing_values_no_missing(self):
        """Test analyze_missing_values with no missing values."""
        df_complete = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.0, 2.0, 3.0]
        })
        
        missing_dict = analyze_missing_values(df_complete, "Test Dataset")
        
        # Should have all zeros
        assert all(count == 0 for count in missing_dict.values())
    
    def test_analyze_missing_values_with_missing(self):
        """Test analyze_missing_values with missing values."""
        df_with_missing = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': ['a', 'b', None],
            'C': [1.0, 2.0, 3.0]
        })
        
        missing_dict = analyze_missing_values(df_with_missing, "Test Dataset")
        
        # Should correctly count missing values
        assert missing_dict['A'] == 1
        assert missing_dict['B'] == 1
        assert missing_dict['C'] == 0
    
    def test_analyze_missing_values_empty_dataframe(self):
        """Test analyze_missing_values with empty DataFrame."""
        df_empty = pd.DataFrame()
        
        missing_dict = analyze_missing_values(df_empty, "Empty Dataset")
        
        assert isinstance(missing_dict, dict)
        assert len(missing_dict) == 0


class TestGetDataInfo:
    """Test suite for get_data_info function."""
    
    def setup_method(self):
        """Set up test data for data info analysis."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='W-SAT')
        self.df_sample = pd.DataFrame({
            'ISBN': ['123'] * len(dates),
            'Title': ['Book A'] * len(dates),
            'Volume': [100 + i for i in range(len(dates))],
            'Value': [10.0 + i for i in range(len(dates))],
            'Author': ['Author A'] * len(dates)
        }, index=dates)
        self.df_sample.index.name = 'End Date'
        
        # Add some missing values
        self.df_sample.loc[self.df_sample.index[0], 'Author'] = np.nan
        self.df_sample.loc[self.df_sample.index[1], 'Volume'] = np.nan
    
    def test_get_data_info_structure(self):
        """Test that get_data_info returns correct structure."""
        info = get_data_info(self.df_sample)
        
        required_keys = [
            'shape', 'memory_usage_mb', 'columns', 'dtypes',
            'date_range', 'unique_isbns', 'unique_titles',
            'missing_values', 'total_missing', 'missing_percentage',
            'numeric_columns', 'numeric_summary'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_get_data_info_shape_and_memory(self):
        """Test shape and memory usage calculations."""
        info = get_data_info(self.df_sample)
        
        assert info['shape'] == self.df_sample.shape
        assert info['memory_usage_mb'] > 0
        assert isinstance(info['memory_usage_mb'], float)
    
    def test_get_data_info_unique_counts(self):
        """Test unique count calculations."""
        info = get_data_info(self.df_sample)
        
        assert info['unique_isbns'] == 1  # Only ISBN '123'
        assert info['unique_titles'] == 1  # Only 'Book A'
    
    def test_get_data_info_missing_values(self):
        """Test missing values analysis."""
        info = get_data_info(self.df_sample)
        
        assert info['missing_values']['Author'] == 1
        assert info['missing_values']['Volume'] == 1
        assert info['total_missing'] == 2
        assert info['missing_percentage'] > 0
    
    def test_get_data_info_numeric_summary(self):
        """Test numeric columns summary."""
        info = get_data_info(self.df_sample)
        
        assert 'Volume' in info['numeric_columns']
        assert 'Value' in info['numeric_columns']
        assert 'Volume' in info['numeric_summary']
        assert 'Value' in info['numeric_summary']
    
    def test_get_data_info_date_range(self):
        """Test date range calculation for datetime index."""
        info = get_data_info(self.df_sample)
        
        assert 'date_range' in info
        assert len(info['date_range']) == 2
        assert info['date_range'][0] == self.df_sample.index.min()
        assert info['date_range'][1] == self.df_sample.index.max()


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_get_isbn_to_title_mapping(self):
        """Test get_isbn_to_title_mapping function."""
        mapping = get_isbn_to_title_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check that all values are strings (titles)
        for isbn, title in mapping.items():
            assert isinstance(isbn, str)
            assert isinstance(title, str)
    
    def test_get_project_directories(self):
        """Test get_project_directories function."""
        directories = get_project_directories()
        
        required_keys = ['project_root', 'raw_data', 'processed_data']
        for key in required_keys:
            assert key in directories
            assert isinstance(directories[key], str)
    
    @patch('os.makedirs')
    def test_ensure_directory_exists(self, mock_makedirs):
        """Test ensure_directory_exists function."""
        test_path = '/test/path'
        result = ensure_directory_exists(test_path)
        
        mock_makedirs.assert_called_once_with(test_path, exist_ok=True)
        assert result == test_path


class TestIntegrationScenarios:
    """Integration-style tests for common preprocessing scenarios."""
    
    def setup_method(self):
        """Set up realistic test data for integration scenarios."""
        # Create realistic sales data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='W-SAT')
        
        # Multiple ISBNs with different patterns
        self.test_data = []
        isbns = ['9780722532935', '9780241003008', '9780140500875']
        titles = ['The Alchemist', 'Very Hungry Caterpillar', 'Very Hungry Caterpillar']
        
        for i, date in enumerate(dates):
            isbn = isbns[i % len(isbns)]
            title = titles[i % len(titles)]
            
            self.test_data.append({
                'ISBN': isbn,
                'End Date': date,
                'Volume': np.random.randint(50, 500),
                'Value': np.random.uniform(5.0, 50.0),
                'ASP': np.random.uniform(0.08, 0.12),
                'RRP': np.random.uniform(8.0, 25.0),
                'Title': title,
                'Author': f'Author {isbn[-3:]}',
                'Binding': np.random.choice(['Paperback', 'Hardcover']),
                'Publisher Group': 'Test Publisher'
            })
        
        self.df_realistic = pd.DataFrame(self.test_data)
    
    def test_full_preprocessing_chain_success(self):
        """Test that the full preprocessing chain works end-to-end."""
        # This tests the integration of multiple functions
        try:
            # Step 1: Convert data types
            df_converted = convert_data_types(self.df_realistic.copy())
            assert pd.api.types.is_datetime64_any_dtype(df_converted['End Date'])
            
            # Step 2: Prepare time series
            df_ts = prepare_time_series_data(df_converted)
            assert df_ts.index.name == 'End Date'
            
            # Step 3: Fill missing weeks (though our test data is complete)
            df_filled = fill_missing_weeks(df_ts)
            assert len(df_filled) >= len(df_ts)
            
            # Step 4: Filter by date
            df_filtered = filter_data_by_date(df_filled, '2023-06-01')
            assert df_filtered.index.min() >= pd.to_datetime('2023-06-01')
            
            # Step 5: Select specific books
            selected_isbns = ['9780722532935', '9780241003008']
            df_selected = select_specific_books(df_filtered, selected_isbns, '2023-01-01')
            assert set(df_selected['ISBN'].unique()).issubset(set(selected_isbns))
            
            # All steps completed successfully
            assert True
            
        except Exception as e:
            pytest.fail(f"Full preprocessing chain failed: {str(e)}")
    
    def test_data_quality_preservation(self):
        """Test that data quality is preserved through transformations."""
        original_row_count = len(self.df_realistic)
        original_isbn_count = self.df_realistic['ISBN'].nunique()
        
        # Process the data
        df_converted = convert_data_types(self.df_realistic.copy())
        df_ts = prepare_time_series_data(df_converted)
        
        # Check that basic data integrity is maintained
        assert len(df_ts) == original_row_count
        assert df_ts['ISBN'].nunique() == original_isbn_count
        
        # Check that no unexpected nulls were introduced in key columns
        assert df_ts['ISBN'].notna().all()
        assert df_ts['Volume'].notna().all()
    
    def test_error_propagation(self):
        """Test that errors are properly propagated through the chain."""
        # Create invalid data
        invalid_data = self.df_realistic.copy()
        invalid_data = invalid_data.drop(columns=['ISBN'])  # Remove required column
        
        with pytest.raises(ValueError, match="Missing required columns"):
            convert_data_types(invalid_data)


# Test fixtures and helpers
@pytest.fixture
def sample_sales_data():
    """Fixture providing sample sales data for tests."""
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='W-SAT')
    volumes = [100, 150, 200, 175, 125, 180, 160, 140, 190, 155, 165, 185, 175]
    values = [10.0, 15.0, 20.0, 17.5, 12.5, 18.0, 16.0, 14.0, 19.0, 15.5, 16.5, 18.5, 17.5]
    
    # Ensure all arrays have the same length
    min_length = min(len(dates), len(volumes), len(values))
    dates = dates[:min_length]
    volumes = volumes[:min_length]
    values = values[:min_length]
    
    return pd.DataFrame({
        'ISBN': ['123'] * min_length,
        'End Date': dates,
        'Volume': volumes,
        'Value': values,
        'Title': ['Test Book'] * min_length,
        'Author': ['Test Author'] * min_length
    })


@pytest.fixture
def isbn_reference_data():
    """Fixture providing ISBN reference data for tests."""
    return pd.DataFrame({
        'ISBN': ['123', '456', '789'],
        'Author': ['Author A', 'Author B', 'Author C'],
        'Publisher': ['Publisher A', 'Publisher B', 'Publisher C'],
        'Category': ['Fiction', 'Non-Fiction', 'Children']
    })


class TestFixtureUsage:
    """Test class demonstrating fixture usage."""
    
    def test_with_sample_data_fixture(self, sample_sales_data):
        """Test using the sample_sales_data fixture."""
        assert len(sample_sales_data) > 0
        assert 'ISBN' in sample_sales_data.columns
        assert 'Volume' in sample_sales_data.columns
    
    def test_with_isbn_reference_fixture(self, isbn_reference_data):
        """Test using the isbn_reference_data fixture."""
        assert len(isbn_reference_data) == 3
        assert 'Author' in isbn_reference_data.columns


# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test suite for performance and edge cases."""
    
    def test_edge_case_extreme_dates(self):
        """Test edge case with extreme date ranges."""
        extreme_dates = pd.DataFrame({
            'ISBN': ['123', '123'],
            # FIX: Change the second date to be within the filter range
            'End Date': ['1900-01-01', '2049-12-31'],
            'Volume': [100, 200],
            'Value': [10.0, 20.0],
            'Title': ['Old Book', 'Future Book']
        })
        
        # Should handle extreme dates without errors
        df_converted = convert_data_types(extreme_dates)
        df_filtered = filter_data_by_date(df_converted.set_index('End Date'), '1950-01-01', '2050-12-31')
        
        # Now, the assertion should be correct
        assert len(df_filtered) == 1
        assert df_filtered['Title'].iloc[0] == 'Future Book' # Optional: more specific check
    
    def test_edge_case_single_row(self):
        """Test edge case with single row of data."""
        single_row = pd.DataFrame({
            'ISBN': ['123'],
            'End Date': ['2023-01-01'],
            'Volume': [100],
            'Value': [10.0],
            'Title': ['Single Row Book']
        })
        
        # Should handle single row without errors
        df_converted = convert_data_types(single_row)
        df_ts = prepare_time_series_data(df_converted)
        
        assert len(df_ts) == 1
        assert df_ts.index.name == 'End Date'


if __name__ == "__main__":
    """
    Run the tests using pytest.
    
    Usage:
        python test_preprocessing.py
        or
        pytest test_preprocessing.py -v
    """
    pytest.main([__file__, "-v", "--tb=short"])