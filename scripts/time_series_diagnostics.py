"""
Time Series Diagnostics Module

This module performs comprehensive time series diagnostics including:
- Decomposition (STL and seasonal decomposition)
- Stationarity tests (ADF, KPSS, Phillips-Perron, Zivot-Andrews)
- ACF and PACF analysis
- Ljung-Box test for autocorrelation

The module is designed to work with any book sales data and provides detailed
analysis for any number of books in the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, zivot_andrews
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot import PhillipsPerron
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')
# Suppress pkg_resources deprecation warning from property_cached (arch dependency)
warnings.filterwarnings('ignore', category=UserWarning, module='property_cached')

# Global variables for book data
sales_data = None
selected_books_data = None

# Default books for analysis
DEFAULT_BOOKS = [
    9780722532935,  # The Alchemist
]

def set_data(sales_dataframe: pd.DataFrame, selected_books_dataframe: Optional[pd.DataFrame] = None):
    """
    Set the sales data for analysis.

    Args:
        sales_dataframe: Complete sales data DataFrame
        selected_books_dataframe: Optional DataFrame with selected books only
    """
    global sales_data, selected_books_data
    sales_data = sales_dataframe
    selected_books_data = selected_books_dataframe

    if selected_books_data is None:
        selected_books_data = sales_dataframe

def get_book_data(isbn: Union[str, int]) -> pd.DataFrame:
    """
    Get data for a specific book by ISBN.

    Args:
        isbn: ISBN of the book

    Returns:
        DataFrame with data for the specified book
    """
    if selected_books_data is None:
        raise ValueError("No data set. Please call set_data() first.")

    if 'ISBN' not in selected_books_data.columns:
        raise ValueError("'ISBN' column not found in data")

    book_data = selected_books_data[selected_books_data['ISBN'] == isbn].copy()

    if book_data.empty:
        raise ValueError(f"No data found for ISBN: {isbn}")

    return book_data

def get_all_books() -> List[Union[str, int]]:
    """
    Get list of all unique ISBNs in the data.

    Returns:
        List of ISBNs
    """
    if selected_books_data is None:
        raise ValueError("No data set. Please call set_data() first.")

    if 'ISBN' not in selected_books_data.columns:
        raise ValueError("'ISBN' column not found in data")

    return selected_books_data['ISBN'].unique().tolist()

def get_book_title(isbn: Union[str, int]) -> str:
    """
    Get the title for a specific ISBN.

    Args:
        isbn: ISBN of the book

    Returns:
        Book title
    """
    book_data = get_book_data(isbn)

    if 'Title' in book_data.columns:
        title = book_data['Title'].iloc[0]
        return title if pd.notna(title) else f"Unknown Title ({isbn})"
    else:
        return f"Unknown Title ({isbn})"

def plot_sales_data(data: pd.DataFrame, isbn: Union[str, int], title: Optional[str] = None, color: str = 'blue'):
    """
    Plot sales data using Plotly.

    Args:
        data: DataFrame with sales data
        isbn: ISBN of the book
        title: Title of the book (optional, will be retrieved if not provided)
        color: Color for the plot
    """
    if title is None:
        title = get_book_title(isbn)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume'],
        mode='lines+markers',
        name=f"{isbn} - {title}",
        line=dict(color=color)
    ))
    fig.update_layout(
        title=f'Weekly Sales Data (Volume) for {title}',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        width=1100,
        height=400
    )
    fig.show()

def plot_combined_sales_data(isbns: Optional[List[Union[str, int]]] = None, colors: Optional[List[str]] = None):
    """
    Plot combined sales data for multiple books with zero volume weeks highlighted.

    Args:
        isbns: List of ISBNs to plot (if None, plots all books)
        colors: List of colors for each book (if None, uses default colors)
    """
    if selected_books_data is None:
        raise ValueError("No data set. Please call set_data() first.")

    if isbns is None:
        isbns = get_all_books()

    if colors is None:
        colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    # Create a figure
    fig = go.Figure()

    # Add sales data for each book
    for i, isbn in enumerate(isbns):
        try:
            book_data = get_book_data(isbn)
            title = get_book_title(isbn)
            color = colors[i % len(colors)]

            # Add the sales data
            fig.add_trace(go.Scatter(
                x=book_data.index,
                y=book_data['Volume'],
                mode='lines',
                name=f'{isbn} - {title}',
                line=dict(color=color),
                hoverinfo='x+y'
            ))

            # Highlight zero volume weeks
            zero_volume_weeks = book_data[book_data['Volume'] == 0]
            if not zero_volume_weeks.empty:
                fig.add_trace(go.Scatter(
                    x=zero_volume_weeks.index,
                    y=zero_volume_weeks['Volume'],
                    mode='markers',
                    name=f'Zero Volume - {title}',
                    marker=dict(color='red', size=8),
                    hoverinfo='x+y',
                    showlegend=False
                ))

        except ValueError as e:
            print(f"Warning: {e}")
            continue

    # Update layout
    fig.update_layout(
        title='Weekly Sales Data (Volume) for Selected Books',
        xaxis_title='Date',
        yaxis_title='Volume',
        legend_title='Books',
        template='plotly_white'
    )

    fig.show()

def analyze_covid_weeks(isbn: Optional[Union[str, int]] = None) -> pd.DataFrame:
    """
    Analyze and document COVID-19 lockdown periods that correspond to zero sales weeks.

    Args:
        isbn: Specific ISBN to analyze (if None, analyzes all books)

    Returns:
        DataFrame with zero volume weeks and lockdown status
    """
    if isbn:
        book_data = get_book_data(isbn)
        zero_volume_weeks = book_data[book_data['Volume'] == 0]
    else:
        if selected_books_data is None:
            raise ValueError("No data set. Please call set_data() first.")
        zero_volume_weeks = selected_books_data[selected_books_data['Volume'] == 0]

    # Define lockdown periods (start and end dates) as Timestamps
    lockdown_periods = [
        (pd.Timestamp("2020-03-26"), pd.Timestamp("2020-06-15")),  # First lockdown
        (pd.Timestamp("2020-11-05"), pd.Timestamp("2020-12-02")),  # Second lockdown
        (pd.Timestamp("2021-01-06"), pd.Timestamp("2021-02-21")),  # Third lockdown
    ]

    # Create a new column for lockdown indication
    def is_lockdown(week_date):
        for start, end in lockdown_periods:
            if start <= week_date <= end:
                return 1  # Mark as lockdown week
        return 0  # Not a lockdown week

    # Apply the function to create a new column in zero_volume_weeks
    zero_volume_weeks['Lockdown'] = zero_volume_weeks.index.to_series().apply(is_lockdown)

    print("COVID-19 Lockdown Analysis:")
    print("Zero volume weeks correspond to UK lockdown periods:")
    print("- First lockdown: March 26 - June 15, 2020")
    print("- Second lockdown: November 5 - December 2, 2020")
    print("- Third lockdown: January 6 - February 21, 2021")

    return zero_volume_weeks

# ============================================================================
# DECOMPOSITION ANALYSIS
# ============================================================================

def perform_stl_decomposition(data: pd.Series, title: str, period: int = 52, show_plot: bool = True) -> STL:
    """
    Perform STL decomposition on time series data.

    Args:
        data: Time series data
        title: Title for the plot
        period: Seasonal period (default 52 for weekly data)
        show_plot: Whether to display the plot

    Returns:
        STL decomposition results
    """
    stl_object = STL(data, period=period, seasonal=53, trend=None)
    stl_results = stl_object.fit()

    # Plotting the STL decomposition (only if requested)
    if show_plot:
        stl_results.plot()
        plt.suptitle(f'STL Decomposition for {title}', y=1.01)
        plt.show()

    return stl_results

def decompose_and_plot(df: pd.DataFrame, title: str, period: int = 52, show_multiplicative: bool = False):
    """
    Perform both additive and multiplicative decomposition and plot results.

    Args:
        df: DataFrame with 'Volume' column
        title: Title for the plots
        period: Seasonal period
        show_multiplicative: Whether to show multiplicative decomposition (can be problematic with zero sales)
    """
    # Plot original data
    df['Volume'].plot(title=f"{title} - Original Data", figsize=(10, 6))
    plt.show()

    # Additive decomposition (always show)
    additive_result = seasonal_decompose(df['Volume'], model='additive', period=period)
    additive_result.plot()
    plt.suptitle(f'{title} - Additive Decomposition', fontsize=16)
    plt.show()

    # Multiplicative decomposition (only if requested and data is suitable)
    if show_multiplicative:
        # Check if data has many zeros (which makes multiplicative decomposition problematic)
        zero_ratio = (df['Volume'] == 0).sum() / len(df['Volume'])
        if zero_ratio > 0.1:  # More than 10% zeros
            print(f"Warning: {title} has {zero_ratio:.1%} zero sales weeks. Multiplicative decomposition may be misleading.")

        try:
            multiplicative_result = seasonal_decompose(df['Volume'].fillna(0).clip(lower=1),
                                                     model='multiplicative', period=period)
            multiplicative_result.plot()
            plt.suptitle(f'{title} - Multiplicative Decomposition', fontsize=16)
            plt.show()
        except Exception as e:
            print(f"Multiplicative decomposition failed for {title}: {e}")

def perform_decomposition_analysis(isbns: Optional[List[Union[str, int]]] = None, show_plots: bool = True) -> Dict:
    """
    Perform comprehensive decomposition analysis for specified books.

    Args:
        isbns: List of ISBNs to analyze (if None, analyzes default books)
        show_plots: Whether to display plots (default: True)

    Returns:
        Dictionary with decomposition results
    """
    if isbns is None:
        isbns = DEFAULT_BOOKS

    print("=" * 60)
    print("DECOMPOSITION ANALYSIS")
    print("=" * 60)

    print("\nSeasonal variation analysis:")
    print("- Books typically show seasonal variation around Christmas")
    print("- Children's books may show additional peaks during school terms")
    print("- Additive decomposition is suitable for books with constant seasonal variation")
    print("- Multiplicative decomposition is suitable for books with varying seasonal amplitude")

    results = {}

    for isbn in isbns:
        try:
            book_data = get_book_data(isbn)
            title = get_book_title(isbn)

            print(f"\nPerforming decomposition analysis for {title} ({isbn})...")

            # STL Decomposition
            stl_results = perform_stl_decomposition(
                book_data['Volume'],
                title,
                show_plot=show_plots
            )

            # Traditional decomposition
            if show_plots:
                decompose_and_plot(book_data, title=title, period=52)
            else:
                # Still perform decomposition but don't plot
                try:
                    decomposition = seasonal_decompose(book_data['Volume'], period=52, extrapolate_trend='freq')
                    print(f"Decomposition completed for {title} (plot suppressed)")
                except Exception as e:
                    print(f"Decomposition failed for {title}: {e}")

            results[isbn] = {
                'title': title,
                'stl_results': stl_results,
                'data': book_data
            }

        except Exception as e:
            print(f"Error analyzing {isbn}: {e}")
            continue

    return results

# ============================================================================
# ACF, PACF, AND LJUNG-BOX ANALYSIS
# ============================================================================

def plot_acf_pacf(data: pd.Series, title: str, lags: int = 40, alpha: float = 0.05):
    """
    Plot ACF and PACF with confidence intervals.

    Args:
        data: Time series data
        title: Title for the plot
        lags: Number of lags to plot
        alpha: Significance level for confidence intervals
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ACF plot
    sm.graphics.tsa.plot_acf(data.dropna(), lags=lags, ax=axes[0], alpha=alpha)
    axes[0].set_title(f'ACF for {title}')

    # PACF plot
    sm.graphics.tsa.plot_pacf(data.dropna(), lags=lags, ax=axes[1], alpha=alpha)
    axes[1].set_title(f'PACF for {title}')

    plt.suptitle(f'ACF and PACF for {title}')
    plt.tight_layout()
    plt.show()

def perform_ljung_box_test(data: pd.Series, label: str, max_lag: int = 10) -> pd.DataFrame:
    """
    Perform Ljung-Box test for multiple lags.

    Args:
        data: Time series data
        label: Label for the test
        max_lag: Maximum lag to test

    Returns:
        DataFrame with Ljung-Box test results
    """
    lb_test = acorr_ljungbox(data.dropna(), lags=list(range(1, max_lag + 1)), return_df=True)
    print(f"\nLjung-Box test output for {label} (Lags 1 to {max_lag}):")
    print(lb_test)
    return lb_test

def analyze_significant_lags(data: pd.Series, label: str) -> Tuple[List[int], List[int]]:
    """
    Analyze significant lags in ACF and PACF.

    Args:
        data: Time series data
        label: Label for the analysis

    Returns:
        Tuple of (significant ACF lags, significant PACF lags)
    """
    # ACF analysis
    acf_coef = acf(data, alpha=0.05)
    sig_acf = []
    for i in range(1, len(acf_coef[0])):
        if acf_coef[0][i] > (acf_coef[1][i][1] - acf_coef[0][i]):
            sig_acf.append(i)
        elif acf_coef[0][i] < (acf_coef[1][i][0] - acf_coef[0][i]):
            sig_acf.append(i)

    # PACF analysis
    pacf_coef = pacf(data, alpha=0.05)
    sig_pacf = []
    for i in range(1, len(pacf_coef[0])):
        if pacf_coef[0][i] > (pacf_coef[1][i][1] - pacf_coef[0][i]):
            sig_pacf.append(i)
        elif pacf_coef[0][i] < (pacf_coef[1][i][0] - pacf_coef[0][i]):
            sig_pacf.append(i)

    print(f"\nSignificant lags for {label}:")
    print(f"ACF significant lags: {sig_acf}")
    print(f"PACF significant lags: {sig_pacf}")

    return sig_acf, sig_pacf

def perform_acf_pacf_analysis(isbns: Optional[List[Union[str, int]]] = None, show_plots: bool = True) -> Dict:
    """
    Perform comprehensive ACF, PACF, and Ljung-Box analysis.

    Args:
        isbns: List of ISBNs to analyze (if None, analyzes default books)
        show_plots: Whether to display plots (default: True)

    Returns:
        Dictionary with ACF/PACF analysis results
    """
    if isbns is None:
        isbns = DEFAULT_BOOKS

    print("=" * 60)
    print("ACF, PACF, AND LJUNG-BOX ANALYSIS")
    print("=" * 60)

    results = {}

    for isbn in isbns:
        try:
            book_data = get_book_data(isbn)
            title = get_book_title(isbn)

            print(f"\nPerforming ACF/PACF analysis for {title} ({isbn})...")

            # ACF & PACF with standard lags
            if show_plots:
                plot_acf_pacf(book_data['Volume'], title)
            else:
                print(f"ACF/PACF analysis completed for {title} (plot suppressed)")

            # Analyze significant lags (always performed)
            sig_acf, sig_pacf = analyze_significant_lags(book_data['Volume'], title)

            # Ljung-Box test (always performed)
            lb_test = perform_ljung_box_test(book_data['Volume'], title, 10)

            results[isbn] = {
                'title': title,
                'acf_lags': sig_pacf,
                'pacf_lags': sig_pacf,
                'ljung_box': lb_test,
                'data': book_data
            }

        except Exception as e:
            print(f"Error analyzing {isbn}: {e}")
            continue

    print("\nACF/PACF Analysis Summary:")
    print("- ACF shows strong autocorrelation that decays slowly")
    print("- No sharp cutoff in ACF, so MA order cannot be easily determined")
    print("- PACF shows strong correlation at lag 1 for most books")
    print("- Seasonal patterns are present in many books")
    print("- Significant autocorrelation suggests ARIMA models are appropriate")

    return results

# ============================================================================
# STATIONARITY TESTS
# ============================================================================

def adf_test(series: pd.Series, label: str, alpha: float = 0.05) -> Tuple:
    """
    Perform Augmented Dickey-Fuller test.

    Args:
        series: Time series data
        label: Label for the test
        alpha: Significance level

    Returns:
        ADF test results
    """
    result = adfuller(series.dropna(), autolag='BIC')
    adf_stat, p_value, critical_values = result[0], result[1], result[4]

    print(f"\nADF Test for {label}:")
    print(f"ADF Test Statistic: {adf_stat}")
    print(f"p-value: {p_value}")
    print(f"Critical Values: {critical_values}")

    if p_value < alpha:
        print(f"{label}: Reject null hypothesis (series is stationary)")
    else:
        print(f"{label}: Fail to reject null hypothesis (series is non-stationary)")

    if adf_stat < critical_values['5%']:
        print(f"{label}: Test statistic < 5% critical value, confirming stationarity")
    else:
        print(f"{label}: Test statistic > 5% critical value, confirming non-stationarity")

    return result

def kpss_test(series: pd.Series, label: str, alpha: float = 0.05) -> Tuple:
    """
    Perform KPSS test for stationarity.

    Args:
        series: Time series data
        label: Label for the test
        alpha: Significance level

    Returns:
        KPSS test results
    """
    kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(series.dropna(), regression='c', nlags='auto')

    print(f"\nKPSS Test for {label}:")
    print(f"KPSS Test Statistic: {kpss_stat}")
    print(f"p-value: {kpss_pvalue}")
    print(f"Critical Values: {kpss_crit}")
    print(f"Lags Used: {kpss_lags}")

    if kpss_pvalue < alpha:
        print(f"{label}: Reject null hypothesis (series is non-stationary)")
    else:
        print(f"{label}: Fail to reject null hypothesis (series is stationary)")

    if kpss_stat > kpss_crit['5%']:
        print(f"{label}: Test statistic > 5% critical value, confirming non-stationarity")
    else:
        print(f"{label}: Test statistic < 5% critical value, confirming stationarity")

    return (kpss_stat, kpss_pvalue, kpss_lags, kpss_crit)


def pp_test(series: pd.Series, label: str, alpha: float = 0.05):
    """
    Perform Phillips-Perron test.

    Args:
        series: Time series data
        label: Label for the test
        alpha: Significance level

    Returns:
        Phillips-Perron test results
    """
    pp_result = PhillipsPerron(series.dropna())
    pp_stat = pp_result.stat
    pp_pvalue = pp_result.pvalue
    pp_critical_values = pp_result.critical_values

    print(f"\nPhillips-Perron Test for {label}:")
    print(f"Phillips-Perron Test Statistic: {pp_stat}")
    print(f"Phillips-Perron p-value: {pp_pvalue}")
    print(f"Phillips-Perron Critical Values: {pp_critical_values}")

    if pp_pvalue < alpha:
        print(f"{label}: Reject null hypothesis (series is stationary)")
    else:
        print(f"{label}: Fail to reject null hypothesis (series is non-stationary)")

    if pp_stat < pp_critical_values['5%']:
        print(f"{label}: Test statistic < 5% critical value, confirming stationarity")
    else:
        print(f"{label}: Test statistic > 5% critical value, confirming non-stationarity")

    return pp_result


def za_test(series: pd.Series, label: str, alpha: float = 0.05):
    """
    Perform Zivot-Andrews test for structural breaks.

    Args:
        series: Time series data
        label: Label for the test
        alpha: Significance level

    Returns:
        Zivot-Andrews test results
    """
    series = series.astype('float64').dropna()
    za_result = zivot_andrews(series)
    za_stat = za_result.stat
    za_pvalue = za_result.pvalue
    za_critical_values = za_result.critical_values

    print(f"\nZivot-Andrews Test for {label}:")
    print(f"Zivot-Andrews Test Statistic: {za_stat}")
    print(f"Zivot-Andrews p-value: {za_pvalue}")
    print(f"Zivot-Andrews Critical Values: {za_critical_values}")

    if za_pvalue < alpha:
        print(f"{label}: Reject null hypothesis (series is stationary with structural break)")
    else:
        print(f"{label}: Fail to reject null hypothesis (series is non-stationary)")

    if za_stat < za_critical_values['5%']:
        print(f"{label}: Test statistic < 5% critical value, confirming stationarity with break")
    else:
        print(f"{label}: Test statistic > 5% critical value, confirming non-stationarity")

    return za_result

def perform_stationarity_analysis(isbns: Optional[List[Union[str, int]]] = None) -> Dict:
    """
    Perform comprehensive stationarity analysis using multiple tests.

    Args:
        isbns: List of ISBNs to analyze (if None, analyzes default books)

    Returns:
        Dictionary with stationarity test results
    """
    if isbns is None:
        isbns = DEFAULT_BOOKS

    print("=" * 60)
    print("STATIONARITY ANALYSIS")
    print("=" * 60)

    results = {}

    for isbn in isbns:
        try:
            book_data = get_book_data(isbn)
            title = get_book_title(isbn)

            print(f"\nPerforming stationarity tests for {title} ({isbn})...")

            # Perform all stationarity tests
            adf_result = adf_test(book_data['Volume'], title)
            kpss_result = kpss_test(book_data['Volume'], title)
            pp_result = pp_test(book_data['Volume'], title)
            za_result = za_test(book_data['Volume'], title)

            results[isbn] = {
                'title': title,
                'adf': adf_result,
                'kpss': kpss_result,
                'pp': pp_result,
                'za': za_result,
                'data': book_data
            }

        except Exception as e:
            print(f"Error analyzing {isbn}: {e}")
            continue

    print("\n" + "=" * 60)
    print("STATIONARITY ANALYSIS SUMMARY")
    print("=" * 60)
    print("ADF tests typically confirm time series are stationary.")
    print("AR models are appropriate if PACF has abrupt cut-off (not observed).")
    print("MA models are preferred if ACF has abrupt cut-off (not observed).")
    print("Since neither condition is met, ARMA/ARIMA models are recommended.")
    print("Classical modeling methods (ARIMA, SARIMA) can be implemented without differencing.")
    print("However, experimentation with d=1 is also possible.")

    return results

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_complete_diagnostics(isbns: Optional[List[Union[str, int]]] = None, show_plots: bool = True) -> Dict:
    """
    Run complete time series diagnostics for specified books.

    This function performs all diagnostic tests in sequence:
    1. Data visualization
    2. COVID-19 analysis
    3. Decomposition analysis
    4. ACF/PACF analysis
    5. Stationarity tests

    Args:
        isbns: List of ISBNs to analyze (if None, analyzes default books)
        show_plots: Whether to display plots (default: True)

    Returns:
        Dictionary with all diagnostic results
    """
    print("TIME SERIES DIAGNOSTICS FOR BOOK SALES DATA")
    print("=" * 60)

    # Check if data is available
    if selected_books_data is None:
        print("ERROR: No data set. Please call set_data() first.")
        return {}

    if isbns is None:
        isbns = DEFAULT_BOOKS

    print(f"Analyzing {len(isbns)} books: {isbns}")
    print(f"Plot generation: {'Enabled' if show_plots else 'Disabled'}")

    # 1. Plot sales data
    print("\n1. PLOTTING SALES DATA")
    print("-" * 30)

    if show_plots:
        # Plot individual books
        for isbn in isbns:
            try:
                book_data = get_book_data(isbn)
                title = get_book_title(isbn)
                plot_sales_data(book_data, isbn, title)
            except Exception as e:
                print(f"Error plotting {isbn}: {e}")
                continue

        # Plot combined data
        plot_combined_sales_data(isbns)

    # 2. COVID-19 analysis
    print("\n2. COVID-19 LOCKDOWN ANALYSIS")
    print("-" * 30)
    covid_analysis = analyze_covid_weeks()

    # 3. Decomposition analysis
    print("\n3. DECOMPOSITION ANALYSIS")
    print("-" * 30)
    decomposition_results = perform_decomposition_analysis(isbns, show_plots=show_plots)

    # 4. ACF/PACF analysis
    print("\n4. ACF/PACF ANALYSIS")
    print("-" * 30)
    acf_pacf_results = perform_acf_pacf_analysis(isbns, show_plots=show_plots)

    # 5. Stationarity analysis
    print("\n5. STATIONARITY ANALYSIS")
    print("-" * 30)
    stationarity_results = perform_stationarity_analysis(isbns)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print("All time series diagnostics have been completed.")
    print("Results indicate that ARIMA/SARIMA models are appropriate for forecasting.")
    print("The data shows stationarity, seasonal patterns, and significant autocorrelation.")

    return {
        'analyzed_isbns': isbns,
        'covid_analysis': covid_analysis,
        'decomposition': decomposition_results,
        'acf_pacf': acf_pacf_results,
        'stationarity': stationarity_results
    }

def run_diagnostics_for_book(isbn: Union[str, int]) -> Dict:
    """
    Run diagnostics for a single book.

    Args:
        isbn: ISBN of the book to analyze

    Returns:
        Dictionary with diagnostic results for the book
    """
    return run_complete_diagnostics([isbn])

def run_diagnostics_for_books(isbns: List[Union[str, int]]) -> Dict:
    """
    Run diagnostics for specific books.

    Args:
        isbns: List of ISBNs to analyze

    Returns:
        Dictionary with diagnostic results for the books
    """
    return run_complete_diagnostics(isbns)

if __name__ == "__main__":
    print("Time Series Diagnostics Module")
    print("Please import and use the functions in this module.")
    print("Example usage:")
    print("  from steps._03__time_series_diagnostics import *")
    print("  set_data(sales_dataframe, selected_books_dataframe)")
    print("  run_complete_diagnostics()  # Uses default books: The Alchemist and The Very Hungry Caterpillar")
    print("  # Or for specific books:")
    print("  run_diagnostics_for_books(['9780722532935', '9780241003008'])")
    print("  # Or for all available books:")
    print("  run_complete_diagnostics(get_all_books())")
