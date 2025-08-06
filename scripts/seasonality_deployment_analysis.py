"""
Comprehensive seasonality analysis for book sales data.

This script analyzes seasonal patterns across all books in the dataset
to identify which books show strong seasonality for SARIMA model deployment.

Output:
- CSV files with seasonality classifications for pipeline deployment
- Interactive visualizations
- Detailed analysis reports
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps._02_preprocessing import load_processed_data
from utils.plotting import save_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SeasonalityAnalyzer:
    """
    Comprehensive seasonality analysis for book sales data.
    """

    def __init__(self, data_path: str = None):
        """
        Initialize the analyzer.

        Args:
            data_path: Path to the processed data file
        """
        self.data_path = data_path or 'data/processed/processed_sales_data_filled.csv'
        self.df = None
        self.seasonality_df = None

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the sales data."""
        logger.info(f"Loading data from {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # Convert date and add time features
        self.df['End Date'] = pd.to_datetime(self.df['End Date'])
        self.df['Year'] = self.df['End Date'].dt.year
        self.df['Month'] = self.df['End Date'].dt.month
        self.df['Week'] = self.df['End Date'].dt.isocalendar().week
        self.df['Quarter'] = self.df['End Date'].dt.quarter

        logger.info(f"Loaded {self.df.shape[0]:,} records for {self.df['ISBN'].nunique()} unique books")
        logger.info(f"Date range: {self.df['End Date'].min()} to {self.df['End Date'].max()}")

        return self.df

    def calculate_book_seasonality_metrics(self, book_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive seasonality metrics for a single book.

        Args:
            book_data: DataFrame with sales data for one book

        Returns:
            Dictionary with seasonality metrics
        """
        # Monthly aggregation
        monthly_volumes = book_data.groupby('Month')['Volume'].sum()
        quarterly_volumes = book_data.groupby('Quarter')['Volume'].sum()

        # Basic variability metrics
        monthly_cv = monthly_volumes.std() / monthly_volumes.mean() if monthly_volumes.mean() > 0 else 0
        quarterly_cv = quarterly_volumes.std() / quarterly_volumes.mean() if quarterly_volumes.mean() > 0 else 0

        # Christmas seasonality (Nov-Dec vs rest of year)
        nov_dec_vol = monthly_volumes.get(11, 0) + monthly_volumes.get(12, 0)
        jan_oct_vol = monthly_volumes.drop([11, 12], errors='ignore').sum()
        christmas_ratio = (nov_dec_vol / 2) / (jan_oct_vol / 10) if jan_oct_vol > 0 else 0

        # Summer seasonality (Jun-Aug vs rest)
        summer_vol = sum([monthly_volumes.get(m, 0) for m in [6, 7, 8]])
        other_vol = monthly_volumes.drop([6, 7, 8], errors='ignore').sum()
        summer_ratio = (summer_vol / 3) / (other_vol / 9) if other_vol > 0 else 0

        # Back-to-school seasonality (Aug-Sep vs rest)
        school_vol = monthly_volumes.get(8, 0) + monthly_volumes.get(9, 0)
        school_ratio = (school_vol / 2) / (other_vol / 10) if other_vol > 0 else 0

        # Peak detection
        peak_month = monthly_volumes.idxmax() if len(monthly_volumes) > 0 else None
        min_month = monthly_volumes.idxmin() if len(monthly_volumes) > 0 else None

        # Seasonal strength (multiple measures)
        seasonal_strength = monthly_volumes.max() / monthly_volumes.min() if monthly_volumes.min() > 0 else np.inf
        iqr_strength = monthly_volumes.quantile(0.75) / monthly_volumes.quantile(0.25) if monthly_volumes.quantile(0.25) > 0 else np.inf

        # Trend analysis
        years = book_data['Year'].unique()
        if len(years) > 1:
            yearly_volumes = book_data.groupby('Year')['Volume'].sum()
            trend_slope = stats.linregress(years, yearly_volumes.reindex(years, fill_value=0))[0]
        else:
            trend_slope = 0

        # Consistency of seasonal pattern across years
        seasonal_consistency = 0
        if len(years) >= 3:
            year_patterns = []
            for year in years:
                year_data = book_data[book_data['Year'] == year]
                if len(year_data) >= 12:  # At least 3 months of data
                    year_monthly = year_data.groupby('Month')['Volume'].sum()
                    year_patterns.append(year_monthly.reindex(range(1, 13), fill_value=0))

            if len(year_patterns) >= 2:
                correlations = []
                for i in range(len(year_patterns) - 1):
                    corr = stats.pearsonr(year_patterns[i], year_patterns[i + 1])[0]
                    if not np.isnan(corr):
                        correlations.append(corr)
                seasonal_consistency = np.mean(correlations) if correlations else 0

        return {
            'Monthly_CV': monthly_cv,
            'Quarterly_CV': quarterly_cv,
            'Christmas_Ratio': christmas_ratio,
            'Summer_Ratio': summer_ratio,
            'School_Ratio': school_ratio,
            'Seasonal_Strength': seasonal_strength,
            'IQR_Strength': iqr_strength,
            'Peak_Month': peak_month,
            'Min_Month': min_month,
            'Trend_Slope': trend_slope,
            'Seasonal_Consistency': seasonal_consistency,
            'Monthly_Volumes': monthly_volumes.to_dict()
        }

    def analyze_all_books(self) -> pd.DataFrame:
        """
        Analyze seasonality patterns across all books in the dataset.

        Returns:
            DataFrame with seasonality analysis results
        """
        if self.df is None:
            self.load_data()

        # Get book metadata
        book_metadata = self.df.groupby(['ISBN']).agg({
            'Title': 'first',
            'Author': 'first',
            'Product Class': 'first',
            'Binding': 'first',
            'Publisher Group': 'first',
            'Volume': 'sum',
            'Value': 'sum',
            'End Date': ['min', 'max', 'count']
        }).reset_index()

        book_metadata.columns = ['ISBN', 'Title', 'Author', 'Product_Class', 'Binding',
                                'Publisher_Group', 'Total_Volume', 'Total_Value',
                                'Start_Date', 'End_Date', 'Data_Points']

        # Filter for books with sufficient data
        min_data_points = 52  # At least 1 year of weekly data
        min_volume = 500      # Minimum total volume for meaningful analysis

        viable_books = book_metadata[
            (book_metadata['Data_Points'] >= min_data_points) &
            (book_metadata['Total_Volume'] >= min_volume)
        ].copy()

        logger.info(f"Analyzing seasonality for {len(viable_books)} books with sufficient data...")

        # Analyze each book
        results = []

        for idx, book in viable_books.iterrows():
            isbn = book['ISBN']
            book_data = self.df[self.df['ISBN'] == isbn].copy()

            # Skip if data is too sparse
            if len(book_data) < 52:
                continue

            # Calculate seasonality metrics
            metrics = self.calculate_book_seasonality_metrics(book_data)

            # Combine with metadata
            result = {
                'ISBN': isbn,
                'Title': book['Title'],
                'Author': book['Author'],
                'Product_Class': book['Product_Class'],
                'Binding': book['Binding'],
                'Publisher_Group': book['Publisher_Group'],
                'Total_Volume': book['Total_Volume'],
                'Total_Value': book['Total_Value'],
                'Data_Points': book['Data_Points'],
                'Years_of_Data': (book['End_Date'] - book['Start_Date']).days / 365.25,
                **metrics
            }

            results.append(result)

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(viable_books)} books...")

        self.seasonality_df = pd.DataFrame(results)
        logger.info(f"Completed seasonality analysis for {len(self.seasonality_df)} books")

        return self.seasonality_df

    def classify_books_for_deployment(self) -> Dict[str, pd.DataFrame]:
        """
        Classify books into deployment categories based on seasonality.

        Returns:
            Dictionary with different book classifications
        """
        if self.seasonality_df is None:
            self.analyze_all_books()

        df = self.seasonality_df.copy()

        # Define classification criteria
        classifications = {}

        # 1. EXCELLENT for SARIMA - Strong, consistent seasonality
        excellent_criteria = (
            (df['Seasonal_Strength'] > 3.0) &
            (df['Monthly_CV'] > 0.4) &
            (df['Seasonal_Consistency'] > 0.3) &
            (df['Total_Volume'] > 1000) &
            (df['Years_of_Data'] >= 2)
        )
        classifications['excellent_sarima'] = df[excellent_criteria].copy()

        # 2. GOOD for SARIMA - Moderate seasonality
        good_criteria = (
            (df['Seasonal_Strength'] > 2.0) &
            (df['Monthly_CV'] > 0.3) &
            (df['Total_Volume'] > 500) &
            (df['Years_of_Data'] >= 1.5) &
            ~excellent_criteria  # Not already in excellent
        )
        classifications['good_sarima'] = df[good_criteria].copy()

        # 3. CHRISTMAS SEASONAL - Strong holiday patterns
        christmas_criteria = (
            (df['Christmas_Ratio'] > 2.0) &
            (df['Peak_Month'].isin([11, 12])) &
            (df['Total_Volume'] > 300)
        )
        classifications['christmas_seasonal'] = df[christmas_criteria].copy()

        # 4. CHILDREN'S BOOKS - Likely seasonal gift books
        children_criteria = (
            df['Product_Class'].str.contains('Children|Picture|Y1|Educational', case=False, na=False) |
            df['Title'].str.contains('Cat in the Hat|Hungry Caterpillar|Dr\. Seuss', case=False, na=False)
        )
        classifications['children_books'] = df[children_criteria].copy()

        # 5. LOW SEASONALITY - Better suited for simpler models
        low_seasonal_criteria = (
            (df['Seasonal_Strength'] < 1.8) &
            (df['Monthly_CV'] < 0.25) &
            (df['Christmas_Ratio'] < 1.3)
        )
        classifications['low_seasonal'] = df[low_seasonal_criteria].copy()

        # 6. HIGH VOLUME STABLE - Consistent sellers, good for any model
        high_volume_criteria = (
            (df['Total_Volume'] > 10000) &
            (df['Years_of_Data'] >= 3) &
            (df['Data_Points'] > 150)
        )
        classifications['high_volume_stable'] = df[high_volume_criteria].copy()

        # 7. RECOMMENDED FOR DEPLOYMENT - Combined best candidates
        deployment_criteria = (
            excellent_criteria |
            good_criteria |
            (christmas_criteria & (df['Total_Volume'] > 1000)) |
            (high_volume_criteria & (df['Seasonal_Strength'] > 1.5))
        )
        classifications['recommended_for_deployment'] = df[deployment_criteria].copy()

        return classifications

    def create_deployment_manifest(self, classifications: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a deployment manifest for the ML pipeline.

        Args:
            classifications: Dictionary of classified books

        Returns:
            DataFrame with deployment recommendations
        """
        # Create manifest with deployment priority
        manifest_data = []

        for category, books_df in classifications.items():
            if category == 'recommended_for_deployment':
                continue  # Skip aggregate category

            for _, book in books_df.iterrows():
                # Assign priority based on category
                if category == 'excellent_sarima':
                    priority = 1
                    model_recommendation = 'SARIMA_SEASONAL'
                    confidence = 'HIGH'
                elif category == 'good_sarima':
                    priority = 2
                    model_recommendation = 'SARIMA_MODERATE'
                    confidence = 'MEDIUM'
                elif category == 'christmas_seasonal':
                    priority = 2
                    model_recommendation = 'SARIMA_CHRISTMAS'
                    confidence = 'HIGH'
                elif category == 'children_books':
                    priority = 2
                    model_recommendation = 'SARIMA_GIFT_PATTERN'
                    confidence = 'MEDIUM'
                elif category == 'high_volume_stable':
                    priority = 3
                    model_recommendation = 'ARIMA_OR_EXPONENTIAL'
                    confidence = 'MEDIUM'
                else:  # low_seasonal
                    priority = 4
                    model_recommendation = 'SIMPLE_EXPONENTIAL'
                    confidence = 'LOW'

                manifest_data.append({
                    'ISBN': book['ISBN'],
                    'Title': book['Title'],
                    'Author': book['Author'],
                    'Product_Class': book['Product_Class'],
                    'Category': category,
                    'Priority': priority,
                    'Model_Recommendation': model_recommendation,
                    'Confidence': confidence,
                    'Total_Volume': book['Total_Volume'],
                    'Seasonal_Strength': book['Seasonal_Strength'],
                    'Christmas_Ratio': book['Christmas_Ratio'],
                    'Monthly_CV': book['Monthly_CV'],
                    'Years_of_Data': book['Years_of_Data'],
                    'Deployment_Ready': priority <= 3
                })

        manifest_df = pd.DataFrame(manifest_data)

        # Remove duplicates (books might be in multiple categories)
        manifest_df = manifest_df.sort_values(['ISBN', 'Priority']).groupby('ISBN').first().reset_index()

        return manifest_df

    def create_seasonality_visualizations(self) -> List[go.Figure]:
        """
        Create comprehensive seasonality visualizations.

        Returns:
            List of plotly figures
        """
        if self.seasonality_df is None:
            raise ValueError("Must run analyze_all_books() first")

        df = self.seasonality_df
        figures = []

        # 1. Seasonality Overview Dashboard
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Seasonal Strength vs Christmas Ratio',
                'Monthly Coefficient of Variation vs Total Volume',
                'Peak Month Distribution',
                'Seasonality by Product Class'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Scatter plot: Seasonal Strength vs Christmas Ratio
        fig1.add_trace(
            go.Scatter(
                x=df['Christmas_Ratio'],
                y=df['Seasonal_Strength'],
                mode='markers',
                text=df['Title'].str[:40] + '...',
                hovertemplate='<b>%{text}</b><br>' +
                             'Christmas Ratio: %{x:.2f}<br>' +
                             'Seasonal Strength: %{y:.2f}<br>' +
                             'Volume: %{customdata:,}<extra></extra>',
                customdata=df['Total_Volume'],
                marker=dict(
                    size=np.log(df['Total_Volume'] + 1) * 3,
                    color=df['Monthly_CV'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Monthly Coefficient of Variation", x=0.45)
                ),
                name='Books'
            ),
            row=1, col=1
        )

        # Volume vs CV with trend
        fig1.add_trace(
            go.Scatter(
                x=df['Total_Volume'],
                y=df['Monthly_CV'],
                mode='markers',
                text=df['Title'].str[:40] + '...',
                hovertemplate='<b>%{text}</b><br>' +
                             'Volume: %{x:,}<br>' +
                             'Monthly Coefficient of Variation: %{y:.3f}<extra></extra>',
                marker=dict(
                    size=8,
                    color=df['Christmas_Ratio'],
                    colorscale='Reds',
                    showscale=False
                ),
                name='CV vs Volume'
            ),
            row=1, col=2
        )

        # Peak month distribution
        peak_counts = df['Peak_Month'].value_counts().sort_index()
        fig1.add_trace(
            go.Bar(
                x=peak_counts.index,
                y=peak_counts.values,
                name='Peak Month Distribution',
                marker_color='lightblue',
                hovertemplate='Month %{x}: %{y} books<extra></extra>'
            ),
            row=2, col=1
        )

        # Product class seasonality
        class_stats = df.groupby('Product_Class')['Seasonal_Strength'].mean().sort_values(ascending=True).tail(15)
        fig1.add_trace(
            go.Bar(
                x=class_stats.values,
                y=[cls[:35] + '...' if len(cls) > 35 else cls for cls in class_stats.index],
                orientation='h',
                name='Avg Seasonal Strength',
                marker_color='orange',
                hovertemplate='%{y}<br>Avg Strength: %{x:.1f}x<extra></extra>'
            ),
            row=2, col=2
        )

        fig1.update_layout(
            title_text="Book Sales Seasonality Analysis Dashboard",
            height=900,
            showlegend=False
        )

        fig1.update_xaxes(title_text="Christmas Ratio", row=1, col=1)
        fig1.update_yaxes(title_text="Seasonal Strength", row=1, col=1)
        fig1.update_xaxes(title_text="Total Volume (log scale)", type="log", row=1, col=2)
        fig1.update_yaxes(title_text="Monthly Coefficient of Variation", row=1, col=2)
        fig1.update_xaxes(title_text="Month", row=2, col=1)
        fig1.update_yaxes(title_text="Number of Books", row=2, col=1)

        figures.append(fig1)

        # 2. Top Seasonal Books Detail View
        top_seasonal = df.nlargest(20, 'Seasonal_Strength')

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=top_seasonal['Seasonal_Strength'],
            y=top_seasonal['Title'].str[:45] + '...',
            orientation='h',
            marker=dict(
                color=top_seasonal['Christmas_Ratio'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Christmas Ratio")
            ),
            text=top_seasonal['Christmas_Ratio'].round(1),
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Seasonal Strength: %{x:.1f}x<br>' +
                         'Christmas Ratio: %{marker.color:.1f}x<br>' +
                         'Product Class: %{customdata}<extra></extra>',
            customdata=top_seasonal['Product_Class']
        ))

        fig2.update_layout(
            title="Top 20 Most Seasonal Books",
            xaxis_title="Seasonal Strength (Peak/Min Ratio)",
            yaxis_title="Book Title",
            height=800
        )

        figures.append(fig2)

        return figures

    def generate_comprehensive_report(self, classifications: Dict[str, pd.DataFrame],
                                    manifest_df: pd.DataFrame):
        """
        Generate a comprehensive text report of the seasonality analysis.

        Args:
            classifications: Dictionary of classified books
            manifest_df: Deployment manifest DataFrame
        """
        if self.seasonality_df is None:
            raise ValueError("Must run analysis first")

        df = self.seasonality_df

        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE BOOK SALES SEASONALITY ANALYSIS")
        print("=" * 100)

        print(f"\n📈 DATASET OVERVIEW:")
        print(f"   • Total books analyzed: {len(df):,}")
        print(f"   • Average seasonal strength: {df['Seasonal_Strength'].mean():.2f}x")
        print(f"   • Median seasonal strength: {df['Seasonal_Strength'].median():.2f}x")
        print(f"   • Books with Christmas boost > 2x: {len(df[df['Christmas_Ratio'] > 2.0]):,}")
        print(f"   • Books with December peak: {len(df[df['Peak_Month'] == 12]):,}")
        print(f"   • Books with November peak: {len(df[df['Peak_Month'] == 11]):,}")

        print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
        deployment_ready = manifest_df[manifest_df['Deployment_Ready'] == True]
        high_confidence = manifest_df[manifest_df['Confidence'] == 'HIGH']

        print(f"   ✅ Ready for SARIMA deployment: {len(deployment_ready):,} books")
        print(f"   🎯 High confidence predictions: {len(high_confidence):,} books")
        print(f"   📊 Total volume represented: {deployment_ready['Total_Volume'].sum():,} units")

        print(f"\n🏆 TOP SEASONAL BOOKS (Excellent SARIMA candidates):")
        excellent_books = classifications['excellent_sarima'].nlargest(10, 'Seasonal_Strength')
        for idx, book in excellent_books.iterrows():
            print(f"\n   {len(excellent_books) - idx:2d}. 📚 {book['Title'][:50]}")
            print(f"       Author: {book['Author']}")
            print(f"       Category: {book['Product_Class'][:60]}")
            print(f"       🔥 Seasonal Strength: {book['Seasonal_Strength']:.1f}x")
            print(f"       🎄 Christmas Boost: {book['Christmas_Ratio']:.1f}x")
            print(f"       📊 Monthly CV: {book['Monthly_CV']:.3f}")
            print(f"       📈 Total Volume: {book['Total_Volume']:,} units")
            print(f"       ⏱️  Data Span: {book['Years_of_Data']:.1f} years")

        print(f"\n🎄 CHRISTMAS SEASONAL BOOKS:")
        christmas_books = classifications['christmas_seasonal'].nlargest(15, 'Christmas_Ratio')
        for idx, book in christmas_books.iterrows():
            print(f"   • {book['Title'][:45]}... (Christmas: {book['Christmas_Ratio']:.1f}x, Peak: Month {book['Peak_Month']})")

        print(f"\n📚 PRODUCT CATEGORY INSIGHTS:")
        category_analysis = df.groupby('Product_Class').agg({
            'Seasonal_Strength': ['mean', 'count'],
            'Christmas_Ratio': 'mean',
            'Monthly_CV': 'mean',
            'Total_Volume': 'sum'
        }).round(2)

        category_analysis.columns = ['Avg_Seasonal_Strength', 'Book_Count', 'Avg_Christmas_Ratio', 'Avg_Monthly_CV', 'Total_Category_Volume']
        category_analysis = category_analysis.sort_values('Avg_Seasonal_Strength', ascending=False)

        print("\n   Top 10 most seasonal product categories:")
        for category, stats in category_analysis.head(10).iterrows():
            print(f"   • {category[:55]}...")
            print(f"     📊 Avg Seasonal Strength: {stats['Avg_Seasonal_Strength']:.1f}x")
            print(f"     🎄 Avg Christmas Boost: {stats['Avg_Christmas_Ratio']:.1f}x")
            print(f"     📖 Books: {stats['Book_Count']} | Volume: {stats['Total_Category_Volume']:,}")

        print(f"\n⚠️  LOW SEASONALITY BOOKS (Consider simpler models):")
        low_seasonal = classifications['low_seasonal'].nsmallest(10, 'Seasonal_Strength')
        for idx, book in low_seasonal.iterrows():
            print(f"   • {book['Title'][:45]}... (Strength: {book['Seasonal_Strength']:.1f}x, CV: {book['Monthly_CV']:.3f})")

        print(f"\n🚀 DEPLOYMENT SUMMARY BY MODEL TYPE:")
        model_counts = manifest_df['Model_Recommendation'].value_counts()
        for model, count in model_counts.items():
            total_vol = manifest_df[manifest_df['Model_Recommendation'] == model]['Total_Volume'].sum()
            print(f"   • {model}: {count:,} books ({total_vol:,} units)")

    def save_results(self, classifications: Dict[str, pd.DataFrame],
                    manifest_df: pd.DataFrame, figures: List[go.Figure]):
        """
        Save all analysis results to files.

        Args:
            classifications: Dictionary of classified books
            manifest_df: Deployment manifest
            figures: List of visualization figures
        """
        # Create output directories
        output_dir = "outputs/seasonality_analysis"
        plots_dir = "outputs/plots/interactive"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Save main analysis
        self.seasonality_df.to_csv(f"{output_dir}/full_seasonality_analysis.csv", index=False)
        logger.info(f"Saved full analysis to {output_dir}/full_seasonality_analysis.csv")

        # Save classifications
        for category, df_cat in classifications.items():
            df_cat.to_csv(f"{output_dir}/{category}_books.csv", index=False)
            logger.info(f"Saved {category} books: {len(df_cat)} entries")

        # Save deployment manifest
        manifest_df.to_csv(f"{output_dir}/deployment_manifest.csv", index=False)
        logger.info(f"Saved deployment manifest: {len(manifest_df)} books")

        # Save key deployment lists for pipeline integration
        deployment_ready = manifest_df[manifest_df['Deployment_Ready'] == True]

        # High priority SARIMA candidates
        sarima_candidates = deployment_ready[
            deployment_ready['Model_Recommendation'].str.contains('SARIMA')
        ]['ISBN'].tolist()

        with open(f"{output_dir}/sarima_deployment_isbns.txt", 'w') as f:
            f.write("# ISBNs recommended for SARIMA model deployment\n")
            f.write("# Generated by seasonality_deployment_analysis.py\n")
            f.write(f"# Total: {len(sarima_candidates)} books\n")
            f.write(f"# Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for isbn in sarima_candidates:
                f.write(f"{isbn}\n")

        # Save Python list format for easy pipeline integration
        with open(f"{output_dir}/sarima_deployment_isbns.py", 'w') as f:
            f.write("# SARIMA deployment ISBNs for ML pipeline integration\n")
            f.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total books: {len(sarima_candidates)}\n\n")
            f.write("SARIMA_DEPLOYMENT_ISBNS = [\n")
            for isbn in sarima_candidates:
                f.write(f"    '{isbn}',\n")
            f.write("]\n\n")
            f.write("# Usage in your pipeline:\n")
            f.write("# from outputs.seasonality_analysis.sarima_deployment_isbns import SARIMA_DEPLOYMENT_ISBNS\n")

        logger.info(f"Saved SARIMA deployment ISBNs: {len(sarima_candidates)} books")

        # Save visualizations
        for idx, fig in enumerate(figures):
            filename = f"{plots_dir}/seasonality_analysis_{idx+1}.html"
            save_plot(fig, filename)
            logger.info(f"Saved visualization to {filename}")

        # Create summary statistics file for pipeline integration
        summary_stats = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_books_analyzed': len(self.seasonality_df),
            'books_ready_for_deployment': len(deployment_ready),
            'sarima_candidates': len(sarima_candidates),
            'high_confidence_books': len(manifest_df[manifest_df['Confidence'] == 'HIGH']),
            'avg_seasonal_strength': float(self.seasonality_df['Seasonal_Strength'].mean()),
            'avg_christmas_ratio': float(self.seasonality_df['Christmas_Ratio'].mean()),
        }

        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(f"{output_dir}/analysis_summary.csv", index=False)

        logger.info("All results saved successfully!")

    def run_analysis(self):
        """
        Run the complete seasonality analysis pipeline.

        Returns:
            Tuple of (classifications, manifest_df, figures)
        """
        logger.info("Starting comprehensive seasonality analysis...")

        # Load and analyze data
        self.load_data()
        seasonality_df = self.analyze_all_books()

        # Classify books for deployment
        classifications = self.classify_books_for_deployment()

        # Create deployment manifest
        manifest_df = self.create_deployment_manifest(classifications)

        # Generate report
        self.generate_comprehensive_report(classifications, manifest_df)

        # Create visualizations
        figures = self.create_seasonality_visualizations()

        # Save all results
        self.save_results(classifications, manifest_df, figures)

        logger.info("Seasonality analysis completed successfully!")

        return classifications, manifest_df, figures


def create_pipeline_integration_files(sarima_isbns: List[str], output_dir: str):
    """
    Create integration files for ZenML pipeline deployment.

    Args:
        sarima_isbns: List of ISBNs recommended for SARIMA
        output_dir: Output directory for files
    """
    # Create a step configuration file
    step_config = f"""
# Generated SARIMA deployment configuration
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
# Books: {len(sarima_isbns)} ISBNs identified as seasonal

from typing import List

class SeasonalityConfig:
    \"\"\"Configuration for seasonal book selection in ML pipeline.\"\"\"

    # Books with strong seasonal patterns - recommended for SARIMA
    SEASONAL_BOOKS: List[str] = {sarima_isbns}

    # Seasonality thresholds used in analysis
    MIN_SEASONAL_STRENGTH = 2.0
    MIN_CHRISTMAS_RATIO = 1.5
    MIN_MONTHLY_CV = 0.3
    MIN_VOLUME = 500
    MIN_YEARS_DATA = 1.5

    @classmethod
    def get_seasonal_books(cls) -> List[str]:
        \"\"\"Get list of seasonal book ISBNs for SARIMA modeling.\"\"\"
        return cls.SEASONAL_BOOKS

    @classmethod
    def is_seasonal_book(cls, isbn: str) -> bool:
        \"\"\"Check if a book is classified as seasonal.\"\"\"
        return isbn in cls.SEASONAL_BOOKS
"""

    with open(f"{output_dir}/seasonality_config.py", 'w') as f:
        f.write(step_config)

    logger.info(f"Created pipeline integration config: {output_dir}/seasonality_config.py")


def quick_seasonality_check(isbn: str, data_path: str = None) -> Dict:
    """
    Quick seasonality check for a single book.

    Args:
        isbn: ISBN to check
        data_path: Path to data file

    Returns:
        Dictionary with seasonality metrics
    """
    analyzer = SeasonalityAnalyzer(data_path)
    analyzer.load_data()

    book_data = analyzer.df[analyzer.df['ISBN'] == isbn]
    if book_data.empty:
        return {'error': f'No data found for ISBN {isbn}'}

    metrics = analyzer.calculate_book_seasonality_metrics(book_data)

    # Add interpretation
    metrics['is_seasonal'] = (
        metrics['Seasonal_Strength'] > 2.0 and
        metrics['Monthly_CV'] > 0.3
    )
    metrics['is_christmas_seasonal'] = (
        metrics['Christmas_Ratio'] > 1.5 and
        metrics['Peak_Month'] in [11, 12]
    )
    metrics['sarima_recommended'] = (
        metrics['Seasonal_Strength'] > 2.0 and
        metrics['Monthly_CV'] > 0.3 and
        book_data['Volume'].sum() > 500
    )

    return metrics


def main():
    """
    Main function to run comprehensive seasonality analysis.
    """
    print("\n🔍 COMPREHENSIVE BOOK SALES SEASONALITY ANALYSIS")
    print("=" * 80)
    print("This analysis will:")
    print("✅ Identify books with strong seasonal patterns")
    print("✅ Classify books for SARIMA model deployment")
    print("✅ Create deployment manifests for your ML pipeline")
    print("✅ Generate interactive visualizations")
    print("✅ Save results for pipeline integration")
    print("=" * 80)

    # Initialize and run analyzer
    analyzer = SeasonalityAnalyzer()

    try:
        classifications, manifest_df, figures = analyzer.run_analysis()

        # Create pipeline integration files
        deployment_ready = manifest_df[manifest_df['Deployment_Ready'] == True]
        sarima_candidates = deployment_ready[
            deployment_ready['Model_Recommendation'].str.contains('SARIMA')
        ]['ISBN'].tolist()

        create_pipeline_integration_files(sarima_candidates, "outputs/seasonality_analysis")

        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Files saved to: outputs/seasonality_analysis/")
        print(f"📈 Visualizations saved to: outputs/plots/interactive/")
        print(f"🚀 Ready for pipeline deployment: {len(sarima_candidates)} books")

        # Quick summary for immediate use
        print(f"\n🎯 QUICK DEPLOYMENT SUMMARY:")
        print(f"   • Excellent SARIMA candidates: {len(classifications['excellent_sarima'])}")
        print(f"   • Good SARIMA candidates: {len(classifications['good_sarima'])}")
        print(f"   • Christmas seasonal books: {len(classifications['christmas_seasonal'])}")
        print(f"   • Children's books: {len(classifications['children_books'])}")
        print(f"   • Low seasonality books: {len(classifications['low_seasonal'])}")

        print(f"\n📁 KEY FILES FOR YOUR PIPELINE:")
        print(f"   • deployment_manifest.csv - Complete deployment guide")
        print(f"   • sarima_deployment_isbns.txt - List of SARIMA-ready ISBNs")
        print(f"   • sarima_deployment_isbns.py - Python import for pipeline")
        print(f"   • seasonality_config.py - Configuration class for ZenML")

        return True

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error: {e}")
        print("Please check your data files and dependencies.")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎉 Seasonality analysis completed successfully!")
        print(f"You can now integrate the results into your SARIMA deployment pipeline.")

        print(f"\n💡 NEXT STEPS:")
        print(f"1. Review the generated visualizations in outputs/plots/interactive/")
        print(f"2. Check the deployment manifest in outputs/seasonality_analysis/")
        print(f"3. Integrate sarima_deployment_isbns.py into your ZenML pipeline")
        print(f"4. Use the seasonality classifications to optimize model selection")

        print(f"\n🔗 PIPELINE INTEGRATION EXAMPLE:")
        print(f"```python")
        print(f"from outputs.seasonality_analysis.sarima_deployment_isbns import SARIMA_DEPLOYMENT_ISBNS")
        print(f"from outputs.seasonality_analysis.seasonality_config import SeasonalityConfig")
        print(f"")
        print(f"# In your ZenML step:")
        print(f"seasonal_books = SeasonalityConfig.get_seasonal_books()")
        print(f"for isbn in seasonal_books:")
        print(f"    if SeasonalityConfig.is_seasonal_book(isbn):")
        print(f"        # Deploy SARIMA model for this book")
        print(f"        pass")
        print(f"```")
    else:
        print(f"\n❌ Analysis failed. Please check the error messages above.")
