"""
Analyze correlation between book seasonality patterns and revenue value.

This script investigates whether the most seasonal books are also the most valuable,
helping to prioritize forecasting efforts on books that are both seasonal and profitable.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plotting import save_plot

def load_seasonality_data():
    """Load the seasonality analysis results."""
    data_path = "outputs/seasonality_analysis/full_seasonality_analysis.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded seasonality data for {len(df):,} books")
    return df

def calculate_value_metrics(df):
    """Calculate additional value-related metrics."""
    df = df.copy()
    
    # Revenue per unit
    df['Revenue_Per_Unit'] = df['Total_Value'] / df['Total_Volume']
    
    # Revenue per data point (efficiency)
    df['Revenue_Per_Week'] = df['Total_Value'] / df['Data_Points']
    
    # Volume efficiency 
    df['Volume_Per_Week'] = df['Total_Volume'] / df['Data_Points']
    
    # Value ranking
    df['Value_Rank'] = df['Total_Value'].rank(ascending=False)
    df['Seasonality_Rank'] = df['Seasonal_Strength'].rank(ascending=False)
    
    return df

def analyze_correlations(df):
    """Analyze correlations between seasonality and value metrics."""
    
    # Key seasonality metrics
    seasonality_cols = [
        'Monthly_CV', 'Seasonal_Strength', 'Christmas_Ratio', 
        'Summer_Ratio', 'School_Ratio', 'IQR_Strength'
    ]
    
    # Key value metrics  
    value_cols = [
        'Total_Value', 'Total_Volume', 'Revenue_Per_Unit', 
        'Revenue_Per_Week', 'Volume_Per_Week'
    ]
    
    correlations = {}
    
    print("\n" + "="*80)
    print("📊 CORRELATION ANALYSIS: SEASONALITY vs VALUE METRICS")
    print("="*80)
    
    for season_metric in seasonality_cols:
        correlations[season_metric] = {}
        print(f"\n📈 {season_metric}:")
        
        for value_metric in value_cols:
            # Remove infinite values for correlation calculation
            mask = np.isfinite(df[season_metric]) & np.isfinite(df[value_metric])
            if mask.sum() < 10:  # Need at least 10 valid points
                continue
                
            pearson_r, pearson_p = pearsonr(df[mask][season_metric], df[mask][value_metric])
            spearman_r, spearman_p = spearmanr(df[mask][season_metric], df[mask][value_metric])
            
            correlations[season_metric][value_metric] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p, 
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': mask.sum()
            }
            
            # Print significant correlations
            if pearson_p < 0.05:
                significance = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*"
                print(f"   • {value_metric}: r = {pearson_r:.3f} {significance}")
    
    return correlations

def identify_top_valuable_books(df, top_n=10):
    """Identify the top N most valuable books by lifetime value with their seasonality metrics."""
    
    # Get top books by total value
    top_valuable_books = df.nlargest(top_n, 'Total_Value')
    
    print(f"\n💰 TOP {top_n} MOST VALUABLE BOOKS (By Lifetime Value):")
    print("="*100)
    
    for idx, book in top_valuable_books.iterrows():
        rank = list(top_valuable_books.index).index(idx) + 1
        print(f"\n{rank:2d}. 📚 {book['Title'][:60]}")
        print(f"    Author: {book['Author']}")
        print(f"    💰 Lifetime Value: £{book['Total_Value']:,.0f}")
        print(f"    📦 Total Volume: {book['Total_Volume']:,} units")
        print(f"    💵 Revenue per Unit: £{book['Revenue_Per_Unit']:.2f}")
        print(f"    🔥 Seasonal Strength: {book['Seasonal_Strength']:.1f}x")
        print(f"    📊 Monthly CV: {book['Monthly_CV']:.2f}")
        print(f"    🎄 Christmas Ratio: {book['Christmas_Ratio']:.1f}x")
        print(f"    ☀️ Summer Ratio: {book['Summer_Ratio']:.1f}x") 
        print(f"    🏫 School Ratio: {book['School_Ratio']:.1f}x")
        print(f"    📅 Peak Month: {book['Peak_Month']}")
        print(f"    📖 Category: {book['Product_Class'][:70]}")
        print(f"    📈 Data Points: {book['Data_Points']} weeks")
    
    return top_valuable_books

def identify_sweet_spot_books(df, top_n=20):
    """Identify books that are both highly seasonal and highly valuable."""
    
    # Normalize rankings (0-1 scale)
    df['Value_Score'] = 1 - (df['Value_Rank'] - 1) / (len(df) - 1)
    df['Seasonality_Score'] = 1 - (df['Seasonality_Rank'] - 1) / (len(df) - 1)
    
    # Combined score (equal weighting)
    df['Sweet_Spot_Score'] = (df['Value_Score'] + df['Seasonality_Score']) / 2
    
    # Get top performers
    sweet_spot_books = df.nlargest(top_n, 'Sweet_Spot_Score')
    
    print(f"\n🎯 TOP {top_n} SWEET SPOT BOOKS (High Seasonality + High Value):")
    print("="*80)
    
    for idx, book in sweet_spot_books.iterrows():
        print(f"\n{len(sweet_spot_books) - list(sweet_spot_books.index).index(idx):2d}. 📚 {book['Title'][:50]}")
        print(f"    Author: {book['Author']}")
        print(f"    💰 Total Value: £{book['Total_Value']:,.0f} (Rank #{book['Value_Rank']:.0f})")
        print(f"    🔥 Seasonal Strength: {book['Seasonal_Strength']:.1f}x (Rank #{book['Seasonality_Rank']:.0f})")
        print(f"    🎄 Christmas Ratio: {book['Christmas_Ratio']:.1f}x")
        print(f"    📊 Sweet Spot Score: {book['Sweet_Spot_Score']:.3f}")
        print(f"    📖 Category: {book['Product_Class'][:60]}")
    
    return sweet_spot_books

def create_value_seasonality_visualizations(df):
    """Create comprehensive visualizations comparing value and seasonality."""
    
    figures = []
    
    # 1. Value vs Seasonality Scatter Plot
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=df['Total_Value'],
        y=df['Seasonal_Strength'],
        mode='markers',
        text=df['Title'].str[:40] + '...',
        hovertemplate='<b>%{text}</b><br>' +
                     'Total Value: £%{x:,.0f}<br>' +
                     'Seasonal Strength: %{y:.1f}x<br>' +
                     'Christmas Ratio: %{customdata:.1f}x<br>' +
                     'Product Class: %{meta}<extra></extra>',
        customdata=df['Christmas_Ratio'],
        meta=df['Product_Class'],
        marker=dict(
            size=np.log(df['Total_Volume'] + 1) * 2,
            color=df['Monthly_CV'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Monthly Coefficient<br>of Variation"),
            opacity=0.7
        ),
        name='Books'
    ))
    
    fig1.update_layout(
        title="Book Value vs Seasonality Analysis",
        xaxis_title="Total Revenue Value (£)",
        yaxis_title="Seasonal Strength (Peak/Min Ratio)", 
        xaxis_type="log",
        yaxis_type="log",
        height=600
    )
    
    figures.append(('value_vs_seasonality_scatter', fig1))
    
    # 2. Sweet Spot Analysis Dashboard
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Value Rank vs Seasonality Rank',
            'Revenue per Unit vs Seasonal Strength',
            'Top Product Classes by Combined Score',
            'Christmas Seasonality vs Revenue'
        ]
    )
    
    # Value rank vs seasonality rank
    fig2.add_trace(
        go.Scatter(
            x=df['Value_Rank'],
            y=df['Seasonality_Rank'],
            mode='markers',
            text=df['Title'].str[:30] + '...',
            hovertemplate='<b>%{text}</b><br>' +
                         'Value Rank: #%{x:.0f}<br>' +
                         'Seasonality Rank: #%{y:.0f}<br>' +
                         'Sweet Spot Score: %{customdata:.3f}<extra></extra>',
            customdata=df['Sweet_Spot_Score'],
            marker=dict(
                size=8,
                color=df['Sweet_Spot_Score'],
                colorscale='RdYlGn',
                showscale=False
            ),
            name='Rankings'
        ),
        row=1, col=1
    )
    
    # Revenue per unit vs seasonality
    fig2.add_trace(
        go.Scatter(
            x=df['Revenue_Per_Unit'],
            y=df['Seasonal_Strength'],
            mode='markers',
            text=df['Title'].str[:30] + '...',
            marker=dict(size=6, color='blue', opacity=0.6),
            name='Revenue vs Seasonality'
        ),
        row=1, col=2
    )
    
    # Top product classes analysis
    class_analysis = df.groupby('Product_Class').agg({
        'Sweet_Spot_Score': 'mean',
        'Total_Value': 'sum',
        'Seasonal_Strength': 'mean'
    }).sort_values('Sweet_Spot_Score', ascending=True).tail(15)
    
    fig2.add_trace(
        go.Bar(
            x=class_analysis['Sweet_Spot_Score'],
            y=[cls[:40] + '...' if len(cls) > 40 else cls for cls in class_analysis.index],
            orientation='h',
            marker_color='orange',
            name='Class Sweet Spot Score'
        ),
        row=2, col=1
    )
    
    # Christmas seasonality vs revenue
    fig2.add_trace(
        go.Scatter(
            x=df['Christmas_Ratio'],
            y=df['Total_Value'],
            mode='markers',
            text=df['Title'].str[:30] + '...',
            marker=dict(
                size=6,
                color=df['Peak_Month'],
                colorscale='RdYlBu_r',
                showscale=False
            ),
            name='Christmas vs Value'
        ),
        row=2, col=2
    )
    
    fig2.update_layout(
        title_text="Value-Seasonality Sweet Spot Analysis",
        height=900,
        showlegend=False
    )
    
    # Update axis labels
    fig2.update_xaxes(title_text="Value Rank (Lower = More Valuable)", row=1, col=1)
    fig2.update_yaxes(title_text="Seasonality Rank (Lower = More Seasonal)", row=1, col=1)
    fig2.update_xaxes(title_text="Revenue per Unit (£)", row=1, col=2)
    fig2.update_yaxes(title_text="Seasonal Strength", row=1, col=2)
    fig2.update_xaxes(title_text="Average Sweet Spot Score", row=2, col=1)
    fig2.update_xaxes(title_text="Christmas Ratio", row=2, col=2)
    fig2.update_yaxes(title_text="Total Revenue Value (£)", row=2, col=2)
    
    figures.append(('sweet_spot_dashboard', fig2))
    
    return figures

def generate_insights_report(df, correlations, sweet_spot_books):
    """Generate comprehensive insights report."""
    
    print(f"\n🎯 SEASONALITY-VALUE CORRELATION INSIGHTS")
    print("="*80)
    
    # Key statistics
    high_seasonal = df[df['Seasonal_Strength'] > 5.0]
    high_value = df[df['Total_Value'] > df['Total_Value'].quantile(0.8)]
    both_high = df[(df['Seasonal_Strength'] > 5.0) & (df['Total_Value'] > df['Total_Value'].quantile(0.8))]
    
    print(f"\n📊 KEY STATISTICS:")
    print(f"   • Books with high seasonality (>5x): {len(high_seasonal):,}")
    print(f"   • Books with high value (top 20%): {len(high_value):,}")  
    print(f"   • Books with BOTH high seasonality AND high value: {len(both_high):,}")
    print(f"   • Overlap percentage: {len(both_high)/len(high_seasonal)*100:.1f}% of seasonal books are also high-value")
    
    # Value comparison by seasonality
    print(f"\n💰 VALUE BY SEASONALITY LEVEL:")
    low_seasonal = df[df['Seasonal_Strength'] < 2.0]
    med_seasonal = df[(df['Seasonal_Strength'] >= 2.0) & (df['Seasonal_Strength'] < 5.0)]
    
    print(f"   • Low seasonality (<2x): Avg value £{low_seasonal['Total_Value'].mean():,.0f}")
    print(f"   • Medium seasonality (2-5x): Avg value £{med_seasonal['Total_Value'].mean():,.0f}")
    print(f"   • High seasonality (>5x): Avg value £{high_seasonal['Total_Value'].mean():,.0f}")
    
    # Christmas books analysis
    christmas_books = df[df['Christmas_Ratio'] > 2.0]
    print(f"\n🎄 CHRISTMAS SEASONALITY INSIGHTS:")
    print(f"   • Books with strong Christmas boost (>2x): {len(christmas_books):,}")
    print(f"   • Average value of Christmas books: £{christmas_books['Total_Value'].mean():,.0f}")
    print(f"   • Average value of non-Christmas books: £{df[df['Christmas_Ratio'] <= 2.0]['Total_Value'].mean():,.0f}")
    
    # Product class insights
    print(f"\n📚 TOP PRODUCT CLASSES (by combined value + seasonality):")
    class_analysis = df.groupby('Product_Class').agg({
        'Sweet_Spot_Score': 'mean',
        'Total_Value': ['sum', 'mean'],
        'Seasonal_Strength': 'mean',
        'ISBN': 'count'
    }).round(2)
    
    class_analysis.columns = ['Avg_Sweet_Spot', 'Total_Category_Value', 'Avg_Book_Value', 'Avg_Seasonality', 'Book_Count']
    class_analysis = class_analysis.sort_values('Avg_Sweet_Spot', ascending=False)
    
    for idx, (category, stats) in enumerate(class_analysis.head(10).iterrows()):
        print(f"   {idx+1:2d}. {category[:50]}...")
        print(f"       Sweet Spot Score: {stats['Avg_Sweet_Spot']:.3f} | Books: {stats['Book_Count']}")
        print(f"       Avg Value: £{stats['Avg_Book_Value']:,.0f} | Avg Seasonality: {stats['Avg_Seasonality']:.1f}x")

def save_results(df, correlations, figures, sweet_spot_books, top_valuable_books):
    """Save all analysis results."""
    
    # Create output directories
    output_dir = "outputs/value_seasonality_analysis"
    plots_dir = "outputs/plots/interactive"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save enhanced dataset
    df.to_csv(f"{output_dir}/value_seasonality_analysis.csv", index=False)
    print(f"Saved enhanced analysis to {output_dir}/value_seasonality_analysis.csv")
    
    # Save top valuable books
    top_valuable_books.to_csv(f"{output_dir}/top_10_valuable_books.csv", index=False)
    print(f"Saved top 10 valuable books to {output_dir}/top_10_valuable_books.csv")
    
    # Save sweet spot books
    sweet_spot_books.to_csv(f"{output_dir}/sweet_spot_books.csv", index=False)
    print(f"Saved sweet spot books to {output_dir}/sweet_spot_books.csv")
    
    # Save correlation matrix
    corr_data = []
    for season_metric, value_correlations in correlations.items():
        for value_metric, corr_stats in value_correlations.items():
            corr_data.append({
                'Seasonality_Metric': season_metric,
                'Value_Metric': value_metric,
                'Pearson_r': corr_stats['pearson_r'],
                'Pearson_p': corr_stats['pearson_p'],
                'Spearman_r': corr_stats['spearman_r'],
                'Spearman_p': corr_stats['spearman_p'],
                'Sample_Size': corr_stats['n_samples']
            })
    
    corr_df = pd.DataFrame(corr_data)
    corr_df.to_csv(f"{output_dir}/correlation_matrix.csv", index=False)
    print(f"Saved correlations to {output_dir}/correlation_matrix.csv")
    
    # Save visualizations
    for idx, (name, fig) in enumerate(figures):
        filename = f"{plots_dir}/value_seasonality_{name}.html"
        save_plot(fig, filename)
        print(f"Saved visualization to {filename}")

def main():
    """Run the complete value-seasonality analysis."""
    
    print("\n💰 VALUE-SEASONALITY CORRELATION ANALYSIS")
    print("="*80)
    print("Analyzing correlation between book seasonality and revenue value...")
    
    # Load data
    df = load_seasonality_data()
    
    # Calculate value metrics
    df = calculate_value_metrics(df)
    
    # Analyze correlations
    correlations = analyze_correlations(df)
    
    # Identify top valuable books
    top_valuable_books = identify_top_valuable_books(df, top_n=10)
    
    # Identify sweet spot books
    sweet_spot_books = identify_sweet_spot_books(df, top_n=25)
    
    # Create visualizations
    figures = create_value_seasonality_visualizations(df)
    
    # Generate insights report
    generate_insights_report(df, correlations, sweet_spot_books)
    
    # Save results
    save_results(df, correlations, figures, sweet_spot_books, top_valuable_books)
    
    print(f"\n✅ VALUE-SEASONALITY ANALYSIS COMPLETE!")
    print(f"📊 Files saved to: outputs/value_seasonality_analysis/")
    print(f"📈 Visualizations saved to: outputs/plots/interactive/")
    
    return df, correlations, sweet_spot_books, figures, top_valuable_books

if __name__ == "__main__":
    df, correlations, sweet_spot_books, figures, top_valuable_books = main()