"""
Highlight specific user books in seasonality analysis and show their rankings.

This script finds user-specified books and highlights them in the seasonality visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plotting import save_plot

def load_and_analyze_user_books(user_isbns):
    """Load seasonality data and analyze user's specific books."""
    
    # Load full seasonality data
    df = pd.read_csv("outputs/seasonality_analysis/full_seasonality_analysis.csv", dtype={'ISBN': str})
    
    # Find user books
    user_books = df[df['ISBN'].isin(user_isbns)].copy()
    
    print(f"Found {len(user_books)} user books in dataset of {len(df)} total books")
    
    if len(user_books) == 0:
        print("❌ No user books found in the seasonality analysis!")
        print("Available ISBNs:")
        for isbn in df['ISBN'].head(10):
            print(f"  {isbn}")
        return None, None
    
    # Calculate rankings
    df['Seasonality_Rank'] = df['Seasonal_Strength'].rank(ascending=False)
    df['Value_Rank'] = df['Total_Value'].rank(ascending=False)
    df['Volume_Rank'] = df['Total_Volume'].rank(ascending=False)
    df['Christmas_Rank'] = df['Christmas_Ratio'].rank(ascending=False)
    
    # Get user book rankings
    user_books_with_ranks = df[df['ISBN'].isin(user_isbns)].copy()
    
    return df, user_books_with_ranks

def print_user_book_analysis(user_books, total_books):
    """Print detailed analysis of user books."""
    
    print("\n" + "="*80)
    print("📚 YOUR BOOKS SEASONALITY ANALYSIS")
    print("="*80)
    
    for _, book in user_books.iterrows():
        print(f"\n📖 {book['Title']}")
        print(f"    Author: {book['Author']}")
        print(f"    ISBN: {book['ISBN']}")
        print(f"    Category: {book['Product_Class']}")
        print(f"    Publisher: {book['Publisher_Group']}")
        
        print(f"\n🏆 RANKINGS (out of {total_books} books):")
        print(f"    🔥 Seasonality: #{int(book['Seasonality_Rank'])} (Seasonal Strength: {book['Seasonal_Strength']:.1f}x)")
        print(f"    💰 Value: #{int(book['Value_Rank'])} (Total Value: £{book['Total_Value']:,.0f})")
        print(f"    📊 Volume: #{int(book['Volume_Rank'])} (Total Volume: {book['Total_Volume']:,.0f})")
        print(f"    🎄 Christmas: #{int(book['Christmas_Rank'])} (Christmas Ratio: {book['Christmas_Ratio']:.1f}x)")
        
        print(f"\n📈 SEASONALITY METRICS:")
        print(f"    📊 Monthly Coefficient of Variation: {book['Monthly_CV']:.3f}")
        print(f"    🎄 Christmas Boost: {book['Christmas_Ratio']:.1f}x")
        print(f"    🏫 Back-to-School Boost: {book['School_Ratio']:.1f}x")
        print(f"    ☀️ Summer Pattern: {book['Summer_Ratio']:.1f}x")
        print(f"    📅 Peak Month: {int(book['Peak_Month'])}")
        print(f"    📅 Minimum Month: {int(book['Min_Month'])}")
        print(f"    ⏱️ Data Span: {book['Years_of_Data']:.1f} years")
        
        # Monthly breakdown
        monthly_volumes = eval(book['Monthly_Volumes'])
        print(f"\n📅 MONTHLY SALES PATTERN:")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, month in enumerate(months, 1):
            volume = monthly_volumes.get(i, 0)
            print(f"    {month}: {volume:,} units")
        
        # Classification
        classifications = []
        if book['Seasonal_Strength'] > 3.0 and book['Monthly_CV'] > 0.4:
            classifications.append("Excellent SARIMA candidate")
        elif book['Seasonal_Strength'] > 2.0 and book['Monthly_CV'] > 0.3:
            classifications.append("Good SARIMA candidate")
        if book['Christmas_Ratio'] > 2.0:
            classifications.append("Christmas seasonal")
        if book['Total_Volume'] > 1000000:
            classifications.append("High volume")
        
        print(f"\n🏷️ CLASSIFICATIONS: {', '.join(classifications) if classifications else 'Low seasonality'}")
        print("-" * 80)

def create_highlighted_visualizations(df, user_books):
    """Create visualizations with user books highlighted."""
    
    figures = []
    
    # 1. Value vs Seasonality Scatter with highlighted user books
    fig1 = go.Figure()
    
    # Add all other books as background
    other_books = df[~df['ISBN'].isin(user_books['ISBN'])]
    fig1.add_trace(go.Scatter(
        x=other_books['Total_Value'],
        y=other_books['Seasonal_Strength'],
        mode='markers',
        text=other_books['Title'].str[:40] + '...',
        hovertemplate='<b>%{text}</b><br>' +
                     'Total Value: £%{x:,.0f}<br>' +
                     'Seasonal Strength: %{y:.1f}x<br>' +
                     'Seasonality Rank: #%{customdata:.0f}<extra></extra>',
        customdata=other_books['Seasonality_Rank'],
        marker=dict(
            size=6,
            color='lightgray',
            opacity=0.4
        ),
        name='Other Books'
    ))
    
    # Highlight user books
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (_, book) in enumerate(user_books.iterrows()):
        fig1.add_trace(go.Scatter(
            x=[book['Total_Value']],
            y=[book['Seasonal_Strength']], 
            mode='markers+text',
            text=[book['Title'][:20] + '...'],
            textposition='top center',
            hovertemplate=f"<b>{book['Title']}</b><br>" +
                         f"Total Value: £{book['Total_Value']:,.0f}<br>" +
                         f"Seasonal Strength: {book['Seasonal_Strength']:.1f}x<br>" +
                         f"Seasonality Rank: #{int(book['Seasonality_Rank'])}<br>" +
                         f"Value Rank: #{int(book['Value_Rank'])}<extra></extra>",
            marker=dict(
                size=15,
                color=colors[idx % len(colors)],
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name=f"{book['Title'][:20]}..."
        ))
    
    fig1.update_layout(
        title="Your Books Highlighted: Value vs Seasonality Analysis",
        xaxis_title="Total Revenue Value (£)",
        yaxis_title="Seasonal Strength (Peak/Min Ratio)",
        xaxis_type="log", 
        yaxis_type="log",
        height=700,
        font=dict(size=12)
    )
    
    figures.append(('user_books_value_seasonality', fig1))
    
    # 2. Ranking comparison dashboard
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Seasonality Ranking Position',
            'Value Ranking Position', 
            'Monthly Sales Patterns',
            'Christmas vs Summer Seasonality'
        ]
    )
    
    # Seasonality ranking bar chart
    fig2.add_trace(
        go.Bar(
            x=[book['Title'][:15] + '...' for _, book in user_books.iterrows()],
            y=[int(book['Seasonality_Rank']) for _, book in user_books.iterrows()],
            text=[f"#{int(book['Seasonality_Rank'])}" for _, book in user_books.iterrows()],
            textposition='outside',
            marker=dict(color=['red', 'blue'][:len(user_books)]),
            name='Seasonality Rank'
        ),
        row=1, col=1
    )
    
    # Value ranking bar chart
    fig2.add_trace(
        go.Bar(
            x=[book['Title'][:15] + '...' for _, book in user_books.iterrows()],
            y=[int(book['Value_Rank']) for _, book in user_books.iterrows()],
            text=[f"#{int(book['Value_Rank'])}" for _, book in user_books.iterrows()],
            textposition='outside',
            marker=dict(color=['red', 'blue'][:len(user_books)]),
            name='Value Rank'
        ),
        row=1, col=2
    )
    
    # Monthly patterns
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for idx, (_, book) in enumerate(user_books.iterrows()):
        monthly_volumes = eval(book['Monthly_Volumes'])
        monthly_data = [monthly_volumes.get(i, 0) for i in range(1, 13)]
        
        fig2.add_trace(
            go.Scatter(
                x=months,
                y=monthly_data,
                mode='lines+markers',
                name=book['Title'][:20] + '...',
                line=dict(color=colors[idx % len(colors)], width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
    
    # Christmas vs Summer comparison
    fig2.add_trace(
        go.Scatter(
            x=[book['Christmas_Ratio'] for _, book in user_books.iterrows()],
            y=[book['Summer_Ratio'] for _, book in user_books.iterrows()],
            mode='markers+text',
            text=[book['Title'][:15] + '...' for _, book in user_books.iterrows()],
            textposition='top center',
            marker=dict(
                size=15,
                color=[colors[i] for i in range(len(user_books))],
                symbol='star'
            ),
            name='Your Books'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig2.update_layout(
        title_text="Your Books: Detailed Seasonality Analysis",
        height=900,
        showlegend=True
    )
    
    # Update y-axis to be inverted for rankings (lower rank = better)
    fig2.update_yaxes(title_text="Rank Position (Lower = Better)", autorange="reversed", row=1, col=1)
    fig2.update_yaxes(title_text="Rank Position (Lower = Better)", autorange="reversed", row=1, col=2)
    fig2.update_yaxes(title_text="Monthly Volume", row=2, col=1)
    fig2.update_xaxes(title_text="Christmas Ratio", row=2, col=2)
    fig2.update_yaxes(title_text="Summer Ratio", row=2, col=2)
    
    figures.append(('user_books_detailed_analysis', fig2))
    
    return figures

def save_results(figures, user_books):
    """Save visualizations and user book analysis."""
    
    # Create output directories
    plots_dir = "outputs/plots/interactive"
    output_dir = "outputs/user_books_analysis"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save user books analysis
    user_books.to_csv(f"{output_dir}/user_books_seasonality.csv", index=False)
    print(f"Saved user books analysis to {output_dir}/user_books_seasonality.csv")
    
    # Save visualizations
    for idx, (name, fig) in enumerate(figures):
        filename = f"{plots_dir}/{name}.html"
        save_plot(fig, filename)
        print(f"Saved visualization to {filename}")

def main():
    """Main function to analyze and highlight user books."""
    
    # User's books ISBNs
    user_isbns = ['9780722532935', '9780241003008']
    user_book_names = ['The Alchemist', 'Very Hungry Caterpillar, The']
    
    print(f"\n🔍 ANALYZING YOUR BOOKS IN SEASONALITY RANKINGS")
    print("="*80)
    print(f"Searching for:")
    for isbn, name in zip(user_isbns, user_book_names):
        print(f"   📚 {name} ({isbn})")
    
    # Load data and find user books
    df, user_books = load_and_analyze_user_books(user_isbns)
    
    if user_books is None:
        return
    
    print(f"\n✅ Found {len(user_books)} of your books in the analysis!")
    
    # Print detailed analysis
    print_user_book_analysis(user_books, len(df))
    
    # Create highlighted visualizations
    figures = create_highlighted_visualizations(df, user_books)
    
    # Save results
    save_results(figures, user_books)
    
    print(f"\n🎯 SUMMARY FOR YOUR BOOKS:")
    print("="*40)
    for _, book in user_books.iterrows():
        percentile = (len(df) - book['Seasonality_Rank']) / len(df) * 100
        print(f"📚 {book['Title'][:30]}...")
        print(f"   🏆 Seasonality Rank: #{int(book['Seasonality_Rank'])} ({percentile:.1f}th percentile)")
        print(f"   💰 Value Rank: #{int(book['Value_Rank'])}")
        print(f"   🔥 Seasonal Strength: {book['Seasonal_Strength']:.1f}x")
        print(f"   🎄 Christmas Boost: {book['Christmas_Ratio']:.1f}x")
    
    print(f"\n📈 VISUALIZATIONS CREATED:")
    print(f"   • user_books_value_seasonality.html - Your books highlighted on main plot")
    print(f"   • user_books_detailed_analysis.html - Detailed dashboard for your books")
    
    return df, user_books, figures

if __name__ == "__main__":
    df, user_books, figures = main()