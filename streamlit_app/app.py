"""
Book Sales Forecasting - Streamlit App

A simple, self-contained Streamlit application for book sales forecasting
using deployed Vertex AI endpoints. Designed to be refactor-friendly.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import sys
import os
from typing import Dict, List, Optional

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'streamlit_utils'))

try:
    from streamlit_utils.endpoint_client import VertexAIEndpointClient, create_client
    from streamlit_utils.simple_plots import (
        plot_historical_sales, 
        plot_forecast_with_historical, 
        plot_single_prediction,
        format_prediction_text
    )
    from streamlit_utils.data_helpers import DataLoader, format_data_for_display, get_data_insights
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure you're running this from the streamlit_app directory")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Book Sales Forecasting",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = create_client()
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []


def main():
    """Main application function."""
    st.title("📚 Book Sales Forecasting")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "🔮 Forecast", "📊 Historical Data", "⚙️ System Status"]
        )
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This app provides sales forecasting for book titles using 
        machine learning models deployed on Google Cloud Vertex AI.
        
        **Features:**
        - Real-time predictions
        - Historical data visualization  
        - Model performance metrics
        """)
    
    # Route to selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Forecast":
        show_forecast_page()
    elif page == "📊 Historical Data":
        show_historical_page()
    elif page == "⚙️ System Status":
        show_system_status()


def show_home_page():
    """Display the home page."""
    st.header("🏠 Welcome to Book Sales Forecasting")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Quick Start
        
        1. **📖 Select a Book**: Choose from our available book titles
        2. **📅 Pick a Date**: Select a future date for your prediction
        3. **🔮 Get Forecast**: Click to get AI-powered sales prediction
        4. **📊 View Results**: See prediction with historical context
        
        ### 📚 Available Books
        """)
        
        # Show available books
        try:
            available_books = st.session_state.client.get_available_books()
            for isbn, book_info in available_books.items():
                st.markdown(f"""
                **{book_info['title']}** by *{book_info['author']}*  
                ISBN: `{isbn}`
                """)
        except Exception as e:
            st.error(f"Error loading books: {e}")
    
    with col2:
        st.info("""
        **💡 Tips**
        
        • Start with the Forecast page
        • Check Historical Data to understand trends
        • View System Status to check endpoint health
        • Predictions work for dates up to 1 year ahead
        """)
        
        # Quick system check
        with st.expander("🔍 Quick System Check"):
            with st.spinner("Checking system..."):
                try:
                    health = st.session_state.client.health_check()
                    if health.get('vertex_ai_available', False):
                        st.success("✅ Vertex AI connected")
                    else:
                        st.warning("⚠️ Using mock predictions")
                    
                    data_summary = st.session_state.data_loader.get_data_summary()
                    st.info(f"📊 {data_summary.get('total_books', 0)} books available")
                    
                except Exception as e:
                    st.error(f"System check failed: {e}")


def show_forecast_page():
    """Display the forecast page."""
    st.header("🔮 Sales Forecasting")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Prediction Settings")
        
        # Book selection
        try:
            available_books = st.session_state.client.get_available_books()
            book_options = {
                f"{info['title']} - {info['author']}": isbn 
                for isbn, info in available_books.items()
            }
            
            selected_book_display = st.selectbox(
                "📖 Select a Book:",
                options=list(book_options.keys()),
                help="Choose the book you want to forecast sales for"
            )
            
            selected_isbn = book_options[selected_book_display]
            selected_book_info = available_books[selected_isbn]
            
        except Exception as e:
            st.error(f"Error loading books: {e}")
            return
        
        # Date selection
        st.markdown("---")
        
        # Get latest date from historical data for reference
        try:
            latest_date = st.session_state.data_loader.get_latest_date(selected_isbn)
            if latest_date:
                min_date = latest_date.date() + timedelta(days=1)
                st.info(f"📅 Latest historical data: {latest_date.strftime('%Y-%m-%d')}")
            else:
                min_date = date.today()
                st.info("📅 Using today as reference date")
        except Exception:
            min_date = date.today()
            st.info("📅 Using today as reference date")
        
        max_date = date.today() + timedelta(days=365)  # 1 year ahead
        
        # Choice between single date and date range
        prediction_type = st.radio(
            "📊 Prediction Type:",
            ["Single Date", "Date Range"],
            help="Choose whether to predict for a single date or a range of dates"
        )
        
        if prediction_type == "Single Date":
            target_date = st.date_input(
                "🗓️ Target Prediction Date:",
                value=min_date + timedelta(days=30),  # Default to 1 month ahead
                min_value=min_date,
                max_value=max_date,
                help="Select a future date to predict sales for"
            )
            end_date = None
        else:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                target_date = st.date_input(
                    "🗓️ Start Date:",
                    value=min_date + timedelta(days=30),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the start date for predictions"
                )
            with col_date2:
                end_date = st.date_input(
                    "🏁 End Date:",
                    value=min_date + timedelta(days=60),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the end date for predictions"
                )
            
            # Validate date range
            if end_date and target_date and end_date <= target_date:
                st.error("End date must be after start date")
                end_date = None
        
        # Prediction button
        st.markdown("---")
        
        predict_button = st.button(
            "🔮 Get Sales Prediction",
            type="primary",
            help="Click to generate AI-powered sales forecast"
        )
        
    with col2:
        st.subheader("📖 Selected Book")
        st.markdown(f"""
        **Title:** {selected_book_info['title']}  
        **Author:** {selected_book_info['author']}  
        **ISBN:** `{selected_isbn}`  
        """)
        
        # Show prediction details
        if target_date:
            target_datetime = datetime.combine(target_date, datetime.min.time())
            
            try:
                latest_date = st.session_state.data_loader.get_latest_date(selected_isbn)
                if latest_date:
                    weeks_ahead = max(1, int((target_datetime - latest_date).days / 7))
                else:
                    weeks_ahead = max(1, int((target_datetime - datetime.now()).days / 7))
                
                if prediction_type == "Single Date":
                    st.info(f"""
                    **Prediction Details:**
                    - Target Date: {target_date}
                    - Weeks Ahead: {weeks_ahead}
                    - Forecast Horizon: ~{weeks_ahead/4:.1f} months
                    """)
                else:
                    if end_date:
                        end_datetime = datetime.combine(end_date, datetime.min.time())
                        if latest_date:
                            weeks_end = max(1, int((end_datetime - latest_date).days / 7))
                        else:
                            weeks_end = max(1, int((end_datetime - datetime.now()).days / 7))
                        
                        days_span = (end_date - target_date).days
                        st.info(f"""
                        **Date Range Prediction Details:**
                        - Start Date: {target_date} ({weeks_ahead} weeks ahead)
                        - End Date: {end_date} ({weeks_end} weeks ahead)  
                        - Date Span: {days_span} days (~{days_span/7:.1f} weeks)
                        """)
                
            except Exception as e:
                st.warning(f"Could not calculate forecast details: {e}")
    
    # Handle prediction
    if predict_button and target_date:
        # Validate date range if selected
        if prediction_type == "Date Range" and (not end_date or end_date <= target_date):
            st.error("Please select a valid date range (end date must be after start date)")
        else:
            with st.spinner("🔄 Generating prediction..."):
                try:
                    if prediction_type == "Single Date":
                        # Single date prediction
                        target_datetime = datetime.combine(target_date, datetime.min.time())
                        prediction = st.session_state.client.get_prediction(selected_isbn, target_datetime)
                        
                        # Store in history
                        st.session_state.predictions_history.append({
                            'timestamp': datetime.now(),
                            'prediction': prediction
                        })
                        
                        predictions = [prediction]  # Wrap in list for consistent handling
                    
                    else:
                        # Date range prediction
                        predictions = []
                        current_date = target_date
                        
                        # Generate predictions for weekly intervals
                        while current_date <= end_date:
                            target_datetime = datetime.combine(current_date, datetime.min.time())
                            prediction = st.session_state.client.get_prediction(selected_isbn, target_datetime)
                            prediction['date_in_range'] = current_date.isoformat()
                            predictions.append(prediction)
                            current_date += timedelta(weeks=1)  # Weekly predictions
                        
                        # Store range in history
                        st.session_state.predictions_history.append({
                            'timestamp': datetime.now(),
                            'prediction': {
                                'type': 'date_range',
                                'start_date': target_date.isoformat(),
                                'end_date': end_date.isoformat(),
                                'predictions_count': len(predictions),
                                'title': selected_book_info['title']
                            }
                        })
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Show prediction details
                        if prediction_type == "Single Date":
                            prediction = predictions[0]
                            if 'error' not in prediction:
                                st.success("✅ Prediction generated successfully!")
                                
                                # Format prediction text
                                prediction_text = format_prediction_text(prediction)
                                st.markdown(prediction_text)
                                
                                # Show prediction chart
                                fig = plot_single_prediction(prediction)
                                st.plotly_chart(fig, use_container_width=True)
                                
                            else:
                                st.error(f"❌ Prediction failed: {prediction['error']}")
                                if 'suggestion' in prediction:
                                    st.info(f"💡 {prediction['suggestion']}")
                        
                        else:
                            # Date range predictions
                            successful_predictions = [p for p in predictions if 'error' not in p]
                            
                            if successful_predictions:
                                st.success(f"✅ Generated {len(successful_predictions)} predictions!")
                                
                                # Create a summary table
                                prediction_df = pd.DataFrame([{
                                    'Date': p.get('date_in_range', p.get('target_date', 'Unknown')),
                                    'Predicted Sales': p.get('predicted_sales', 0),
                                    'Type': 'Mock' if p.get('prediction_type') == 'mock' else 'Real'
                                } for p in successful_predictions])
                                
                                # Display table
                                st.subheader("📊 Prediction Summary")
                                st.dataframe(prediction_df, use_container_width=True)
                                
                                # Show chart for range predictions
                                try:
                                    import plotly.express as px
                                    fig = px.line(prediction_df, x='Date', y='Predicted Sales', 
                                                title=f"Sales Forecast Range - {selected_book_info['title']}")
                                    fig.update_layout(showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                except:
                                    st.info("Chart visualization not available")
                                
                            else:
                                st.error("❌ All predictions in range failed")
                                failed_prediction = predictions[0] if predictions else {}
                                if 'suggestion' in failed_prediction:
                                    st.info(f"💡 {failed_prediction['suggestion']}")
                
                    with col2:
                        # Additional info
                        sample_prediction = predictions[0] if predictions else {}
                        if 'error' not in sample_prediction and sample_prediction:
                            pred_type = sample_prediction.get('prediction_type', 'unknown')
                            if pred_type == 'mock':
                                st.warning("""
                                ⚠️ **Development Mode**
                                
                                These are mock predictions for testing. 
                                Deploy your models to get real predictions.
                                """)
                            else:
                                st.success("""
                                ✨ **Live Prediction**
                                
                                These predictions come from your deployed 
                                machine learning model.
                                """)
                        
                        if prediction_type == "Date Range" and predictions:
                            st.info(f"""
                            **Range Summary:**
                            - Total Predictions: {len(predictions)}
                            - Date Range: {target_date} to {end_date}
                            - Frequency: Weekly
                            """)
                
                except Exception as e:
                    st.error(f"❌ Error generating prediction: {e}")
                    logger.error(f"Prediction error: {e}")
    
    # Show recent predictions history
    if st.session_state.predictions_history:
        st.markdown("---")
        st.subheader("📜 Recent Predictions")
        
        # Show last 5 predictions
        recent_predictions = st.session_state.predictions_history[-5:]
        for i, pred_record in enumerate(reversed(recent_predictions)):
            with st.expander(f"Prediction {len(recent_predictions)-i} - {pred_record['timestamp'].strftime('%H:%M:%S')}"):
                pred = pred_record['prediction']
                if 'error' not in pred:
                    st.write(f"**Book:** {pred.get('title', 'Unknown')}")
                    st.write(f"**Date:** {pred.get('target_date', 'Unknown')}")
                    st.write(f"**Prediction:** {pred.get('predicted_sales', 0):.1f} units")
                else:
                    st.write(f"**Error:** {pred['error']}")


def show_historical_page():
    """Display historical data analysis."""
    st.header("📊 Historical Sales Data")
    
    # Book selection for historical view
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            available_books = st.session_state.data_loader.get_available_books()
            book_options = {"All Books": None}
            book_options.update({
                f"{info['title']} - {info['author']}": isbn 
                for isbn, info in available_books.items()
            })
            
            selected_book_display = st.selectbox(
                "📖 Select Book for Analysis:",
                options=list(book_options.keys())
            )
            
            selected_isbn = book_options[selected_book_display]
            
            
        except Exception as e:
            st.error(f"Error loading books: {e}")
            return
    
    with col2:
        # Display options
        show_raw_data = st.checkbox("📋 Show Raw Data", value=False)
        show_insights = st.checkbox("💡 Show Data Insights", value=True)
    
    # Load and display data
    try:
        with st.spinner("📈 Loading historical data..."):
            historical_data = st.session_state.data_loader.load_historical_data(selected_isbn)
            
            
            if historical_data.empty:
                st.warning("📭 No historical data available")
                return
            
            # Plot historical data
            st.subheader(f"📈 Sales Trends")
            
            # Fix column names for plotting
            plot_data = historical_data.copy()
            if 'End_Date' in plot_data.columns:
                plot_data['End Date'] = plot_data['End_Date']
            
            
            fig = plot_historical_sales(
                plot_data, 
                isbn=selected_isbn,
                title=f"Historical Sales - {selected_book_display}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data insights
            if show_insights:
                st.subheader("💡 Data Insights")
                
                insights = get_data_insights(historical_data)
                
                if 'error' not in insights:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Avg Weekly Sales", 
                            f"{insights.get('avg_weekly_sales', 0):.1f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Data Span", 
                            f"{insights.get('data_span_weeks', 0)} weeks"
                        )
                    
                    with col3:
                        st.metric(
                            "Min Sales", 
                            f"{insights.get('min_weekly_sales', 0):.1f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Max Sales", 
                            f"{insights.get('max_weekly_sales', 0):.1f}"
                        )
            
            # Raw data display
            if show_raw_data:
                st.subheader("📋 Raw Data")
                
                display_data = format_data_for_display(historical_data)
                st.dataframe(display_data, use_container_width=True)
                
                # Download option
                csv = historical_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"sales_data_{selected_isbn or 'all_books'}.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"❌ Error loading historical data: {e}")
        logger.error(f"Historical data error: {e}")


def show_system_status():
    """Display system status and diagnostics."""
    st.header("⚙️ System Status")
    
    # Endpoint client status
    st.subheader("🔗 Vertex AI Connection")
    
    with st.spinner("Checking Vertex AI status..."):
        try:
            health = st.session_state.client.health_check()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if health.get('vertex_ai_available', False):
                    st.success("✅ Vertex AI Connected")
                else:
                    st.warning("⚠️ Vertex AI Not Available (Using Mock Mode)")
                
                st.info(f"**Project ID:** {health.get('project_id', 'Unknown')}")
                st.info(f"**Region:** {health.get('region', 'Unknown')}")
            
            with col2:
                st.info(f"**Available Books:** {health.get('available_books', 0)}")
                st.info(f"**Cached Endpoints:** {health.get('endpoints_cached', 0)}")
                
                if 'total_endpoints' in health:
                    st.info(f"**Total Endpoints:** {health['total_endpoints']}")
        
        except Exception as e:
            st.error(f"❌ Error checking Vertex AI status: {e}")
    
    # Endpoint deployment status
    if 'book_endpoints_found' in health:
        st.subheader("🎯 Model Endpoints")
        
        for endpoint_info in health['book_endpoints_found']:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**{endpoint_info['title']}**")
            
            with col2:
                st.write(f"`{endpoint_info['endpoint']}`")
            
            with col3:
                if endpoint_info['deployed']:
                    st.success("✅ Deployed")
                else:
                    st.error("❌ Not Found")
    
    # Data status
    st.subheader("📊 Data Status")
    
    try:
        data_summary = st.session_state.data_loader.get_data_summary()
        
        st.info(f"**Total Books:** {data_summary.get('total_books', 0)}")
        st.info(f"**Total Records:** {data_summary.get('total_records', 0)}")
        
        if data_summary.get('available_books'):
            st.subheader("📚 Book Data Details")
            
            for isbn, book_info in data_summary['available_books'].items():
                with st.expander(f"{book_info['title']} ({isbn})"):
                    st.write(f"**Records:** {book_info['records']}")
                    st.write(f"**Date Range:** {book_info['date_start']} to {book_info['date_end']}")
                    
    except Exception as e:
        st.error(f"❌ Error checking data status: {e}")
    
    # System recommendations
    st.subheader("💡 System Recommendations")
    
    recommendations = []
    
    if not health.get('vertex_ai_available', False):
        recommendations.append("🔧 Deploy models using: `python deploy_models.py --deploy-all`")
    
    if health.get('total_endpoints', 0) == 0:
        recommendations.append("📡 No endpoints found - check your deployment")
    
    if data_summary.get('total_records', 0) == 0:
        recommendations.append("📊 No historical data found - check data files")
    
    if not recommendations:
        st.success("✅ System looks healthy!")
    else:
        for rec in recommendations:
            st.warning(rec)


# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"App error: {e}")
        
        with st.expander("🐛 Debug Information"):
            st.code(str(e))
            st.info("""
            **Troubleshooting Tips:**
            1. Make sure you're running from the streamlit_app directory
            2. Check that all dependencies are installed
            3. Verify your data files exist in ../data/processed/
            4. Check the logs for more detailed error information
            """)