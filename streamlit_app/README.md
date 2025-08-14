# Book Sales Forecasting - Streamlit App

A self-contained Streamlit application for book sales forecasting using Vertex AI endpoints.

## 🚀 Quick Start

### Development Mode (With Mock Predictions)
```bash
# From the streamlit_app directory
cd /Users/zarif/Documents/Projects/book_sales_deployment/streamlit_app

# Run the app
streamlit run app.py
```

### Production Mode (With Real Endpoints)
1. First deploy your models:
```bash
# From project root
python deploy_models.py --deploy-all
```

2. Then run the Streamlit app:
```bash
cd streamlit_app
streamlit run app.py
```

## 📁 App Structure

```
streamlit_app/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies  
├── streamlit_utils/
│   ├── endpoint_client.py     # Vertex AI endpoint client
│   ├── simple_plots.py        # Plotly visualization functions
│   └── data_helpers.py        # Data loading helpers
└── README.md                  # This file
```

## 🎯 Features

### 🏠 Home Page
- Welcome and quick start guide
- Available books overview
- System status check

### 🔮 Forecast Page  
- Book selection dropdown
- Date picker for target prediction date
- Real-time prediction generation
- Results visualization
- Prediction history

### 📊 Historical Data Page
- Historical sales trends
- Interactive Plotly charts
- Data insights and metrics
- Raw data viewing and download

### ⚙️ System Status Page
- Vertex AI connection status
- Endpoint deployment status
- Data availability check
- System recommendations

## 🔧 Configuration

The app automatically detects:
- Available books from data files
- Vertex AI endpoint availability
- Mock vs real prediction mode

### Book Configuration
Books are configured in `endpoint_client.py`:
```python
self.book_metadata = {
    "9780722532935": {
        "title": "The Alchemist",
        "author": "Paulo Coelho",
        "endpoint_name": "book-sales-9780722532935"
    },
    "9780241003008": {
        "title": "The Very Hungry Caterpillar", 
        "author": "Eric Carle",
        "endpoint_name": "book-sales-9780241003008"
    }
}
```

## 🎨 Design Philosophy

This app follows a **simple, self-contained** approach:

### ✅ Self-Contained Benefits
- **Refactor-friendly**: Won't break during pipeline restructuring
- **Independent**: Can run without main codebase dependencies
- **Mock-ready**: Works with sample data during development
- **Plotly-first**: Uses modern, interactive visualizations

### 🔄 Development Workflow
1. **Development**: Use mock predictions while models train
2. **Testing**: Deploy endpoints and test real predictions  
3. **Production**: Full integration with deployed models
4. **Enhancement**: Add features after pipeline refactor

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in the right directory
cd streamlit_app
python -c "import streamlit_utils.endpoint_client"
```

**No Historical Data:**
- App will create sample data automatically
- Check that data files exist in `../data/processed/`

**Vertex AI Connection Issues:**
- App falls back to mock predictions automatically
- Check authentication: `gcloud auth list`
- Verify project access: `gcloud config get-value project`

**Endpoint Not Found:**
- Deploy models first: `python deploy_models.py --deploy-all`
- Check System Status page for endpoint status

## 🚀 Next Steps

### Immediate Enhancements
- [ ] Add confidence intervals to predictions
- [ ] Multi-week forecast horizon slider
- [ ] Export prediction reports
- [ ] Model performance comparison

### Post-Refactor Enhancements  
- [ ] Integration with refactored pipeline
- [ ] Advanced seasonality analysis
- [ ] A/B testing capabilities
- [ ] Real-time model monitoring

## 📊 Usage Examples

### Basic Prediction
1. Go to **🔮 Forecast** page
2. Select "The Alchemist - Paulo Coelho" 
3. Pick a date 30 days in the future
4. Click "🔮 Get Sales Prediction"

### Historical Analysis
1. Go to **📊 Historical Data** page
2. Select specific book or "All Books"
3. View interactive trend charts
4. Check "Show Data Insights" for metrics

### System Health Check
1. Go to **⚙️ System Status** page
2. Check Vertex AI connection
3. Verify endpoint deployment status
4. Review data availability

---

*This app is designed to be simple and refactor-friendly, providing immediate value while supporting future enhancements.*