import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def plot_prediction(series_train, series_test, forecast, forecast_int=None, fitted_values=None, title="Forecast Plot"):
    """
    Plot ARIMA/SARIMA forecast with training data, test data, and confidence intervals.

    Parameters:
    - series_train: Training time series data
    - series_test: Test time series data
    - forecast: Forecast values
    - forecast_int: Confidence intervals (DataFrame with 'lower' and 'upper' columns)
    - fitted_values: Fitted values from the model
    - title: Plot title

    Returns:
    - fig: Plotly figure object
    - mae: Mean Absolute Error
    - mape: Mean Absolute Percentage Error
    """
    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)

    # Create a figure
    fig = go.Figure()

    # Add the training data trace
    fig.add_trace(go.Scatter(x=series_train.index, y=series_train,
                             mode='lines', name='Train / Actual',
                             line=dict(color='blue')))

    # Add the test data (actual values) trace
    fig.add_trace(go.Scatter(x=series_test.index, y=series_test,
                             mode='lines', name='Test / Actual',
                             line=dict(color='black')))

    # Add the forecast data trace
    fig.add_trace(go.Scatter(x=series_test.index, y=forecast,
                             mode='lines', name='Forecast',
                             line=dict(color='red')))

    # Add fitted values if provided
    if fitted_values is not None:
        fig.add_trace(go.Scatter(x=fitted_values.index, y=fitted_values,
                                 mode='lines', name='Fitted Values',
                                 line=dict(color='green', dash='dash')))

    # If forecast intervals are available, add them as shaded areas
    if forecast_int is not None:
        fig.add_trace(go.Scatter(
            x=series_test.index.tolist() + series_test.index[::-1].tolist(),
            y=forecast_int['upper'].tolist() + forecast_int['lower'][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(169, 169, 169, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Confidence Interval',
            showlegend=False
        ))

    # Update layout for titles and labels
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(font=dict(size=16)),
        template='plotly_white',
        width=800,
        height=500
    )

    # Return the figure for saving
    return fig, mae, mape


def create_sarima_forecast(sarima_model, train_data, test_data, n_periods=32, title="SARIMA Forecast"):
    """
    Create SARIMA forecast with confidence intervals and fitted values.

    Parameters:
    - sarima_model: Fitted SARIMA model
    - train_data: Training data (Series or DataFrame with 'Volume' column)
    - test_data: Test data (Series or DataFrame with 'Volume' column)
    - n_periods: Number of periods to forecast
    - title: Plot title

    Returns:
    - fig: Plotly figure object
    - mae: Mean Absolute Error
    - mape: Mean Absolute Percentage Error
    - forecast_results: Dictionary with forecast data
    """

    # Get forecast
    forecast_sarima = sarima_model.get_forecast(steps=n_periods)

    # Extract predicted mean and confidence intervals
    predicted_mean_sarima = forecast_sarima.predicted_mean
    confidence_intervals_sarima = forecast_sarima.conf_int(alpha=0.05)  # 95% CI

    # Convert confidence intervals to DataFrame
    forecast_int_df_sarima = pd.DataFrame({
        "lower": confidence_intervals_sarima.iloc[:, 0],
        "upper": confidence_intervals_sarima.iloc[:, 1]
    }, index=predicted_mean_sarima.index)

    # Extract the fitted values
    fitted_values_sarima = sarima_model.get_prediction(start=train_data.index[0], end=train_data.index[-1])
    fitted_values_series_sarima = fitted_values_sarima.predicted_mean

    # Handle both Series and DataFrame inputs
    if isinstance(train_data, pd.DataFrame):
        train_series = train_data['Volume']
    else:
        train_series = train_data

    if isinstance(test_data, pd.DataFrame):
        test_series = test_data['Volume']
    else:
        test_series = test_data

    # Call the plotting function
    fig, mae, mape = plot_prediction(
        series_train=train_series,
        series_test=test_series,
        forecast=predicted_mean_sarima,
        forecast_int=forecast_int_df_sarima,
        fitted_values=fitted_values_series_sarima,
        title=title
    )

    # Print the metrics
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.4f}")

    # Prepare forecast results
    forecast_results = {
        'predicted_mean': predicted_mean_sarima,
        'confidence_intervals': forecast_int_df_sarima,
        'fitted_values': fitted_values_series_sarima,
        'mae': mae,
        'mape': mape
    }

    return fig, mae, mape, forecast_results


# Example usage (commented out - uncomment and modify as needed):

# Number of periods to forecast
n_periods = 32

# Create forecast and plot
fig, mae, mape, results = create_sarima_forecast(
    sarima_model=sarima_alchemist,
    train_data=train_data_alchemist,
    test_data=test_data_alchemist,
    n_periods=n_periods,
    title='SARIMA(1, 0, 0)x(1, 0, 0, 52) Alchemist forecast'
)

# Show the plot
fig.show()

# Save the plot if needed
# fig.write_html("sarima_forecast.html")
# fig.write_image("sarima_forecast.png")
