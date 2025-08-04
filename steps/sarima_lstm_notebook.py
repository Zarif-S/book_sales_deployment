train_data_alchemist.info()

# Define the SARIMA model.
sarima_model_alchemist = SARIMAX(endog= train_data_alchemist['Volume'],                      # Target series
                #exog=exo_ts,                   # Exogenous series
                order=(1, 0, 0),               # p, d, q orders
                seasonal_order=(1, 0, 0, 52),  # P, D, Q, and m
                trend='c')                     # Type of trend: Including a constant

# Fit the model.
sarima_alchemist = sarima_model_alchemist.fit(maxiter=500, disp=False)

# Print the statistical results of the model.
sarima_alchemist.summary()

# Number of periods to forecast
n_periods = 32

# Get forecast
forecast_SARIMA_alchemist = sarima_alchemist.get_forecast(steps=n_periods)

# Extract predicted mean and confidence intervals
predicted_mean_SARIMA_alchemist = forecast_SARIMA_alchemist.predicted_mean
confidence_intervals_SARIMA_alchemist = forecast_SARIMA_alchemist.conf_int(alpha=0.05)  # 95% CI

# Convert confidence intervals to DataFrame
forecast_int_df_SARIMA_alchemist = pd.DataFrame({
    "lower": confidence_intervals_SARIMA_alchemist.iloc[:, 0],
    "upper": confidence_intervals_SARIMA_alchemist.iloc[:, 1]
}, index=predicted_mean_SARIMA_alchemist.index)  # Use the index of the predicted mean

# Extract the fitted values
fitted_values_SARIMA_alchemist = sarima_alchemist.get_prediction(start=train_data_alchemist.index[0], end=train_data_alchemist.index[-1])
fitted_values_series_SARIMA_alchemist = fitted_values_SARIMA_alchemist.predicted_mean  # This will give you the fitted values as a Series

# Call the function with fitted values to get the figure
fig, mae, mape = plot_prediction(
    series_train=train_data_alchemist['Volume'],
    series_test=test_data_alchemist['Volume'],
    forecast=predicted_mean_SARIMA_alchemist,  # Use predicted mean for forecast
    forecast_int=forecast_int_df_SARIMA_alchemist,
    fitted_values=fitted_values_series_SARIMA_alchemist,
    title='SARIMA(1, 0, 0)x(1, 0, 0, 52) Alchemist forecast'
)

# Print the metrics
print("MAE:", mae)
print("MAPE:", mape)

# Show the plot
fig.show()

# Dynamically assign the next model number
model_number = len(model_results_list) + 1  # Will assign 7 if 6 models are already in the list

# Define the new model configurations for the ARIMA model for "The Alchemist"
model_name = f"SARIMAX(1, 0, 0)x(1, 0, 0, 52)"  # Dynamically name the model

model_results = {
    "Log Likelihood": -3835.230,
    "AIC": 7678.459,
    "BIC": 7696.197,
    "Sample": "01-07-2012 - 12-09-2023",
    "Intercept": 42.5576,
    "AR.L1": 0.8028,
    "AR.S.L52": 0.6030,
    "Sigma2": 1.252e+04,
    "Ljung-Box (L1)": 0.98,
    "Jarque-Bera (JB)": 1236.31,
    "Heteroskedasticity (H)": 2.94,
    "Skewness": 0.07,
    "Kurtosis": 9.90,
    "Time Taken": "44 seconds"
}

# Call the function to plot predictions and get metrics (assumed function)
fig, mae, mape = plot_prediction(
    series_train=train_data_alchemist,
    series_test=test_data_alchemist['Volume'],
    forecast=predicted_mean_SARIMA_alchemist,
    forecast_int=forecast_int_df_SARIMA_alchemist,
    fitted_values=fitted_values_series_SARIMA_alchemist,
    title='SARIMA(1, 0, 0)x(1, 0, 0, 52) Alchemist forecast'
)

# Log the first model results with Model Number, Book, and Dataset
log_model_results(model_number, model_name, model_results, mae, mape,"The Alchemist", "After 2012-01-01 weekly")

# Specify the file path where you want to save the data and model
save_path_alchemist_sarima = 'sarima_model_alchemist.pkl'

# Save everything in a dictionary
save_data_alchemist_sarima = {
    'model': sarima_alchemist,  # Save the trained ARIMA model
    'train_data_alchemist': train_data_alchemist,  # Training data
    'test_data_alchemist': test_data_alchemist,  # Test data
    'forecast': forecast_SARIMA_alchemist,  # Forecast data
    'forecast_int': forecast_int_df_SARIMA_alchemist,  # Forecast confidence intervals
    'aic_alchemist': model_results_1['AIC'],  # AIC value
    'bic_alchemist': model_results_1['BIC'],  # BIC value
}

# Open a file in write-binary mode and save the dictionary
with open(save_path_alchemist_sarima, 'wb') as f:
    pickle.dump(save_data_alchemist_sarima, f)

print(f"Model and data saved to {save_path_alchemist_sarima}")

# Download the file to your local machine
files.download(save_path_alchemist_sarima)

####
#
#
def scale_volume(train_data, test_data, column='Volume'):
    # Convert train and test data to DataFrame if they are Series
    if isinstance(train_data, pd.Series):
        train_data = train_data.to_frame()

    if isinstance(test_data, pd.Series):
        test_data = test_data.to_frame()

    # Check if 'Volume_scaled' already exists in the train or test data
    if 'Volume_scaled' in train_data.columns or 'Volume_scaled' in test_data.columns:
        print("Scaling already applied to these datasets.")
        return train_data, test_data, None

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Apply MinMaxScaler to the train data and create the 'Volume_scaled' column
    train_data['Volume_scaled'] = scaler.fit_transform(train_data[[column]])

    # Apply the same scaler to the test data
    test_data['Volume_scaled'] = scaler.transform(test_data[[column]])

    # Ensure the index remains the same
    train_data = train_data.set_index(train_data.index)
    test_data = test_data.set_index(test_data.index)

    return train_data, test_data, scaler

# Apply the function to Alchemist datasets
train_data_alchemist, test_data_alchemist, scaler_alchemist = scale_volume(train_data_alchemist, test_data_alchemist)

# Apply the function to Caterpillar datasets
train_data_caterpillar, test_data_caterpillar, scaler_caterpillar = scale_volume(train_data_caterpillar, test_data_caterpillar)

# Display the results
display(train_data_alchemist.head())
display(test_data_alchemist.head())
display(train_data_caterpillar.head())
display(test_data_caterpillar.head())

# Sliding window function
def create_input_sequences(lookback, forecast, sequence_data):
  input_sequences = []
  output_sequences = []

  for i in range(lookback, len(sequence_data) - forecast + 1):
      input_sequences.append(sequence_data[i - lookback: i]) # Grab lookback values
      output_sequences.append(sequence_data[i: i + forecast]) # Grab forecast values

  return { "input_sequences": input_sequences,"output_sequences": output_sequences }
  # return np.array(input_sequences), np.array(output_sequences) might need to do this

# Combine the train and test data
combined_data = pd.concat([train_data_alchemist['Volume_scaled'], test_data_alchemist['Volume_scaled']])

# Verify the continuity of the time series
print("Last date of training set:", train_data_alchemist.index[-1])
print("First date of test set:", test_data_alchemist.index[0])

# Check that the dates are directly continuous
assert train_data_alchemist.index[-1] < test_data_alchemist.index[0], "The test data must follow the train data!"

# Define your lookback and forecast values
lookback = 52  # Use the last 52 observations
forecast = 32  # Predict the next 32 observations

# Create input-output sequences for the combined data
combined_sequences = create_input_sequences(lookback, forecast, combined_data.values)

# Separate the input and output sequences
X_combined = np.array(combined_sequences["input_sequences"])
Y_combined = np.array(combined_sequences["output_sequences"])

# Reshape the input sequences for LSTM [samples, time steps, features]
X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)

# Split back into train and test sets based on the dates
train_length = len(train_data_alchemist) - lookback  # Consider the lookback length to avoid overlap

# Split the sequences
X_train, X_test = X_combined[:train_length], X_combined[train_length:]
Y_train, Y_test = Y_combined[:train_length], Y_combined[train_length:]

# Verify shapes
print("X_train shape:", X_train.shape) # Expected: (number_of_samples, 12, 1)
print("Y_train shape:", Y_train.shape) # Expected: (number_of_samples, 32)
print("X_test shape:", X_test.shape) # Expected: (1, 12, 1)
print("Y_test shape:", Y_test.shape) # Expected: (1, 32)

# Optional: Check that the time continuity is preserved by printing the relevant dates
train_start_date = train_data_alchemist.index[0]
train_end_date = train_data_alchemist.index[-1]
test_start_date = test_data_alchemist.index[0]
print(f"Train start date: {train_start_date}, Train end date: {train_end_date}, Test start date: {test_start_date}")

#uncomment below when everything is working and want to add tunable lookback

# Define the model for hyperparameter tuning.
def tuned_model(hp):
    model = Sequential()

    # Tune the lookback parameter
    # Use Input layer to define the input shape
    lookback = hp.Int('lookback', min_value=6, max_value=52, step=5)  # Tunable lookback value #min_value=6, max_value=36, step=6)

    model.add(Input(shape=(lookback, 1)))  # Update the input shape with the tunable lookback
    model.add(LSTM(hp.Int('input_unit', min_value=4, max_value=128, step=8), return_sequences=True))

    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=4, max_value=128, step=8), return_sequences=True))

    model.add(LSTM(hp.Int('layer_2_neurons', min_value=4, max_value=128, step=8)))
    model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))  # Single Dropout layer after the last LSTM

    #model.add(Dense(1, activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='relu')))  # 1 == forecast
    # Output 32 values (for 32-step forecast)
    model.add(Dense(forecast))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model

#"""

# Forecast for the next 32 periods (reuse from code block 1)
n_periods = 32  # Forecast horizon

# Reuse forecast and confidence intervals extraction from Code Block 1
sarima_forecast_result_alchemist = sarima_alchemist.get_forecast(steps=n_periods)  # Similar to forecast_SARIMA_alchemist
sarima_forecast_values_alchemist = sarima_forecast_result_alchemist.predicted_mean  # Similar to predicted_mean_SARIMA_alchemist
sarima_forecast_conf_int_alchemist = sarima_forecast_result_alchemist.conf_int()  # Similar to confidence_intervals_SARIMA_alchemist

# In-sample predictions (reuse from code block 1 for fitted values)
fitted_values_SARIMA_alchemist = sarima_alchemist.get_prediction(start=train_data_alchemist.index[0], end=train_data_alchemist.index[-1])
y_train_pred = fitted_values_SARIMA_alchemist.predicted_mean  # Similar to fitted_values_series_SARIMA_alchemist

# Obtain residuals (specific to block 2)
residuals_train = sarima_alchemist.resid

# Create DataFrame (combine predictions, residuals, and forecast similar to block 1)
result_sarima_df_alchemist = pd.DataFrame({
    'Volume': train_data_alchemist['Volume'],
    'Volume Scaled': train_data_alchemist['Volume_scaled'],
    'SARIMA train_Predict': y_train_pred,  # In-sample predictions
    'SARIMA Residuals': residuals_train,   # In-sample residuals
    'SARIMA Test_Forecast': sarima_forecast_values_alchemist,  # Forecast values
    'SARIMA Forecast CI Lower': sarima_forecast_conf_int_alchemist.iloc[:, 0],  # Lower CI
    'SARIMA Forecast CI Upper': sarima_forecast_conf_int_alchemist.iloc[:, 1]   # Upper CI
})

# Display the result
display(result_sarima_df_alchemist.head())

# Add SARIMA residuals to the DataFrame as a new column
train_data_alchemist['SARIMA Residuals'] = residuals_train
display(train_data_alchemist.head())

# Assuming result_sarima_df_caterpillar and result_sarima_df_alchemist are your DataFrames
# Save the DataFrames as CSV files
result_sarima_df_caterpillar.to_csv('result_sarima_df_caterpillar.csv', index=False)
result_sarima_df_alchemist.to_csv('result_sarima_df_alchemist.csv', index=False)

# If you're using Google Colab or Jupyter Notebook, you can download the files like this:
from google.colab import files

# Download the files to your local machine
files.download('result_sarima_df_caterpillar.csv')
files.download('result_sarima_df_alchemist.csv')

# Assuming result_sarima_df_caterpillar and result_sarima_df_alchemist are your DataFrames
# Save the DataFrames as CSV files
train_data_alchemist.to_csv('train_data_alchemist.csv', index=False)
test_data_alchemist.to_csv('test_data_alchemist.csv', index=False)

train_data_caterpillar.to_csv('train_data_caterpillar.csv', index=False)
test_data_caterpillar.to_csv('test_data_caterpillar.csv', index=False)

# If you're using Google Colab or Jupyter Notebook, you can download the files like this:
from google.colab import files

# Download the files to your local machine
files.download('train_data_alchemist.csv')
files.download('test_data_alchemist.csv')

files.download('train_data_caterpillar.csv')
files.download('test_data_caterpillar.csv')

print(train_data_alchemist.shape)  # For the training DataFrame
print(test_data_alchemist.shape)    # For the test DataFrame

#print("Fitted Values:\n", sarima.fittedvalues.head()) #cant remember what this was supposed to be
#print("Residuals:\n", residuals_train.head()) #cant remember what this was supposed to be need to check older versions from the 4th Oct

# Initialize the scaler
scaler_alchemist = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the SARIMA residuals from the training data
result_sarima_df_alchemist['SARIMA Residuals_scaled'] = scaler_alchemist.fit_transform(result_sarima_df_alchemist[['SARIMA Residuals']].values)

# Transform (not fit) the SARIMA test forecast using the same scaler (use .values to avoid feature name mismatch)
result_sarima_df_alchemist['SARIMA Test_Forecast_Scaled'] = scaler_alchemist.transform(result_sarima_df_alchemist[['SARIMA Test_Forecast']].values)

# Ensure the index remains as the DatetimeIndex from the original Series
result_sarima_df_alchemist = result_sarima_df_alchemist.set_index(result_sarima_df_alchemist.index)

# Display the results for verification
display(result_sarima_df_alchemist[-40:])
display(test_data_alchemist.head())

# Initialize the scaler
scaler_caterpillar = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the SARIMA residuals from the training data
result_sarima_df_caterpillar['SARIMA Residuals_scaled'] = scaler_caterpillar.fit_transform(result_sarima_df_caterpillar[['SARIMA Residuals']].values)

# Transform (not fit) the SARIMA test forecast using the same scaler (use .values to avoid feature name mismatch)
result_sarima_df_caterpillar['SARIMA Test_Forecast_Scaled'] = scaler_caterpillar.transform(result_sarima_df_caterpillar[['SARIMA Test_Forecast']].values)

# Ensure the index remains as the DatetimeIndex from the original Series
result_sarima_df_caterpillar = result_sarima_df_caterpillar.set_index(result_sarima_df_caterpillar.index)

# Display the results for verification
display(result_sarima_df_caterpillar[-40:])
display(test_data_caterpillar.head())

# Load the model results list and DataFrame from the pickle files
with open('model_results_list_seq_SARIMA.pkl', 'rb') as file:
    model_results_list_seq_SARIMA = pickle.load(file)

with open('model_results_df_seq_SARIMA.pkl', 'rb') as file:
    model_results_df_seq_SARIMA = pickle.load(file)

model_results_list = model_results_list_seq_SARIMA
model_results_list

model_results_df.info()

# Flatten the Model Config dictionary into the main DataFrame
model_results_df = pd.json_normalize(model_results_list)
model_results_df

# Define your lookback and forecast values
lookback = 12  # Use the last 12 observations
forecast = 32  # Predict the next 32 observations

# Define the cut-off point (655-32)
cutoff = len(result_sarima_df_alchemist) - forecast

# Create combined_data where the first 655-32 rows come from 'SARIMA Residuals_scaled'
# and the last 32 rows come from 'SARIMA Test_Forecast_Scaled'
combined_data = pd.concat([
    result_sarima_df_alchemist['SARIMA Residuals_scaled'][:cutoff],  # First part
    result_sarima_df_alchemist['SARIMA Test_Forecast_Scaled'][cutoff:]  # Last part
]) #this is overcomplicated but it will work

# Create input-output sequences for the combined data
combined_sequences = create_input_sequences(lookback, forecast, combined_data.values)

# Separate the input and output sequences
X_combined = np.array(combined_sequences["input_sequences"])
Y_combined = np.array(combined_sequences["output_sequences"])

# Reshape the input sequences for LSTM [samples, time steps, features]
X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)

# Split back into train and test sets based on the dates
train_length = len(train_data_alchemist) - lookback  # Consider the lookback length to avoid overlap

# Split the sequences
X_train, X_test = X_combined[:train_length], X_combined[train_length:]
Y_train, Y_test = Y_combined[:train_length], Y_combined[train_length:]

# Verify shapes
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Optional: Check that the time continuity is preserved by printing the relevant dates
train_start_date = train_data_alchemist.index[0]
train_end_date = train_data_alchemist.index[-1]
test_start_date = test_data_alchemist.index[0]
print(f"Train start date: {train_start_date}, Train end date: {train_end_date}, Test start date: {test_start_date}")

""" defined earlier

# Modify the LSTM model for multi-step forecasting
def tuned_model(hp):
    model = Sequential()

    # Input layer with a lookback of 12
    model.add(Input(shape=(lookback, 1)))  # Input shape matches lookback (12 steps)

    model.add(LSTM(hp.Int('input_unit', min_value=4, max_value=128, step=8), return_sequences=True))

    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=4, max_value=128, step=8), return_sequences=True))

    model.add(LSTM(hp.Int('layer_2_neurons', min_value=4, max_value=128, step=8)))
    model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))

    # Output 32 values (for 32-step forecast)
    model.add(Dense(forecast))  # Output layer for 32-step forecast

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model
"""

tuner = RandomSearch(
        tuned_model,
        objective='mse',
        max_trials=3,
        executions_per_trial=1,
        project_name='LSTM_SARIMA_residuals_alchemist'
    )

Duration: 6mins

tuner.search(
        x=X_train,
        y=Y_train,
        epochs=100,
        batch_size=32#,
        #validation_data=(),
)

hp = tuner.get_best_hyperparameters()[0]
hp.values

# Retrieve the best model from the tuner
best_model = tuner.get_best_models(num_models=1)[0]
print(best_model.summary())

# Make predictions
train_best_predict = best_model.predict(X_train)  # Predictions on training data (residuals)
test_best_predict = best_model.predict(X_test)    # Predictions on test data (residuals)

# Check the shapes of the predictions
print("Train predictions shape:", train_best_predict.shape)  # Should be (611, 32)
print("Test predictions shape:", test_best_predict.shape)    # Should be (1, 32)

train_best_predict

# Inverse transform the predictions
train_best_predict = scaler_alchemist.inverse_transform(train_best_predict)
test_best_predict = scaler_alchemist.inverse_transform(test_best_predict)
# If your Y_test was generated with forecast=32, it will contain 32 values for each input
#Y_best_test = scaler.inverse_transform([Y_best_test]).flatten()
Y_best_test = scaler_alchemist.inverse_transform(Y_test.reshape(-1, 1)).flatten()  # Reshape to 2D for inverse scaling

# Flattening the predicted values
train_best_predict_flat = train_best_predict.flatten()
test_best_predict_flat = test_best_predict.flatten()  # Shape: (32,)
print(train_best_predict_flat.shape)
# For fitting, slice the first 623 values from the flattened predictions
#train_best_predict_flat = train_best_predict.flatten()[:len(train_data_alchemist)]
#print(train_best_predict_flat.shape)
# Convert to a Pandas Series for indexing
#fitted_values_series = pd.Series(train_best_predict_flat, index=train_data_alchemist.index[:len(train_best_predict_flat)])
#fitted_values_series

len(test_data_alchemist["Volume"])

test_best_predict

# Reshape test_best_predict to 1D array
#test_best_predict_flat = test_best_predict.flatten()  # Shape will now be (32,) - done above

# Combine the SARIMA predictions and the LSTM predicted residuals for the final forecast
final_forecast = predicted_mean_SARIMA_alchemist.values + test_best_predict_flat

# Calculate the mean squared error on the test set.
mse = mean_squared_error(result_sarima_df_alchemist['SARIMA Test_Forecast'][-32:], final_forecast)
print(f"Mean Squared Error on Test Data: {mse}")

#ignore below output for mse as it was using the wrong data, not going to re run this as not actually using mse as a metric to compare models, instead focusing on mae and mape

print(pd.DataFrame({'Original Data': test_data_alchemist["Volume"],'Hybrid SARIMA + LSTM': final_forecast, 'SARIMA': predicted_mean_SARIMA_alchemist, 'LSTM': test_best_predict_flat})) #'LSTM':xgb_residuals_pred, #ARIMA???

final_forecast

# Call the function with the final forecast and without fitted values
slice_length = 655 - 32

fig, mae, mape = plot_prediction(
    series_train=train_data_alchemist['Volume'][:slice_length],  #result_sarima_df_alchemist['Original Data'] works too, they are the same #looking back on this, it looked like I had sliced incorrectly but on further inspection it was sliced correctly thankfully
    series_test=result_sarima_df_alchemist['SARIMA Test_Forecast'][-32:],
    forecast=final_forecast,  # Ensure final_forecast_series is defined
    forecast_int=None,  # Set to None since we're ignoring confidence intervals
    title='Hybrid Sequential - Alchemist forecast - SARIMA(1, 0, 0)x(1, 0, 0, 52) then LSTM trained on residuals'
)

# Print the metrics
print("MAE:", mae)
print("MAPE:", mape)

fig.show()

lstm_hybrid_hyperparameters = {
    'lookback': 21,
    'input_unit': 108,
    'n_layers': 1,
    'lstm_0_units': 100,
    'layer_2_neurons': 116,
    'Dropout_rate': 0.1,
    'lstm_1_units': 4,
    'trials': 3  # Added this based on your input
}

# The index in model_results_list for Model Number 7 (row 6 in zero-indexed DataFrame)
model_number = 7

# Locate the model in model_results_list with the correct model number
for model in model_results_list:
    if model['Model Number'] == model_number:
        # Update the model name to reflect the hybrid approach
        model['Model Name'] = 'SARIMA(1, 0, 0)x(1, 0, 0, 52) + LSTM on Residuals'

        # Add the LSTM portion parameters under the Model Config
        model['Model Config']['LSTM'] = {
            'Hyperparameters': lstm_hybrid_hyperparameters,
            'Total Parameters': 235552,  # Sum of SARIMA + LSTM parameters
            'Trainable Parameters': 235552,
            'Non-trainable Parameters': 0,
            'Duration': '6 mins (LSTM)'  # This is the duration for the LSTM portion
        }

# Optional: Save back to the DataFrame if you're working with one
model_results_df = pd.json_normalize(model_results_list)
display(model_results_df)

# The index in model_results_list for Model Number 8 (row 7 in zero-indexed DataFrame)
model_number = 7

# Locate the model in model_results_list with the correct model number
for model in model_results_list:
    if model['Model Number'] == model_number:
        # Update the MAE and MAPE values
        model['MAE'] = 9.378952440805733
        model['MAPE'] = 0.014480204012729434

# Optional: Save back to the DataFrame if you're working with one
model_results_df = pd.json_normalize(model_results_list)
display(model_results_df)
