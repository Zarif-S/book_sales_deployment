!pip install keras-tuner --upgrade
!pip install sktime==0.28.0
!pip install pmdarima

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.layers import Input
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Import XGBoost.
import xgboost as xgb
from xgboost import XGBRegressor

import shutil
from google.colab import files

from sktime.forecasting.compose import (make_reduction)
from sktime.forecasting.model_selection import (ExpandingWindowSplitter, ForecastingGridSearchCV)
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
#plt.rcParams["figure.figsize"] = (10, 5)

# Import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.graph_objs as go

# Import of time series related libraries
import statsmodels.api as sm
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
import statsmodels.graphics.api as smgraphics
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
import gdown

!pip install mpld3
import mpld3

import plotly.io as pio

import pickle #or can use joblib

# import warnings
# warnings.filterwarnings('ignore')
# pd.set_option('display.width', None)

"""# Load the data"""

url_ISBN = 'https://docs.google.com/spreadsheets/d/1OlWWukTpNCXKT21n92yGF1rWHZPwy3a-/edit?usp=sharing&ouid=108468335093833318634&rtpof=true&sd=true'

# Define the destination file name
destination_file = 'ISBN.xlsx'

# Construct the download URL using the file ID
download_url = 'https://docs.google.com/spreadsheets/d/1OlWWukTpNCXKT21n92yGF1rWHZPwy3a-/export?format=xlsx'

# Download the file using gdown
gdown.download(download_url, destination_file, quiet=False)

# Load all four sheets into separate DataFrames and add a column for the source
df_fiction = pd.read_excel('ISBN.xlsx', sheet_name="F - Adult Fiction")
df_fiction['Source'] = 'F - Adult Fiction'

df_non_fiction_specialist = pd.read_excel('ISBN.xlsx', sheet_name="S Adult Non-Fiction Specialist")
df_non_fiction_specialist['Source'] = 'S Adult Non-Fiction Specialist'

df_non_fiction_trade = pd.read_excel('ISBN.xlsx', sheet_name="T Adult Non-Fiction Trade")
df_non_fiction_trade['Source'] = 'T Adult Non-Fiction Trade'

df_children_ya = pd.read_excel('ISBN.xlsx', sheet_name="Y Children's, YA & Educational")
df_children_ya['Source'] = "Y Children's, YA & Educational"

# Concatenate all four DataFrames into one final DataFrame with the source column
df_ISBNs = pd.concat([df_fiction,
                      df_non_fiction_specialist,
                      df_non_fiction_trade,
                      df_children_ya], ignore_index=True)

# Print the data from one of the sheets
df_ISBNs.head()

df_ISBNs.info()

missing_values_count = df_ISBNs.isna().sum()
print(f"Number of missing values in df_ISBNs: \n{missing_values_count}")

"""There are some nulls present in RRP and Author features/columns"""

# Define the destination file name
destination_file = 'UK_weekly_data.xlsx'

# Construct the download URL using the new file ID
download_url = 'https://docs.google.com/spreadsheets/d/1uMYiErVo4YI8QVBGzEXixtoxTfgnpgHi/export?format=xlsx'

# Download the file using gdown
gdown.download(download_url, destination_file, quiet=False)

# Load all four sheets into separate DataFrames and add a column for the source
df_fiction_uk = pd.read_excel(destination_file, sheet_name="F Adult Fiction")
df_fiction_uk['Source'] = 'F - Adult Fiction'

df_non_fiction_specialist_uk = pd.read_excel(destination_file, sheet_name="S Adult Non-Fiction Specialist")
df_non_fiction_specialist_uk['Source'] = 'S Adult Non-Fiction Specialist'

df_non_fiction_trade_uk = pd.read_excel(destination_file, sheet_name="T Adult Non-Fiction Trade")
df_non_fiction_trade_uk['Source'] = 'T Adult Non-Fiction Trade'

df_children_ya_uk = pd.read_excel(destination_file, sheet_name="Y Children's, YA & Educational")
df_children_ya_uk['Source'] = "Y Children's, YA & Educational"

# Concatenate all four DataFrames into one final DataFrame for the UK weekly data
df_UK_weekly = pd.concat([df_fiction_uk,
                          df_non_fiction_specialist_uk,
                          df_non_fiction_trade_uk,
                          df_children_ya_uk], ignore_index=True)

# Print the first few rows of the final DataFrame
df_UK_weekly.head()

df_UK_weekly.info()

missing_values_count = df_UK_weekly.isna().sum()
print(f"Number of missing values in df_UK_weekly: {missing_values_count}")

# Step 1: Merge df_UK_weekly with df_ISBNs on the ISBN column
df_merged = df_UK_weekly.merge(df_ISBNs[['ISBN', 'Author']], on='ISBN', how='left', suffixes=('', '_ISBN'))
print("Sample of merged DataFrame (showing rows where Author is missing in df_UK_weekly):")
display(df_merged[df_merged['Author'].isna()].head())

# Step 2: Fill missing 'Author' values in df_UK_weekly with 'Author' from df_ISBNs only when available
df_merged['Author'] = df_merged['Author'].fillna(df_merged['Author_ISBN'])
print("\nSample after filling missing 'Author' values (showing rows where Author was previously missing):")
display(df_merged[df_merged['Author_ISBN'].notna()].head())

# Step 3: Drop the extra column from the merge
df_UK_weekly_filled = df_merged.drop(columns=['Author_ISBN'])
print("\nSample after dropping extra columns:")
display(df_UK_weekly_filled.head())

# Step 4: Check for remaining missing values
missing_values_count_filled = df_UK_weekly_filled.isna().sum()
print(f"\nNumber of missing values after filling: {missing_values_count_filled}")

df_merged = None
missing_values_count_filled = None

"""seems no impact to author missing values (rrp too, code for it removed)"""

unique_titles = df_UK_weekly['Title'].unique()
print(unique_titles)

"""The Alchemist"""

df_ISBNs_raw = df_ISBNs.copy()
df_UK_weekly_raw = df_UK_weekly.copy()

"""# 2. Conducting initial data investigation

1. Note that the data provided is weekly data. If no sales happened in a particular week, there will be no data representation for that week. This means that the data is not at fixed intervals.
As a result, resample the data and fill in missing values with 0, such that even weeks with 0 sales is represented.
2. Convert the ISBNs to a string value.
3. Convert date to datetime object. (Recall that setting the date as the index has several advantages for time series handling.)
4. Filter out the ISBNs (from all four tabs) wherein sales data exists beyond 2024-07-01. Show all the ISBNs that satisfy this criterion. Capture this in your report.
5. Plot the data of all the ISBNs from the previous step by placing them in a loop.
6. Investigate these plots to understand the general sales patterns, and comment on the general patterns visible in the data. Do the patterns drastically change for the period 1–12 years vs the period 12–24 years? Explain why or why not with possible reasons.
7. Select The Alchemist from the list for further analysis. Focus on the period >2012-01-01. Filter the sales data for this book to retain the date range >2012-01-01, until the final datapoint.
"""

display(df_ISBNs_raw.head())
display(df_UK_weekly_raw.head())

# Check the data type of the ISBN column
print("Data type of ISBN before conversion:", df_UK_weekly_raw['ISBN'].dtype)

# Convert ISBN to string
df_UK_weekly_raw['ISBN'] = df_UK_weekly_raw['ISBN'].astype(str)

# Verify the conversion
print("Data type of ISBN after conversion:", df_UK_weekly_raw['ISBN'].dtype)

# Set 'End Date' as index
df_UK_weekly_raw.set_index('End Date', inplace=True)

display(df_UK_weekly_raw.head())

# Sort the DataFrame by index (End Date) in chronological order
df_UK_weekly_raw.sort_index(inplace=True)

# Display the sorted DataFrame for the specific ISBN
display(df_UK_weekly_raw[df_UK_weekly_raw['ISBN'] == '9781903840122'].head())

# Initialize a list to hold the completed DataFrames for each ISBN
completed_dfs = []

# Group by ISBN and fill in missing weeks using asfreq
for isbn, group in df_UK_weekly_raw.groupby('ISBN'):
    group = group.asfreq('W-SAT', fill_value=0)  # Fill missing dates with 0
    group['ISBN'] = isbn  # Add back the ISBN column
    completed_dfs.append(group)

# Concatenate all completed DataFrames
df_filled = pd.concat(completed_dfs)

df_filled

# Group by 'ISBN'
grouped = df_filled.groupby('ISBN')

# Define an aggregation function to retain non-numeric columns and resample numeric columns
def resample_group(group):
    # Resample the 'Value' column and other numeric columns
    resampled = group.resample('W-SAT').agg({
        'Value': 'mean',        # Weekly average of Value
        'ASP': 'mean',          # Weekly average of ASP
        'RRP': 'mean',          # Weekly average of RRP
        'Volume': 'mean',       # Weekly average of Volume
        'Title': 'first',       # Keep first occurrence of Title
        'Author': 'first',      # Keep first occurrence of Author
        'Binding': 'first',     # Keep first occurrence of Binding
        'Imprint': 'first',     # Keep first occurrence of Imprint
        'Publisher Group': 'first',  # Keep first occurrence of Publisher Group
        'Product Class': 'first',    # Keep first occurrence of Product Class
        'Source': 'first',           # Keep first occurrence of Source
        # Add other columns as necessary
    })

    return resampled

# Apply the function to each group
weekly_resampled = grouped.apply(resample_group)

# Reset index to make it easier to work with
weekly_resampled = weekly_resampled.reset_index()

# Check the result
display(weekly_resampled.head())

# Set 'End Date' as index
weekly_resampled.set_index('End Date', inplace=True)

# Sort the DataFrame by index (End Date) in chronological order
weekly_resampled.sort_index(inplace=True)

weekly_resampled

# Display the sorted DataFrame for the specific ISBN
display(weekly_resampled[weekly_resampled['ISBN'] == '9781903840122'])

# Sort the DataFrame by index (End Date) in chronological order
df_filled.sort_index(inplace=True)

# Display the sorted DataFrame for the specific ISBN
display(df_filled[df_filled['ISBN'] == '9781903840122'])

#df_weekly_resampled_9781903840122.head()

#df_filled_9781903840122.head()

# Filter the data for ISBN '9781903840122'
df_weekly_resampled_9781903840122 = weekly_resampled[weekly_resampled['ISBN'] == '9781903840122']
df_weekly_resampled_9781903840122['Volume'] = df_weekly_resampled_9781903840122['Volume'].astype('float64')

df_filled_9781903840122 = df_filled[df_filled['ISBN'] == '9781903840122'].drop(columns=['Interval'])
df_weekly_resampled_9781903840122 = df_weekly_resampled_9781903840122[df_filled_9781903840122.columns]

df_weekly_resampled_9781903840122 = df_weekly_resampled_9781903840122.reset_index(drop=True)
df_filled_9781903840122 = df_filled_9781903840122.reset_index(drop=True)

# Step 1: Compare data types
print("Data types in weekly_resampled:")
print(df_weekly_resampled_9781903840122.dtypes)

print("\nData types in df_filled:")
print(df_filled_9781903840122.dtypes)

# Step 2: Compare statistical summary (mean, std, etc.)
print("\nStatistical summary for weekly_resampled:")
display(df_weekly_resampled_9781903840122.describe())

print("\nStatistical summary for df_filled:")
display(df_filled_9781903840122.describe())

# Step 3: Check if all columns are equal (if not, it will display the differences)
comparison = df_weekly_resampled_9781903840122.compare(df_filled_9781903840122, align_axis=0)
if comparison.empty:
    print("\nNo differences between the two DataFrames.")
else:
    print("\nDifferences between the two DataFrames:")
    display(comparison)

# Filter the DataFrame for dates after 2024-07-01
df_UK_weekly_filtered = df_filled[df_filled.index >= '2024-07-01']

# Display the filtered DataFrame
display(df_UK_weekly_filtered)

# Check the date range for df_UK_weekly_raw
date_range = df_UK_weekly_raw.index.min(), df_UK_weekly_raw.index.max()
print("Date Range:", date_range)

date_range_filtered = df_UK_weekly_filtered.index.min(), df_UK_weekly_filtered.index.max()
print("Date Range filtered:", date_range_filtered)

"""These are the ISBNs wherin sales data exists beyond 2024-07-01"""

# Get unique ISBNs
unique_isbns = df_UK_weekly_filtered['ISBN'].unique()

print(unique_isbns)
print(len(unique_isbns))

# Set up the plot
plt.figure(figsize=(12, 6))

# Loop through each unique ISBN and plot the Volume
for isbn in unique_isbns:
    plt.plot(df_UK_weekly_filtered[df_UK_weekly_filtered['ISBN'] == isbn].index,
             df_UK_weekly_filtered[df_UK_weekly_filtered['ISBN'] == isbn]['Volume'],
             marker='o', label=isbn)

# Customize the plot
plt.title('Weekly Volume for Each ISBN > 2024-07-01')
plt.xlabel('End Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)
plt.legend(title='ISBN', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()

# Create a figure
fig = go.Figure()

# Loop through each unique ISBN and add a trace to the plot
for isbn in unique_isbns:
    current_data = df_UK_weekly_filtered[df_UK_weekly_filtered['ISBN'] == isbn]

    # Check if there's data for the current ISBN
    if not current_data.empty:
        fig.add_trace(go.Scatter(
            x=current_data.index,
            y=current_data['Volume'],
            mode='lines+markers',  # Use lines and markers
            name=isbn  # Use ISBN as the trace name
        ))

# Customize the layout
fig.update_layout(
    title='Weekly Volume for Each ISBN',
    xaxis_title='End Date',
    yaxis_title='Volume',
    legend_title='ISBN',
    xaxis=dict(tickangle=45),
    template='plotly_white',  # Optional: Use a clean layout
)

# Show the plot
fig.show()

"""Below plot is not required in rubric, done optionally"""

# Check the date range for df_filled
date_range = df_filled.index.min(), df_filled.index.max()
print("Date Range:", date_range)

# Filter the DataFrame for the first 12 years (2001 to 2013)
df_UK_weekly_filtered_first_12 = df_filled[df_filled.index < '2013-01-06']

# Aggregate the data to yearly sums (or means)
df_UK_yearly = df_UK_weekly_filtered_first_12.groupby(['ISBN', df_UK_weekly_filtered_first_12.index.year]) \
    .agg({'Volume': 'sum'}).reset_index()

# Rename the year column for clarity
df_UK_yearly.rename(columns={'index': 'Year', df_UK_weekly_filtered_first_12.index.year.name: 'Year'}, inplace=True)

date_range_filtered = df_UK_weekly_filtered_first_12.index.min(), df_UK_weekly_filtered_first_12.index.max()
print("Date Range filtered:", date_range_filtered)

# Set up the plot
plt.figure(figsize=(12, 6))

# Loop through each unique ISBN and plot the yearly aggregated Volume
for isbn in unique_isbns:
    isbn_data = df_UK_yearly[df_UK_yearly['ISBN'] == isbn]
    plt.plot(isbn_data['Year'], isbn_data['Volume'], marker='o', label=isbn)  # Accessing the Year column correctly

# Customize the plot
plt.title('Yearly Volume for Each ISBN (First 12 Years)')
plt.xlabel('Year')
plt.ylabel('Volume')

# Format y-axis with commas
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xticks(rotation=45)
plt.legend(title='ISBN', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# Adjust layout to avoid overlaps
plt.tight_layout()

# Show the plot
plt.show()

# Create a Plotly figure
fig = go.Figure()

# Loop through each unique ISBN and add a trace for the yearly aggregated Volume
for isbn in unique_isbns:
    isbn_data = df_UK_yearly[df_UK_yearly['ISBN'] == isbn]
    fig.add_trace(go.Scatter(
        x=isbn_data['Year'],
        y=isbn_data['Volume'],
        mode='lines+markers',
        name=isbn
    ))

# Customize the plot
fig.update_layout(
    title='Yearly Volume for Each ISBN (First 12 Years)',
    xaxis_title='Year',
    yaxis_title='Volume',
    yaxis=dict(tickformat=','),
    legend_title='ISBN',
    xaxis_tickangle=45,
    margin=dict(l=0, r=0, t=50, b=0)
)

# Show the plot
fig.show()

# Filter the DataFrame for the last 12 years (2012 to 2024)
df_UK_weekly_filtered_last_12 = df_filled[df_filled.index >= '2012-07-20']

# Aggregate the data to yearly sums (or means)
df_UK_yearly_last_12 = df_UK_weekly_filtered_last_12.groupby(['ISBN', df_UK_weekly_filtered_last_12.index.year]) \
    .agg({'Volume': 'sum'}).reset_index()

# Print the structure of the aggregated DataFrame
print(df_UK_yearly_last_12.head())
print(df_UK_yearly_last_12.columns)  # Check column names

# Aggregate the data to yearly sums based on 'End Date'
df_UK_yearly_last_12 = df_UK_weekly_filtered_last_12.groupby(['ISBN', df_UK_weekly_filtered_last_12.index.year]) \
    .agg({'Volume': 'sum'}).reset_index()

# Set up the plot
plt.figure(figsize=(12, 6))

# Loop through each unique ISBN and plot the yearly aggregated Volume
for isbn in df_UK_yearly_last_12['ISBN'].unique():
    isbn_data = df_UK_yearly_last_12[df_UK_yearly_last_12['ISBN'] == isbn]
    plt.plot(isbn_data['End Date'], isbn_data['Volume'], marker='o', label=isbn)

# Customize the plot
plt.title('Yearly Volume for Each ISBN (Last 12 Years)')
plt.xlabel('Year')
plt.ylabel('Volume')

# Format y-axis with commas
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xticks(rotation=45)
plt.legend(title='ISBN', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# Adjust layout to avoid overlaps
plt.tight_layout()

# Show the plot
plt.show()

# Aggregate the data to yearly sums based on 'End Date'
df_UK_yearly_last_12 = df_UK_weekly_filtered_last_12.groupby(['ISBN', df_UK_weekly_filtered_last_12.index.year]) \
    .agg({'Volume': 'sum'}).reset_index()

# Create a Plotly figure
fig = go.Figure()

# Loop through each unique ISBN and add a trace for the yearly aggregated Volume
for isbn in df_UK_yearly_last_12['ISBN'].unique():
    isbn_data = df_UK_yearly_last_12[df_UK_yearly_last_12['ISBN'] == isbn]
    fig.add_trace(go.Scatter(
        x=isbn_data['End Date'],
        y=isbn_data['Volume'],
        mode='lines+markers',
        name=isbn
    ))

# Customize the plot
fig.update_layout(
    title='Yearly Volume for Each ISBN (Last 12 Years)',
    xaxis_title='Year',
    yaxis_title='Volume',
    yaxis=dict(tickformat=','),
    legend_title='ISBN',
    xaxis_tickangle=45,
    margin=dict(l=0, r=0, t=50, b=0)
)

# Show the plot
fig.show()

# Filter the DataFrame for the full 24 years
df_UK_weekly_full_24 = df_filled

# Aggregate the data to yearly sums based on the 'End Date' column
df_UK_weekly_full_24 = df_UK_weekly_full_24.groupby(['ISBN', df_UK_weekly_full_24.index.year]) \
    .agg({'Volume': 'sum'}).reset_index()

# Rename the 'End Date' column to 'Year' for easier access
df_UK_weekly_full_24.rename(columns={'End Date': 'Year'}, inplace=True)

# Print the structure of the aggregated DataFrame
print(df_UK_weekly_full_24.head())
print(df_UK_weekly_full_24.columns)  # Check column names

# Set up the plot
plt.figure(figsize=(12, 6))

# Loop through each unique ISBN and plot the yearly aggregated Volume
for isbn in df_UK_weekly_full_24['ISBN'].unique():
    isbn_data = df_UK_weekly_full_24[df_UK_weekly_full_24['ISBN'] == isbn]
    plt.plot(isbn_data['Year'], isbn_data['Volume'], marker='o', label=isbn)  # Use 'Year' for the x-axis

# Customize the plot
plt.title('Yearly Volume for Each ISBN (24 years)')
plt.xlabel('Year')
plt.ylabel('Volume')

# Format y-axis with commas
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xticks(rotation=45)
plt.legend(title='ISBN', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# Adjust layout to avoid overlaps
plt.tight_layout()

# Show the plot
plt.show()

# Check for any whitespace in the column names
print(df_UK_yearly_last_12.columns)

# Set up the figure and axes for side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# Plot for the first 12 years
for isbn in unique_isbns:
    isbn_data = df_UK_yearly[df_UK_yearly['ISBN'] == isbn]
    axs[0].plot(isbn_data['Year'], isbn_data['Volume'], marker='o', label=isbn)  # Accessing the Year column correctly

# Customize the first plot
axs[0].set_title('Yearly Volume for Each ISBN (First 12 Years)')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Volume')
axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend(title='ISBN', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[0].grid()

# Plot for the last 12 years
for isbn in df_UK_yearly_last_12['ISBN'].unique():
    isbn_data = df_UK_yearly_last_12[df_UK_yearly_last_12['ISBN'] == isbn]
    axs[1].plot(isbn_data['End Date'], isbn_data['Volume'], marker='o', label=isbn)  # Use 'Year' for the x-axis

# Customize the second plot
axs[1].set_title('Yearly Volume for Each ISBN (Last 12 Years)')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Volume')
axs[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend(title='ISBN', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].grid()

# Adjust layout to avoid overlaps
plt.subplots_adjust(wspace=0.55)  # Adjust the width space between the plots

# Adjust layout to avoid overlaps
plt.tight_layout()

# Show the plots
plt.show()

df_filled.info()

# Filter for books with the specified titles, case-insensitive (only The Alchemist)
filtered_books_titles = df_filled[df_filled['Title'].str.contains(r'Alchemist,? The', case=False, na=False)]

# Display the first few rows to check the filtering
display(filtered_books_titles.head())

filtered_books_alchemist = df_filled[df_filled['Title'].str.contains(r'Alchemist', case=False, na=False)]
unique_books = filtered_books_alchemist[['ISBN', 'Title', 'Binding', 'RRP']].drop_duplicates()
display(unique_books)

# Group by ISBN and Title, then count the occurrences
book_counts_filtered = filtered_books_alchemist.groupby(['ISBN', 'Title', 'Binding', 'RRP']).size().reset_index(name='Count')

# Display the result
display(book_counts_filtered)

"""The Alchemist data processing"""

# Step 1: Filter the data for "The Alchemist"
filtered_books_alchemist = df_filled[df_filled['Title'].str.contains(r'Alchemist,? The', case=False, na=False)]

# Step 2: Keep the data for "The Alchemist" as-is, and ensure the index is preserved
filtered_alchemist = filtered_books_alchemist[filtered_books_alchemist['Title'].str.contains(r'Alchemist,? The', case=False, na=False)][[
    'ISBN', 'Title', 'Author', 'Volume', 'Value', 'ASP', 'RRP', 'Imprint', 'Publisher Group']]

# The index is already properly set as "End Date"
alchemist_books = filtered_alchemist.copy()

# Sort the data by the End Date for readability
alchemist_books.sort_index(inplace=True)

# Display the result
display(alchemist_books.head())


# Step 1: Filter for The Alchemist ISBN and the date range
filtered_sales_data_alchemist = df_filled[
    ((df_filled['ISBN'].isin(['9780722532935'])) &
    (df_filled.index >= '2012-01-01'))
]

# Print the result
display(filtered_sales_data_alchemist)

# Create a mapping of ISBNs to Titles
isbn_to_title = {
    '9780722532935': 'Alchemist, The'
}

# Set up the plot
plt.figure(figsize=(12, 6))

# Loop through each unique ISBN in the filtered data and plot the yearly aggregated Volume
for isbn in filtered_sales_data_alchemist['ISBN'].unique():
    isbn_data = filtered_sales_data_alchemist[filtered_sales_data_alchemist['ISBN'] == isbn]
    # Use both ISBN and Title for the label
    label = f"{isbn} - {isbn_to_title[isbn]}"

    # Set color for The Alchemist
    if isbn == '9780722532935':
        color = 'green'  # Set green for 'Alchemist, The'
    else:
        color = 'green'     # Default color for The Alchemist

    plt.plot(isbn_data.index, isbn_data['Volume'], marker='o', label=label, color=color)

# Customize the plot
plt.title('Sales Data (Volume) for The Alchemist (From 2012 Onward)')
plt.xlabel('Year')
plt.ylabel('Sales Volume')

# Format y-axis with commas
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xticks(rotation=45)
plt.legend(title='ISBN - Book Title', bbox_to_anchor=(1.05, 1), loc='upper left')  # Update legend title
plt.grid()

# Adjust layout to avoid overlaps
plt.tight_layout()

# Show the plot
plt.show()

# Create a mapping of ISBNs to Titles
isbn_to_title = {
    '9780722532935': 'Alchemist, The'
}

# Aggregate data by year
filtered_sales_data_2_books['Year'] = filtered_sales_data_2_books.index.year
yearly_volume = filtered_sales_data_2_books.groupby(['Year', 'ISBN'])['Volume'].sum().reset_index()

# Create a Plotly figure
fig = go.Figure()

# Loop through each unique ISBN in the aggregated data and add traces to the figure
for isbn in yearly_volume['ISBN'].unique():
    isbn_data = yearly_volume[yearly_volume['ISBN'] == isbn]
    title = isbn_to_title[isbn]  # Get the title for the current ISBN
    # Use both ISBN and Title for the name
    fig.add_trace(go.Scatter(x=isbn_data['Year'],
                             y=isbn_data['Volume'],
                             mode='lines+markers',
                             name=f"{isbn} - {title}"))  # Concatenate ISBN and title

# Customize the layout
fig.update_layout(title='Sales Data (Volume) for Two Books (From 2012 Onward)',
                  xaxis_title='Year',
                  yaxis_title='Sales Volume',
                  yaxis_tickformat=',',  # Format y-axis with commas
                  legend_title='ISBN - Book Title',
                  template='plotly_white')

# Show the figure
fig.show()

"""Focusing on The Alchemist for time series analysis and forecasting.
"""

filtered_sales_data_2_books.info()

"""# 3. Classical techniques:

filtered_sales_data_alchemist.head()

# Filter for The Alchemist ISBN
book1_data_alchemist = filtered_sales_data_alchemist[filtered_sales_data_alchemist['ISBN'] == '9780722532935'] # The Alchemist

"""## Decomposition"""

# Plot the sales data for the first book
plt.figure(figsize=(14, 4))
plt.plot(book1_data_alchemist.index, book1_data_alchemist['Volume'], label='9780722532935', color='black')
plt.title('Weekly Sales Data (Volume) for The Alchemist ISBN Paperback')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()


# Create a Plotly figure for "The Alchemist"
fig_alchemist = go.Figure()

# Add trace for the Alchemist book
fig_alchemist.add_trace(go.Scatter(x=book1_data_alchemist.index,
                                    y=book1_data_alchemist['Volume'],
                                    mode='lines+markers',
                                    name='9780722532935',
                                    line=dict(color='black')))

# Customize the layout for "The Alchemist"
fig_alchemist.update_layout(title='Weekly Sales Data (Volume) for The Alchemist ISBN Paperback',
                             xaxis_title='Date',
                             yaxis_title='Volume',
                             template='plotly_white',
                             width=1100,  # Set figure width
                             height=400)  # Set figure height

# Show the figure for "The Alchemist"
fig_alchemist.show()


"""Seasonal variation around Christmas - (Dec 12/22) for The Alchemist.

Interestingly, there are 2 childrens book weeks in a year [reference](https://www.publishersweekly.com/pw/by-topic/childrens/childrens-industry-news/article/91547-2023-children-s-book-week-theme-and-poster-revealed.html) in 2023 this was held May 1–7 and November 6–12.


This trend does not repeat in May so this could be coincidental with christmas sales, however there is an increase in sales in early march, further investigation is required as to why this could be.

Additive decomposition makes the most sense. It is used if the seasonal variation is relatively constant over time or the variation around the trend does not vary with the level of the time series.
"""

# Decomposing the time series using STL
stl_object = STL(book1_data_alchemist.Volume, period=52, seasonal=53,trend=None)
stl_results = stl_object.fit()

# Plotting the STL decomposition
stl_results.plot()
plt.suptitle('Decomposition for The Alchemist', y=1.01)
plt.show()

# Store the STL residuals
stl_residuals_book1 = stl_results.resid

# Decomposing the time series using STL
stl_object = STL(book1_data_alchemist.Volume, period=208, seasonal=53,trend=None)
stl_results = stl_object.fit()

# Plotting the STL decomposition
stl_results.plot()
plt.suptitle('Decomposition for The Alchemist', y=1.01)
plt.show()

# Store the STL residuals
stl_residuals_book1_test = stl_results.resid

# Decomposing the time series using an additive model
decomposition = seasonal_decompose(book1_data_alchemist.Volume, model='additive', period=52)

# Plotting the decomposition
decomposition.plot()
plt.suptitle('Additive Decomposition for The Alchemist', y=1.01)
plt.show()

# Storing the residuals
residuals_book1 = decomposition.resid

# Extracting the trend components
trend_book1 = decomposition.trend.dropna()

trend_book1

# Apply STL decomposition
plt.figure(figsize=(14, 4))

# Decomposing with different parameters
st1 = STL(book1_data_alchemist.Volume,
          period=52, # period of the seasonal component,
          seasonal=53, # allows seasonal component to adapt, small numbers more adaptive,
          trend=None) # allows trend component to adapt, small numbers more adaptive

res1 = st1.fit()

st2 = STL(book1_data_alchemist.Volume, period=10, seasonal=13, trend=None)
res2 = st2.fit()

# Plot the results
plt.plot(book1_data_alchemist.index, book1_data_alchemist['Volume'], label='Actual Volume', color='grey', alpha=0.7)
plt.plot(book1_data_alchemist.index, res1.trend, label='Trend with reasonable parameters - STL', color='blue')
plt.plot(trend_book1.index, trend_book1, label='Trend - Seasonal Additive Decomposition', color='red')
plt.plot(book1_data_alchemist.index, res2.trend, label='Trend with smaller period and seasonality (wrong) - STL', color='orange')

# Adding labels and title
plt.legend()
plt.title('Weekly Volume Data for The Alchemist ISBN 9780722532935')
plt.xlabel('Date')
plt.ylabel('Volume')

# Adjusting y-axis limits if necessary
# plt.ylim(lower_limit, upper_limit) # Uncomment and set limits if needed

plt.show()

"""STL and seasonal decomp perform similarly, wrong period and seasonality shows that the trend retains the seasonality component"""


zero_volume_weeks_alchemist = book1_data_alchemist[book1_data_alchemist['Volume'] == 0]

# Print the weeks with zero volume
print("Weeks with zero volume:")
print(zero_volume_weeks_alchemist[['ISBN', 'Title', 'Volume']])


# Create a figure
fig = go.Figure()

# Add the sales data for the first book
fig.add_trace(go.Scatter(
    x=book1_data_alchemist.index,
    y=book1_data_alchemist['Volume'],
    mode='lines',
    name='9780722532935 - The Alchemist',
    line=dict(color='black'),
    hoverinfo='x+y'
))


# Highlight the weeks with zero volume for The Alchemist
fig.add_trace(go.Scatter(
    x=zero_volume_weeks_alchemist.index,
    y=zero_volume_weeks_alchemist['Volume'],
    mode='markers',
    name='Zero Volume',
    marker=dict(color='red', size=8),
    hoverinfo='x+y'
))

# Update layout
fig.update_layout(
    title='Weekly Sales Data (Volume) for The Alchemist',
    xaxis_title='Date',
    yaxis_title='Volume',
    legend_title='Books',
    template='plotly_white'
)

# Show the plot
fig.show()

# 0 weeks are the same so

zero_volume_weeks = book1_data_alchemist[book1_data_alchemist['Volume'] == 0]

zero_volume_weeks.head()

"""0 values are covid weeks - https://www.instituteforgovernment.org.uk/sites/default/files/timeline-lockdown-web.pdf

26 March
Lockdown measures
legally come into force

15 June
Non-essential shops
reopen in England

5 November
Second national
lockdown comes
into force in
England

6 January
England enters
third national
lockdown

22 February
PM expected to
publish roadmap
for lifting the
lockdown
"""

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

# Display the first 10 entries of the time series with index, Volume, and Lockdown status
for i in range(10):
    end_date = zero_volume_weeks.index[i]  # Get the date from the index
    volume = zero_volume_weeks['Volume'].iloc[i]  # Get the Volume value
    lockdown_status = zero_volume_weeks['Lockdown'].iloc[i]  # Get the Lockdown status
    print(f"End Date: {end_date}, Volume: {volume}, Lockdown: {lockdown_status}")

display(zero_volume_weeks)

# Create ACF plot.
smgraphics.tsa.plot_acf(book1_data_alchemist.Volume, alpha=0.05);
plt.title('Autocorrelation Function (ACF) for The Alchemist');

# ACF default arguments.
# x                     : The time series object.
# lags=None             : Number of lags to calculate. If None selects automatically.
# alpha=0.05            : Confidence level to use for insignificance region.
# adjusted=False        : Related to calculation method.
# fft=False             : Related to calculation method (fast fourier transform).
# missing='none'        : How to treat missing values.
# zero=True             : Return 0-lag autocorrelation?
# bartlett_confint=True : Related to calculation of insignificance region.

# Create ACF plot.

"""ACF shows strong autocorrelation for The Alchemist that decays slowly as the lags increase - statistically significant, there also seems to be some seasonality present. No sharp cutoff in the ACF so cannot determine MA (Q) order easily"""

# Create PACF plot.
smgraphics.tsa.plot_pacf(book1_data_alchemist.Volume, alpha=0.05);
plt.title('Partial Autocorrelation Function (PACF) for The Alchemist');

# PACF default arguments.
# x           : The time series object
# lags=None   : Number of lags to calculate. If None selects automatically
# alpha=0.05  : Confidence level to use for insignificance region
# method='ywm': Related to calculation method
# zero=True   : Return 0-lag autocorrelation?


from statsmodels.stats.diagnostic import acorr_ljungbox

# Calculate The Alchemist / book 1 p-values.

print('Ljung-Box test output for the Alchemist / book 1\n', acorr_ljungbox(book1_data_alchemist.Volume), '...\n')

# Create ACF plot.
smgraphics.tsa.plot_acf(stl_residuals_book1, alpha=0.05);
plt.title('Autocorrelation Function (ACF) for The Alchemist');

# ACF default arguments.
# x                     : The time series object.
# lags=None             : Number of lags to calculate. If None selects automatically.
# alpha=0.05            : Confidence level to use for insignificance region.
# adjusted=False        : Related to calculation method.
# fft=False             : Related to calculation method (fast fourier transform).
# missing='none'        : How to treat missing values.
# zero=True             : Return 0-lag autocorrelation?
# bartlett_confint=True : Related to calculation of insignificance region.


# Calculate The Alchemist / book 1 p-values.

print('Ljung-Box test output for the Alchemist / book 1\n', acorr_ljungbox(stl_residuals_book1), '...\n')


"""## Stationarity test"""

# Testing for stationarity - ADF Test
adf_result = adfuller(book1_data_alchemist.Volume)
print('p-value. original data:', adf_result[1])


adf_result = adfuller(stl_residuals_book1)
print('p-value. residual data:', adf_result[1])


def prepare_data_after_2012(book_data, column_name, split_size=32):
    """
    Prepare training and testing data after 2012-01-01 based on a given split size.

    Args:
        book_data (pd.DataFrame): The DataFrame containing the book data with a time series index.
        column_name (str): The column to split into train and test data.
        split_size (int): The number of entries (weeks or months) to include in the test set.

    Returns:
        pd.Series: Training data (all data except the last split_size weeks/months).
        pd.Series: Test data (last split_size weeks/months).
    """
    # Filter data for dates after 2012-01-01 inclusive
    data_after_2012 = book_data[book_data.index >= '2012-01-01']

    # Ensure there is enough data for splitting
    if len(data_after_2012) < split_size:
        raise ValueError(f"Not enough data available for the test set (at least {split_size} entries required).")

    # Split into train and test data
    train_data = data_after_2012[column_name].iloc[:-split_size]  # All data except the last split_size entries
    test_data = data_after_2012[column_name].iloc[-split_size:]   # Last split_size entries of data

    # Display the results
    display(train_data)
    display(test_data)

    return train_data, test_data

# Prepare data for 'Volume' column for Book 1
train_data_alchemist, test_data_alchemist = prepare_data_after_2012(book1_data_alchemist, 'Volume', 32)


# Updated plot_prediction function
def plot_prediction(series_train, series_test, forecast, forecast_int=None, fitted_values=None, title="Forecast Plot"):
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
        title=title,  # Use the provided title parameter
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(font=dict(size=16)),
        template='plotly_white',
        width=800,
        height=500
    )

    # Return the figure for saving
    return fig, mae, mape

"""# Save train_data_alchemist and test_data_alchemist as CSV files
train_data_alchemist.to_csv('train_data_alchemist.csv', index=True)
test_data_alchemist.to_csv('test_data_alchemist.csv', index=True)

from google.colab import files

# Download the CSV files
files.download('train_data_alchemist.csv')
files.download('test_data_alchemist.csv')

# Classical Modelling

# Define the model with adjustments for faster performance.
ARIMA_model_alchemist = auto_arima(y=train_data_alchemist, X=None,
                   start_p=0, d=None, start_q=0, max_p=2, max_d=0, max_q=2, #max_p = 5 crashed #max_p=3
                   start_P=0, D=None, start_Q=0, max_P=2, max_D=0, max_Q=2,  #max_Q =5 crashed #max_P=3
                   max_order=10,
                   m=52, seasonal=True,
                   stationary=True, information_criterion='aic',
                   alpha=0.05, test='kpss', seasonal_test='ocsb', #where is the stationarity test?
                   stepwise=True, n_jobs=1, maxiter=30,  #do I want a stepwise or parallel strategy? the latter is exhaustive, the former is quicker so opting for that
                   offset_test_args=None, seasonal_test_args=None,
                   suppress_warnings=True, error_action='trace', trace=False,
                   random=False, random_state=None, n_fits=5,
                   return_valid_fits=False, out_of_sample_size=0, scoring='mse',
                   with_intercept='auto', sarimax_kwargs=None)

# Print model results.
ARIMA_model_alchemist.summary()

"""6.4 minutes to finish running"""

# Print model results.
ARIMA_model_alchemist.summary()

# Use the model to forecast the next 32 time steps.
n_periods = 32  # Number of periods to forecast

predictions_ARIMA_alchemist = ARIMA_model_alchemist.predict(n_periods=n_periods, return_conf_int=True, alpha=0.05)

forecast_ARIMA_alchemist = predictions_ARIMA_alchemist[0]  # This is the predicted values (Pandas Series)
forecast_int_ARIMA_alchemist = predictions_ARIMA_alchemist[1]  # This is the confidence intervals (NumPy array)
forecast_int_df_ARIMA_alchemist = pd.DataFrame(forecast_int_ARIMA_alchemist, columns=["lower", "upper"], index=forecast_ARIMA_alchemist.index)


# Optionally extract the fitted values if needed
fitted_values_ARIMA_alchemist = pd.Series(ARIMA_model_alchemist.predict_in_sample(), index=train_data_alchemist.index)

# Call the function with fitted values to get the figure
fig, mae, mape = plot_prediction(series_train=train_data_alchemist, series_test=test_data_alchemist,
                                 forecast=forecast_ARIMA_alchemist, forecast_int=forecast_int_df_ARIMA_alchemist,
                                 fitted_values=fitted_values_ARIMA_alchemist, title='Auto Arima Forecast Alchemist')

# Print the metrics
print("MAE:", mae)
print("MAPE:", mape)

fig.show()

# Save the figure as an HTML file and download it
output_path = "Auto Arima Forecast Alchemist.html"
pio.write_html(fig, file=output_path, auto_open=False)

# Download the HTML file locally
files.download(output_path)

"""Note - SARIMAX(1, 0, 0)x(2, 0, [1], 52)"""


# Initialize a list to hold model results
model_results_list = []

def log_model_results(model_number, model_name, model_results, mae, mape, book, dataset): #rmse,
    # Create a dictionary to store the model information
    model_info = {
        'Model Number': model_number,
        'Book': book,
        'Dataset': dataset,
        'Model Name': model_name,
        'Model Config': model_results,
        'MAE': mae,
        'MAPE': mape#,
        #'RMSE': rmse  # Add RMSE as a new metric
    }
    model_results_list.append(model_info)

# Define your ARIMA model configurations and run the model
model_name_1 = "SARIMAX(1, 0, 0)x(1, 0, 0, 52)"
model_results_1 = {
    "Log Likelihood": -3858.150,
    "AIC": 7724.299,
    "BIC": 7742.037,
    "Sample": "01-07-2012 - 12-09-2023",
    "Intercept": 92.7516,
    "AR.L1": 0.6713,
    "AR.S.L52": 0.5023,
    "Sigma2": 1.149e+04,
    "Ljung-Box (L1)": 0.00,
    "Jarque-Bera (JB)": 0.00,
    "Heteroskedasticity (H)": 2.75,
    "Skewness": 0.35,
    "Kurtosis": 8.33,
    "Time Taken": "6.4 minutes"
}

# Call the function with fitted values to get the figure and metrics
fig_1, mae_1, mape_1 = plot_prediction(
    series_train=train_data_alchemist,
    series_test=test_data_alchemist,
    forecast=forecast_ARIMA_alchemist,
    forecast_int=forecast_int_df_ARIMA_alchemist,
    fitted_values=fitted_values_ARIMA_alchemist,
    title='Auto Arima Forecast Alchemist'
)

# Log the first model results with Model Number, Book, and Dataset
log_model_results(1, model_name_1, model_results_1, mae_1, mape_1, "The Alchemist", "After 2012-01-01 weekly")

model_results_list

# Flatten the Model Config dictionary into the main DataFrame
model_results_df = pd.json_normalize(model_results_list)

# Display the DataFrame
display(model_results_df)

# Define your second ARIMA model configurations and run the model
model_name_2 = "SARIMAX(1, 0, 0)x(2, 0, [1], 52)"
model_results_2 = {
    "Log Likelihood": -4498.761,
    "AIC": 9009.522,
    "BIC": 9036.129,
    "Sample": "01-07-2012 - 12-09-2023",
    "Intercept": 189.9622,
    "AR.L1": 0.8270,
    "AR.S.L52": -0.0733,
    "AR.S.L104": 0.2990,
    "MA.S.L52": 0.4073,
    "Sigma2": 1.072e+05,
    "Ljung-Box (L1)": 0.20,
    "Jarque-Bera (JB)": 0.00,
    "Heteroskedasticity (H)": 4.59,
    "Skewness": 0.88,
    "Kurtosis": 16.61,
    "Time Taken": "16 minutes"
}


model_results_list

# Flatten the Model Config dictionary into the main DataFrame
model_results_df = pd.json_normalize(model_results_list)

# Display the DataFrame
display(model_results_df)

# Specify the file path where you want to save the data and model
save_path_alchemist = 'arima_model_alchemist.pkl'

# Save everything in a dictionary
save_data_alchemist = {
    'model': ARIMA_model_alchemist,  # Save the trained ARIMA model
    'train_data_alchemist': train_data_alchemist,  # Training data
    'test_data_alchemist': test_data_alchemist,  # Test data
    'forecast': forecast_ARIMA_alchemist,  # Forecast data
    'forecast_int': forecast_int_df_ARIMA_alchemist,  # Forecast confidence intervals
    'aic_alchemist': model_results_1['AIC'],  # AIC value
    'bic_alchemist': model_results_1['BIC'],  # BIC value
}

# Open a file in write-binary mode and save the dictionary
with open(save_path_alchemist, 'wb') as f:
    pickle.dump(save_data_alchemist, f)

print(f"Model and data saved to {save_path_alchemist}")

# Download the file to your local machine
files.download(save_path_alchemist)


# Specify the file path for saving the list
save_path_list = 'model_results_list.pkl'

# Save the list using pickle
with open(save_path_list, 'wb') as f:
    pickle.dump(model_results_list, f)

# Download the saved file to your local machine
files.download(save_path_list)

# Specify the file path for saving the dataframe
save_path_df = 'model_results_df.csv'

# Save the dataframe as a CSV file
model_results_df.to_csv(save_path_df, index=False)

# Download the saved file to your local machine
files.download(save_path_df)

## LSTM Alchemist

train_data_alchemist, test_data_alchemist
display(train_data_alchemist.head(5)), display(train_data_alchemist.head(5))

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

# Display the results
display(train_data_alchemist.head())
display(test_data_alchemist.head())

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

print("X_train")
display(X_train.shape)
print("n Y_train")
display(Y_train.shape)
print("X_test")
display(X_test.shape)
print("Y_test")
display(Y_test.shape)

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

#tuner = None

#shutil.rmtree('./LSTM')  # This will remove the existing tuner directory

#import os
#os.remove("./<your_project_name>/tuner0.json")
#os.remove("./<your_project_name>/oracle.json")

# can use overwrite=True in RandomSearch()

"""8mins for 3 trials, 5mins for 1 trial"""

tuner = RandomSearch(
    tuned_model,
    objective='mse',
    max_trials=3,  # 2 trials down from 10, took too long
    executions_per_trial=1,  # reduced from 32 to 1 was taking too long
    project_name='LSTM Alchemist'
)

tuner.search(
    x=X_train,
    y=Y_train,
    epochs=50,  # down from 100 to 50 for speed
    batch_size=32,  # Experiment with different sizes
#    validation_data=(X_val, Y_val)
)

hp = tuner.get_best_hyperparameters()[0]
hp.values

# Retrieve the best model from the tuner
best_model = tuner.get_best_models(num_models=1)[0]
print(best_model.summary())

# Make predictions
train_best_predict = best_model.predict(X_train)  # Predictions on training data
test_best_predict = best_model.predict(X_test)    # Predictions on test data

# Check the shapes of the predictions
print("Train predictions shape:", train_best_predict.shape)  # Should be (611, 32)
print("Test predictions shape:", test_best_predict.shape)    # Should be (1, 32)

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
train_best_predict_flat = train_best_predict.flatten()[:len(train_data_alchemist)]
print(train_best_predict_flat.shape)
# Convert to a Pandas Series for indexing
fitted_values_series = pd.Series(train_best_predict_flat, index=train_data_alchemist.index[:len(train_best_predict_flat)])
fitted_values_series

# Call the function with fitted values to get the figure
fig, mae, mape = plot_prediction(series_train=train_data_alchemist['Volume'], series_test=test_data_alchemist['Volume'],
                                 forecast=test_best_predict_flat, fitted_values=fitted_values_series,
                                 title='LSTM Forecast Alchemist')
rmse_alchemist_lstm = np.sqrt(mean_squared_error(Y_best_test, test_best_predict_flat))

# Print the metrics
print("MAE:", mae)
print("MAPE:", mape)
print(f'Root Mean Squared Error (RMSE): {rmse_alchemist_lstm:.2f}')
fig.show()

# Save the model first
#best_model.save('lstm_model.h5')

# Then download it to your local machine
#files.download('lstm_model.h5')

"""For reasons unknown to me, the below does not work even though it clearly works in demo 8.1.2

# Make predictions on training and test datasets
train_best_predict = best_model.predict(X_train)
test_best_predict = best_model.predict(X_test)

# Output shapes of each layer in the best model
print("Model Layer Outputs:")
for layer in best_model.layers:
    print(layer.output_shape)

# Make predictions for the last available input sequence
last_input_sequence_predictions = best_model.predict(last_available_input_sequence)

# Reshape and inverse transform predictions
last_input_predictions = last_input_sequence_predictions.reshape(-1, 1)
last_input_predictions_inverse = scaler.inverse_transform(last_input_predictions)

# Output predictions and comparison with the original scale
print("Predictions for the Last Input Sequence (inverse scaled):", last_input_predictions_inverse)
print("Last Output Sequence (inverse scaled):", scaler.inverse_transform(output_sequences[-1].reshape(-1, 1)))
"""

# Assuming fig_7, mae_7, and mape_7 are obtained from a function that plots predictions and returns these metrics
fig_7, mae_7, mape_7 = plot_prediction(
    series_train=train_data_alchemist['Volume'],
    series_test=test_data_alchemist['Volume'],
    forecast=test_best_predict_flat,
    forecast_int=None,
    fitted_values=fitted_values_series,
    title='LSTM Forecast Alchemist - Model 7'
)

# Step 1: Define the new model results for Model Number 7
LSTM_alchemist_model = {
    'Hyperparameters': {
        'lookback': 41,
        'input_unit': 108,
        'n_layers': 1,
        'lstm_0_units': 68,
        'layer_2_neurons': 92,
        'Dropout_rate': 0.30,
        'lstm_1_units': 20
    },
    'Total Parameters': 157888,  # Include the total parameters from model summary
    'Trainable Parameters': 157888,
    'Non-trainable Parameters': 0,
    'Duration': '8mins'
}

# Step 2: Log the new model results
model_number = len(model_results_list) + 1  # Set model number dynamically
model_name = 'LSTM Model (on its own)'
mae_alchemist = mae_7  # MAE for the Alchemist model
mape_alchemist = mape_7  # MAPE for the Alchemist model
rmse_alchemist = rmse_alchemist_lstm  # RMSE for the Alchemist model
book_alchemist = 'The Alchemist'  # Book name for the Alchemist model
dataset_alchemist = 'After 2012-01-01 weekly'  # Dataset for Alchemist model

# Step 3: Log the seventh model results with RMSE included
log_model_results(model_number, model_name, LSTM_alchemist_model, mae_alchemist, mape_alchemist, book_alchemist, dataset_alchemist) #rmse_alchemist

model_results_list

# Flatten the Model Config dictionary into the main DataFrame
model_results_df = pd.json_normalize(model_results_list)

# Display the DataFrame
display(model_results_df)



model_results_list

# Flatten the Model Config dictionary into the main DataFrame
model_results_df= pd.json_normalize(model_results_list)

# Display the DataFrame
display(model_results_df)

"""## Saving results"""

# Save the DataFrame as a pickle file
model_results_df.to_pickle('model_results_df.pkl')

# Save the list as a pickle file
with open('model_results_list.pkl', 'wb') as f:
    pickle.dump(model_results_list, f)

# Display a message to confirm saving
print("Files have been saved successfully!")

# Download the files to your local machine
files.download('model_results_df.pkl')
files.download('model_results_list.pkl')

# Hybrid model

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

"""SARIMAX(1, 0, 0)x(2, 0, [1], 52) from auto arima above

Duration 44 seconds to run
"""

"""### section break"""

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



"""This model has a lower MAE and MAPE so performs better, visually more appealing too however the confidence interval is much worse. This is the model I will opt for as optimal for this hybrid approach as it seeems to generalise better"""

# Define the file name where you want to save the model
model_filename = 'sarima_alchemist_model_hybrid.pkl'

# Save the fitted model using pickle
with open(model_filename, 'wb') as model_file:
    pickle.dump(sarima_alchemist, model_file)

print(f"Model saved as {model_filename}")

# Download the model file
files.download(model_filename)

# Define the file name where you want to save the model

# Save the fitted model using pickle
with open(model_filename, 'wb') as model_file:
    pickle.dump(sarima_alchemist, model_file)

print(f"Model saved as {model_filename}")

# Download the model file
files.download(model_filename)

uploaded = files.upload()

# Correct filename to load the pickle file
with open('model_results_list_uptolstm.pkl', 'rb') as file:  # Use the correct filename extension
    model_results_list = pickle.load(file)

model_results_list

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

"""
# Modify the model logging function to handle future updates
def log_model_results(model_number, model_name, model_results, mae, mape, book_name, dataset):
    # Find the index of the model if it already exists (for future updates like adding LSTM)
    existing_model = next((model for model in model_results_list if model['Model Number'] == model_number), None)

    # If model exists, update it. Otherwise, create a new entry
    if existing_model:
        existing_model.update({
            'Model Config': model_results,
            'MAE': mae,
            'MAPE': mape
        })
    else:
        model_results_list.append({
            'Model Number': model_number,
            'Book': book_name,
            'Dataset': dataset,
            'Model Name': model_name,
            'Model Config': model_results,
            'MAE': mae,
            'MAPE': mape
        })
"""

# Flatten the Model Config dictionary into the main DataFrame
model_results_df = pd.json_normalize(model_results_list)

# Display the DataFrame
display(model_results_df)



# Flatten the Model Config dictionary into the main DataFrame
model_results_df = pd.json_normalize(model_results_list)

# Display the DataFrame
display(model_results_df)

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


# Save the DataFrame as a pickle file
model_results_df.to_pickle('model_results_df.pkl')

# Save the list as a pickle file
with open('model_results_list.pkl', 'wb') as f:
    pickle.dump(model_results_list, f)

# Display a message to confirm saving
print("Files have been saved successfully!")

# Download the files to your local machine
files.download('model_results_df.pkl')
files.download('model_results_list.pkl')

"""## continued"""

# Plot the histogram of the residuals.
plt.hist(sarima_alchemist.resid[1:], bins=20)

# Plot the ACF of the residuals.
smgraphics.tsa.plot_acf(sarima_alchemist.resid[1:]);


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

# Forecast for the next 32 periods (reuse from code block 1)
n_periods = 32  # Forecast horizon

result_sarima_df_alchemist.to_csv('result_sarima_df_alchemist.csv', index=False)

# If you're using Google Colab or Jupyter Notebook, you can download the files like this:
from google.colab import files

# Download the files to your local machine
files.download('result_sarima_df_alchemist.csv')

# The Alchemist DataFrame
# Save the DataFrames as CSV files
train_data_alchemist.to_csv('train_data_alchemist.csv', index=False)
test_data_alchemist.to_csv('test_data_alchemist.csv', index=False)


# If you're using Google Colab or Jupyter Notebook, you can download the files like this:
from google.colab import files

# Download the files to your local machine
files.download('train_data_alchemist.csv')
files.download('test_data_alchemist.csv')

print(train_data_alchemist.shape)  # For the training DataFrame
print(test_data_alchemist.shape)    # For the test DataFrame

#print("Fitted Values:\n", sarima.fittedvalues.head()) #cant remember what this was supposed to be
#print("Residuals:\n", residuals_train.head()) #cant remember what this was supposed to be need to check older versions from the 4th Oct

train_data_alchemist['SARIMA Residuals'].head()

train_data_alchemist

"""Apply a hybrid model of SARIMA and LSTM in sequential combination wherein the residuals from SARIMA will be forecasted using LSTM. The final prediction will be the sum of the predictions from SARIMA and LSTM. The LSTM will be trained on the residuals obtained during the training of the SARIMA model. **The forecast horizon will be the final 32 weeks**. Use KerasTuner to get the best model. Plot the results. Display the MAE and MAPE, and comment on the results.

# Save train_data_alchemist and test_data_alchemist as CSV files
train_data_alchemist.to_csv('train_data_alchemist.csv', index=True)
test_data_alchemist.to_csv('test_data_alchemist.csv', index=True)

# Download the CSV files
files.download('train_data_alchemist.csv')
files.download('test_data_alchemist.csv')

# Upload the CSV files back to Colab
uploaded = files.upload()

# Load the CSV files into DataFrames
train_data_alchemist = pd.read_csv('train_data_alchemist.csv', index_col='End Date', parse_dates=True)
test_data_alchemist = pd.read_csv('test_data_alchemist.csv', index_col='End Date', parse_dates=True)

# Convert to Series if they contain only one column
#train_data_alchemist = train_data_alchemist.squeeze()
#test_data_alchemist = test_data_alchemist.squeeze()
"""

result_sarima_df_alchemist

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


""" defined elsewhere

lookback = 12
forecast = 32

"""

uploaded = files.upload()

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


combined_sequences

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

Y_best_test = Y_test.copy()

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

tuner = None

shutil.rmtree('./LSTM_SARIMA_residuals_alchemist')  # This will remove the existing tuner directory

"""LSTM on residuals, hybrid approach

will need to load csv for data, load results df and list, make sure to run the function "log_model_results" as keep forgetting it, also dont change that function, instead just reassing that specific row/modify
"""

tuner = RandomSearch(
        tuned_model,
        objective='mse',
        max_trials=3,
        executions_per_trial=1,
        project_name='LSTM_SARIMA_residuals_alchemist'
    )

"""Duration: 6mins"""

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

# Forecast with ARIMA for the test period (32 steps).
#sarima_forecast = sarima.predict(n_periods=32)

# Forecast for the next 32 periods after the last date in the training set
#last_index = train_data_alchemist.index[-1]  # Get the last index of the training data

# Ensure the forecast starts right after the last training date
#sarima_alchemist_forecast = sarima_alchemist.predict(start=last_index + pd.Timedelta(weeks=1),
                                   #end=last_index + pd.Timedelta(weeks=32))

#display(sarima_forecast.head())
#print(sarima_forecast.shape)

test_best_predict

# Reshape test_best_predict to 1D array
#test_best_predict_flat = test_best_predict.flatten()  # Shape will now be (32,) - done above

# Combine the SARIMA predictions and the LSTM predicted residuals for the final forecast
final_forecast = predicted_mean_SARIMA_alchemist.values + test_best_predict_flat

# Calculate the mean squared error on the test set.
mse = mean_squared_error(result_sarima_df_alchemist['SARIMA Test_Forecast'][-32:], final_forecast)
print(f"Mean Squared Error on Test Data: {mse}")

#ignore below output for mse as it was using the wrong data, not going to re run this as not actually using mse as a metric to compare models, instead focusing on mae and mape

"""We can now compute the mean squared error between the test set and the final forecast values"""

print(pd.DataFrame({'Original Data': test_data_alchemist["Volume"],'Hybrid SARIMA + LSTM': final_forecast, 'SARIMA': predicted_mean_SARIMA_alchemist, 'LSTM': test_best_predict_flat})) #'LSTM':xgb_residuals_pred, #ARIMA???

final_forecast

"""think the slice length here is incorrect, may need to re run it"""

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

"""on next run do this as the code is more consistent
fig, mae, mape = plot_prediction(
    series_train=result_sarima_df_alchemist['Volume'][:slice_length],  # Explicitly using result_sarima_df_alchemist for consistency
    series_test=result_sarima_df_alchemist['SARIMA Test_Forecast'][-32:],  # Use SARIMA Test forecast
    forecast=final_forecast,  # Final combined forecast
    forecast_int=None,  # Set to None since we're ignoring confidence intervals
    title='Hybrid Sequential - Alchemist forecast - SARIMA(1, 0, 0)x(1, 0, 0, 52) then LSTM trained on residuals'
)

At some point need to come back and instead of plotting train and test separately just plot as one and call it actual data
"""

model_results_df

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

"""forgot to update mae and mape"""

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

"""## SARIMA LSTM sequential alchemist"""


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

Y_best_test = Y_test.copy()

#tuner = None


tuner = RandomSearch(
        tuned_model,
        objective='mse',
        max_trials=3,
        executions_per_trial=1,
        project_name='LSTM_SARIMA_residuals_alchemist'
    )

"""Duration: 5mins"""

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

# Save the best model to the recommended Keras format
best_model.save('best_lstm_sarima_residuals_alchemist.keras')

# Download the model file to your local machine
files.download('best_lstm_sarima_residuals_alchemist.keras')

# Inverse transform the predictions
train_best_predict = .inverse_transform(train_best_predict)
test_best_predict = .inverse_transform(test_best_predict)
# If your Y_test was generated with forecast=32, it will contain 32 values for each input
#Y_best_test = scaler.inverse_transform([Y_best_test]).flatten()
Y_best_test = .inverse_transform(Y_test.reshape(-1, 1)).flatten()  # Reshape to 2D for inverse scaling

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

# Reshape test_best_predict to 1D array
#test_best_predict_flat = test_best_predict.flatten()  # Shape will now be (32,) - done above

# Combine the SARIMA predictions and the LSTM predicted residuals for the final forecast
final_forecast = .values + test_best_predict_flat

# Calculate the mean squared error on the test set.
mse = mean_squared_error(["Volume"], final_forecast)
print(f"Mean Squared Error on Test Data: {mse}")

print(pd.DataFrame({'Original Data': ["Volume"],'Hybrid SARIMA + LSTM': final_forecast, 'SARIMA': , 'LSTM': test_best_predict_flat})) #'LSTM':xgb_residuals_pred, #ARIMA???

# Call the function with the final forecast and without fitted values
slice_length = 655 - 32

# Manually create a figure
fig = go.Figure()

# Add the training data trace
fig.add_trace(go.Scatter(
    x=['Volume'][:slice_length].index,
    y=['Volume'][:slice_length],
    mode='lines', name='Train / Actual', line=dict(color='blue')))

# Add the test data (actual values) trace
fig.add_trace(go.Scatter(
    x=result_sarima_df_alchemist['SARIMA Test_Forecast'][-32:].index,
    y=result_sarima_df_alchemist['SARIMA Test_Forecast'][-32:],
    mode='lines', name='Test / Actual', line=dict(color='black')))

# Add the forecast data trace
fig.add_trace(go.Scatter(
    x=result_sarima_df_alchemist.index[-32:],
    y=final_forecast,
    mode='lines', name='Forecast', line=dict(color='red')))

# Update layout with titles and labels
fig.update_layout(
    title={
        'text': 'Hybrid Sequential - Alchemist forecast<br>'
                'SARIMA(1, 1, 0)x(2, 0, [1], 52)<br>'
                'followed by LSTM trained on residuals',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Date",
    yaxis_title="Volume",
    legend=dict(font=dict(size=12)),
    template='plotly_white',
    width=900,
    height=600
)

# Print the metrics
print("MAE:", mae)
print("MAPE:", mape)

fig.show()

model_results_df

lstm_hybrid_hyperparameters_8 = {
    'lookback': 6,
    'input_unit': 20,
    'n_layers': 1,
    'lstm_0_units': 108,
    'layer_2_neurons': 84,
    'Dropout_rate': 0.0,
    'lstm_1_units': 20
}

# The index in model_results_list for Model Number 8 (row 7 in zero-indexed DataFrame)
model_number = 8

# Locate the model in model_results_list with the correct model number
for model in model_results_list:
    if model['Model Number'] == model_number:
        # Update the model name to reflect the hybrid approach
        model['Model Name'] = 'SARIMA(1, 1, 0)x(2, 0, [1], 52) + LSTM on Residuals'

        # Add the LSTM portion parameters under the Model Config
        model['Model Config']['LSTM'] = {
            'Hyperparameters': lstm_hybrid_hyperparameters_8,
            'Total Parameters': 125056,  # LSTM parameters
            'Trainable Parameters': 125056,
            'Non-trainable Parameters': 0,
            'Duration': '5 mins (LSTM)'  # Duration for the LSTM portion
        }

# Optional: Save back to the DataFrame if you're working with one
model_results_df = pd.json_normalize(model_results_list)
display(model_results_df)

"""forgot to update mae and mape"""

# The index in model_results_list for Model Number 8 (row 7 in zero-indexed DataFrame)
model_number = 8

# Locate the model in model_results_list with the correct model number
for model in model_results_list:
    if model['Model Number'] == model_number:
        # Update the MAE and MAPE values
        model['MAE'] = 79.21246973052621
        model['MAPE'] = 0.03610101461322015

# Optional: Save back to the DataFrame if you're working with one
model_results_df = pd.json_normalize(model_results_list)
display(model_results_df)

"""## Saving results"""

model_results_list

# Save the list as a pickle file
with open('model_results_list.pkl', 'wb') as f:
    pickle.dump(model_results_list, f)

# Download the file
files.download('model_results_list.pkl')

# Save the DataFrame as a pickle file (optional)
model_results_df.to_pickle('model_results_df.pkl')

# Download the pickle file
files.download('model_results_df.pkl')

"""unfortunately due to my poor variable naming, getting the lstm results is a little tricker X_train and y_train so having to rerun the lstm section which was done above already to complete this"""
