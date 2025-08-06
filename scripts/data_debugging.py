from zenml.client import Client
import pandas as pd

# --- Load first artifact (merged_data) ---
artifact = Client().get_artifact_version("d39747c6-0ce4-4797-96d3-c31601bc1cd0")
data = artifact.load()

# Normalize column names
data.columns = data.columns.str.strip()

# Reset index so "End Date" becomes a column if it's the index
if isinstance(data.index, pd.DatetimeIndex):
    data = data.reset_index()  # "End Date" becomes a column
    data.rename(columns={'End Date': 'date'}, inplace=True)
else:
    data.rename(columns={'End Date': 'date'}, inplace=True)

# Filter for ISBN and date
data = data[data['ISBN'] == 9780722532935]
data = data[data['date'] >= '2023-12-16']

# Sort ascending by date
data = data.sort_values('date')

# Select relevant columns and rename
volume_merged = data[['date', 'Volume']].rename(columns={'Volume': 'Volume_merged_data'})

print("\n--- Volume from merged_data artifact ---")
print(volume_merged)

# --- Load second artifact (modelling_data) ---
artifact = Client().get_artifact_version("43d0efe9-ad58-4c55-8666-f6d1d28b43df")
# old ba905751-eb2b-42fe-9681-5d8d5fba1a18
data2 = artifact.load()
data2.columns = data2.columns.str.strip()

# Filter for date >= 2023-12-16
data2 = data2[data2['date'] >= '2023-12-16']

# --- Load third artifact (forecast_comparison) ---
artifact = Client().get_artifact_version("277a3bc0-e7cd-4cf6-82d7-330a76c59e17")
# old c2600e22-2140-4c1d-bfb3-3b195fa925ff
data3 = artifact.load()
data3.columns = data3.columns.str.strip()

# Filter for date >= 2023-12-16
data3 = data3[data3['date'] >= '2023-12-16']

volume_modelling = data2[['date', 'volume']].rename(columns={'volume': 'volume_modelling_data'})
volume_forecast = data3[['date', 'actual_volume']].rename(columns={'actual_volume': 'actual_volume_forecast_comparison'})

# Merge second and third artifacts
combined_df = volume_modelling.merge(volume_forecast, on='date', how='outer')

print("\n--- Volume from modelling_data & forecast_comparison artifacts ---")
print(combined_df)
