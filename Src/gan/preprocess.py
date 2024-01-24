'''
This script preprocesses the data for the GAN. It reads the data, converts it to datetime, filters out the correct dates, annualises the TTM, rescales the data, isolates the call and put options, sorts by date, drops non-numeric columns, filters out rows with IV > 1000, normalizes the data and exports the input data.
'''

# Import packages 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read data
Gold = pd.read_excel(r"C:\Users\koens\OneDrive\Bureaublad\Research-Practicum\Data\Real Data\FUT_Option.xlsx")

# Convert to datetime
Gold['date'] = pd.to_datetime(Gold['date'])
Gold['futures_expiration_date'] = pd.to_datetime(Gold['futures_expiration_date'])
Gold['options_expiration_date'] = pd.to_datetime(Gold['options_expiration_date'], errors='coerce')

# Get correct dates and annualise
Gold = Gold[Gold['options_expiration_date'] >= '2019-10-18']
Gold['TTM'] = (Gold['options_expiration_date'] - Gold['date']).dt.days / 365.25

# Rescale
Gold['futures_close'] = Gold['futures_close']/1000000
Gold['strike'] = Gold['strike']/1000000
Gold['bid'] = Gold['bid']/1000000
Gold['ask'] = Gold['ask']/1000000
Gold['settlement'] = Gold['settlement']/1000000
Gold['vega'] = Gold['vega']/1000000

# Isolate call and put
Gold_call = Gold[Gold['call_put'] == 'C'].copy()
Gold_put = Gold[Gold['call_put'] == 'P'].copy()

# Sort by date
Gold_call.sort_values('date', inplace=True)
Gold_put.sort_values('date', inplace=True)

# Drop non-numeric columns if there are any
data = Gold_call.select_dtypes(include=[np.number])
data = data.drop(['delta', 'vega', 'gamma', 'theta'], axis=1)

# Filter out rows with IV > 1000
mask = data['iv'] > 1000
data = data[~mask]

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)

# Export input data
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.to_excel(r"C:\Users\koens\OneDrive\Bureaublad\Research-Practicum\Data\Real Data\gold_preprocessed.xlsx", index=False)

print('Preprocessing complete')