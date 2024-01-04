#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv ("C:/Users/arung/Downloads/DOGE-USD.csv")


# In[4]:


df


# In[5]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[6]:


df


# In[ ]:





# In[9]:


import matplotlib.pyplot as plt

# Plotting Dogecoin-USD Closing Prices
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], color='blue')
plt.title('Dogecoin-USD Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.grid(True)
plt.show()


# In[ ]:





# In[11]:


# Handling missing values by linear interpolation
df['Close'].interpolate(method='linear', inplace=True)


# In[12]:


# Decomposing the time series after handling missing values
result = seasonal_decompose(df['Close'], model='multiplicative', period=1)

# Plotting the decomposition
plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(df.index, result.observed, label='Original', color='blue')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(df.index, result.trend, label='Trend', color='red')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(df.index, result.seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(df.index, result.resid, label='Residuals', color='purple')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# In[18]:


result1 = seasonal_decompose(df['Close'], model='multiplicative', period=7)   # Weekly seasonality
result2 = seasonal_decompose(df['Close'], model='multiplicative', period=14)  # biweekly seasonality
result3 = seasonal_decompose(df['Close'], model='multiplicative', period=30)  # Monthly seasonality
 


# In[21]:


# Seasonal decomposition with different periods
result1 = seasonal_decompose(df['Close'], model='multiplicative', period=7)   # Weekly seasonality
result2 = seasonal_decompose(df['Close'], model='multiplicative', period=14)  # biweekly seasonality
result3 = seasonal_decompose(df['Close'], model='multiplicative', period=30)  # Monthly seasonality

# Print the results or attributes of each decomposition to observe any patterns
print("Weekly Seasonality:")
print(result1.seasonal.head())  # Display the first few rows of the seasonal component
print("biweekly Seasonality:")
print(result2.seasonal.head())  # Display the first few rows of the seasonal component
print("monthly Seasonality:")
print(result3.seasonal.head())  # Display the first few rows of the seasonal component


# In[25]:


import matplotlib.pyplot as plt

# Extract seasonal components for plotting
seasonal_weekly = result1.seasonal
seasonal_biweekly = result2.seasonal
seasonal_monthly = result3.seasonal

# Plotting the seasonal components
plt.figure(figsize=(12, 6))

plt.plot(seasonal_weekly.index, seasonal_weekly, label='Weekly Seasonality')
plt.plot(seasonal_biweekly.index, seasonal_biweekly, label='Biweekly Seasonality')
plt.plot(seasonal_monthly.index, seasonal_monthly, label='Monthly Seasonality')

plt.title('Seasonal Components for Different Periodicities')
plt.xlabel('Date')
plt.ylabel('Seasonal Component')
plt.legend()
plt.grid(True)
plt.show()


# In[26]:


# Assuming 'df' contains your Dogecoin-USD data (replace 'df' with your actual dataframe)
from sklearn.model_selection import train_test_split

# Define the feature(s) used for modeling (e.g., 'Close' prices)
features = ['Close']

# Splitting the data into training and testing sets (e.g., 80% train, 20% test)
train_data, test_data = train_test_split(df[features], test_size=0.2, shuffle=False)

# Check the lengths of the train and test sets
print("Train set length:", len(train_data))
print("Test set length:", len(test_data))


# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np

# Assuming 'train_data' and 'test_data' contain 'Close' prices
time_steps = 10  # Define the number of time steps for GRU input (adjust as needed)
num_features = 1  # Number of features (in this case, 'Close' price)

# Function to prepare data for GRU model with defined time steps
def prepare_data(data, time_steps):
    data_X, data_Y = [], []
    for i in range(len(data) - time_steps):
        data_X.append(data[i:(i + time_steps), 0])
        data_Y.append(data[i + time_steps, 0])
    return np.array(data_X), np.array(data_Y)

# Reshape the data for GRU input
train_values = train_data.values.reshape(-1, 1)  # Reshape for single feature
test_values = test_data.values.reshape(-1, 1)

# Prepare the training data for GRU
train_X, train_Y = prepare_data(train_values, time_steps)

# Reshape the data for GRU input (add additional dimension for features)
train_X = np.reshape(train_X, (train_X.shape[0], time_steps, num_features))

# Define and compile the GRU model
model_gru = Sequential()
model_gru.add(GRU(units=50, return_sequences=True, input_shape=(time_steps, num_features)))
model_gru.add(GRU(units=50))
model_gru.add(Dense(units=1))

model_gru.compile(optimizer='adam', loss='mean_squared_error')

# Train the GRU model
model_gru.fit(train_X, train_Y, epochs=10, batch_size=32)  # Adjust epochs and batch_size as needed

# Prepare test data for forecasting
test_X, test_Y = prepare_data(test_values, time_steps)
test_X = np.reshape(test_X, (test_X.shape[0], time_steps, num_features))

# Forecasting using the trained GRU model
gru_forecast = model_gru.predict(test_X)


# In[31]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_Y, gru_forecast)
print('Mean Squared Error (MSE):', mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test_Y, gru_forecast)
print('Mean Absolute Error (MAE):', mae)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.plot(test_Y, label='Actual')
plt.plot(gru_forecast, label='Predicted', color='red')
plt.title('GRU Model: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




