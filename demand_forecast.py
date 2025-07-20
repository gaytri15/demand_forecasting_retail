import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('demand_data.csv')

# Step 2: Show the first few rows
print("Dataset:\n", data.head())

# Step 3: Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Step 4: Plot the demand over time
plt.figure(figsize=(8, 5))
plt.plot(data['Date'], data['Units_Sold'], marker='o', color='teal')
plt.title("Daily Demand for Shampoo")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

# Step 5.1: Convert Date to ordinal (numeric format)
data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

# Step 5.2: Prepare features and target
X = data[['Date_Ordinal']]
y = data['Units_Sold']

# Step 5.3: Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 5.4: Predict the next 7 days
last_date = data['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

predictions = model.predict(future_ordinals)

# Step 5.5: Show predictions
for date, pred in zip(future_dates, predictions):
    print(f"Predicted demand for {date.strftime('%Y-%m-%d')}: {int(pred)} units")

# Optional: Plot actual + predicted demand
plt.figure(figsize=(8, 5))
plt.plot(data['Date'], data['Units_Sold'], label='Actual Demand', marker='o')
plt.plot(future_dates, predictions, label='Forecasted Demand', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.title('Demand Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
