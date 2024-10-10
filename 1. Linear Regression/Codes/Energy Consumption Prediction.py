# predict future energy consumption from energy usage


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example historical energy consumption data
# Assume monthly data in kWh
months = np.arange(1, 13).reshape(-1, 1)  # 12 months
energy_consumption = np.array([300, 320, 310, 330, 340, 350, 360, 370, 380, 390, 400, 410])

# Create and train the model
model = LinearRegression()
model.fit(months, energy_consumption)

# Predict energy consumption for the next 6 months
future_months = np.arange(13, 19).reshape(-1, 1)  # Months 13 to 18
predicted_consumption = model.predict(future_months)

print(f"Predicted energy consumption for the next 6 months: {predicted_consumption}")

# Visualize the results
plt.figure(figsize=(10, 5))
plt.scatter(months, energy_consumption, color='blue', label='Historical Energy Consumption')
plt.plot(months, model.predict(months), color='red', linestyle='--', label='Trend Line')
plt.scatter(future_months, predicted_consumption, color='green', marker='x', label='Predicted Energy Consumption')
plt.plot(future_months, predicted_consumption, color='green', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Forecasting')
plt.legend()
plt.grid(True)
plt.show()
