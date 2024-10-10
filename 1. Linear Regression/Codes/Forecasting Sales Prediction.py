# predicting future data from historical sales data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example historical sales data
# Assume monthly sales data
months = np.arange(1, 13).reshape(-1, 1)  # 12 months
sales = np.array([200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310])

# Create and train the model
model = LinearRegression()
model.fit(months, sales)

# Predict sales for the next 6 months
future_months = np.arange(13, 19).reshape(-1, 1)  # Months 13 to 18
predicted_sales = model.predict(future_months)

print(f"Predicted sales for the next 6 months: {predicted_sales}")

# Visualize the results
plt.figure(figsize=(10, 5))
plt.scatter(months, sales, color='blue', label='Historical Sales')
plt.plot(months, model.predict(months), color='red', linestyle='--', label='Trend Line')
plt.scatter(future_months, predicted_sales, color='green', marker='x', label='Predicted Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecasting')
plt.legend()
plt.grid(True)
plt.show()
