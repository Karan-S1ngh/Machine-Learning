# predicting marks from no of hrs studied


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Data in dictionary form
data = {
    'Hours Studied': [2, 3.5, 5, 6.5, 8, 9.5],
    'Marks': [50, 60, 75, 80, 90, 95]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Features and target
x = df[['Hours Studied']]  # Features
y = df['Marks']            # Target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Predicting for new data points
new_hours = pd.DataFrame({'Hours Studied': [4, 7, 10,]})
predicted_scores = model.predict(new_hours)

print(f"Predicted Scores for {new_hours['Hours Studied'].values} hours studied: {predicted_scores}")

# Visualizing the results
plt.scatter(x, y, color='blue', label='Actual Scores')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.scatter(new_hours, predicted_scores, color='green', marker='x', label='Predicted Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Marks Prediction Based on Hours Studied')
plt.legend()
plt.grid(True)
plt.show()
