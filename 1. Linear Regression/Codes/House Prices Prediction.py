# predicting house prices from house sizes


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([100,150,200,250,300]).reshape(-1,1)
# House sizes in sq. feet

y = np.array([250000,350000,450000,550000,650000])
# prices

model = LinearRegression()
model.fit(x, y)

new_house_sizes = np.array([175,225]).reshape(-1,1)
predicted_prices = model.predict(new_house_sizes)

print("Intercept = ",model.intercept_)
print("Slope =",model.coef_[0])

plt.scatter(x, y, color='blue', label='data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.scatter(new_house_sizes, predicted_prices, color='green', label='Predictions', marker='x')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.title('Simple Example for Linear Regression')
plt.legend()
plt.show()
