# Gradient Boosting


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()
x, y = iris.data, iris.target

# Split Dataset into Training and Testing Sets
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Initialize Gradient Boost with Base Model
gradient_boosting_model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42)

# Train the AdaBoost Model
gradient_boosting_model.fit(x_train, y_train)

# Make Predictions
y_pred = gradient_boosting_model.predict(x_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Model Accuracy : {accuracy}")
