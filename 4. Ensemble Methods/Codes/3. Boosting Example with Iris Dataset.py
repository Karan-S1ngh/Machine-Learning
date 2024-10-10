# Boosting (Using Gradient Boosting Classifier)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()
x, y = iris.data, iris.target

# Split Dataset into Training and Testing Sets
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Initialise Gradient Boosting Model
boosting_model = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1, random_state = 42)

# Train the Model
boosting_model.fit(x_train, y_train)

# Make Predictions
y_pred = boosting_model.predict(x_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Boosting Model Accuracy : {accuracy : 0.2f}")
