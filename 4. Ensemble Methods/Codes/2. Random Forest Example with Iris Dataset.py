# Random Forest


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()
x, y = iris.data, iris.target

# Split Dataset into Training and Testing Sets
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Initialize the Random Forest
random_forest_model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Train the Random Forest Model
random_forest_model.fit(x_train, y_train)

# Make predictions
y_pred = random_forest_model.predict(x_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy : {accuracy}")
