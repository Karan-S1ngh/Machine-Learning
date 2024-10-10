# Adaboost


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()
x, y = iris.data, iris.target

# Split Dataset into Training and Testing Sets
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Initialize the Base Model
base_model = DecisionTreeClassifier(max_depth = 1)

# Initialize AdaBoost with Base Model
adaboost_model = AdaBoostClassifier(base_model, n_estimators = 50, learning_rate = 1.0, random_state = 42)

# Train the AdaBoost Model
adaboost_model.fit(x_train, y_train)

# Make Predictions
y_pred = adaboost_model.predict(x_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Model Accuracy : {accuracy}")
