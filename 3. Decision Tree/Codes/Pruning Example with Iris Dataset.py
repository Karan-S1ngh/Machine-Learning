# Iris Dataset With and Without Pruning Example


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score

# importing iris dataset 
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# Without Pruning (Before Pruning)

tree_clf = DecisionTreeClassifier(random_state = 42)
tree_clf.fit(x_train, y_train)

y_pred_train = tree_clf.predict(x_train)
y_pred_test = tree_clf.predict(x_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train Accuracy : {train_accuracy}")

test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy : {test_accuracy}")

# Visualisation
import matplotlib.pyplot as plt
plt.figure(figsize = (20,10))
plot_tree(tree_clf, filled = True, feature_names = iris.feature_names, class_names = iris.target_names)
plt.show()


# With Pruning (After Pruning)

pruned_tree_clf = DecisionTreeClassifier(max_depth = 3, random_state = 42)
pruned_tree_clf.fit(x_train, y_train)

y_pred_train_pruned = pruned_tree_clf.predict(x_train)
y_pred_test_pruned = pruned_tree_clf.predict(x_test)

train_accuracy_pruned = accuracy_score(y_train, y_pred_train_pruned)
print(f"Train Accuracy : {train_accuracy_pruned}")

test_accuracy_pruned = accuracy_score(y_test, y_pred_test_pruned)
print(f"Test Accuracy : {test_accuracy_pruned}")

# Visualisation
import matplotlib.pyplot as plt
plt.figure(figsize = (20,10))
plot_tree(pruned_tree_clf, filled = True, feature_names = iris.feature_names, class_names = iris.target_names)
plt.show()
