# Classify Wines based on their physiochemical properties

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Reading Dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url,delimiter=';')

print(df.head())

print(df.describe())

# Checks if there exists any null value
print(df.isnull().sum())

print(df.columns)

# Splitting the dataset into features and target
x = df.drop(columns=['quality'])
y = df['quality']

# Splitting the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Creating and training the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print()
print("Accuracy = ",accuracy)

print()
print("Classification Report =")
print(class_report)

print()
print("Confusion Matrix =")
print(conf_matrix)

# Visualization
plt.figure(figsize = (20,10))
plot_tree(clf, feature_names = x.columns, class_names = [str(c) for c in sorted(y.unique())], filled = True)
plt.show()
