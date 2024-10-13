import pandas as pd

data = pd.read_csv("../Datasets/winequality-red.csv")

missing_values = data.isnull().sum()
print(missing_values)

# Adding three new columns
data['TotalAcidity'] = data['fixed acidity'] + data['volatile acidity']
data['Density_pH'] = data['density'] * data['pH']
data['Sulphates_Chlorides'] = data['sulphates'] / data['chlorides']

print(data.head())




# Using Scalar
from sklearn.preprocessing import StandardScaler

numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'TotalAcidity', 'Density_pH', 'Sulphates_Chlorides']

scaler = StandardScaler()

data[numerical_features] = scaler.fit_transform(data[numerical_features])

print(data.head())

x = data.drop(columns = ['quality'])
y = data['quality']

print(x.shape, y.shape)


### Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (8,6))
sns.countplot(x = 'quality', data = data, palette = 'viridis')
plt.title("Distribution of wine quality")
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()



## Feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier(n_estimators = 100, random_state = 0)
model.fit(x,y)

importance = model.feature_importances_
indices = np.argsort (importance)[::-1]


### Visualisation
plt.figure(figsize = (12, 8))
plt.bar(range(x.shape[1]), importance[indices], align = 'center')
plt.xticks(range(x.shape[1]), [numerical_features[i] for i in indices], rotation = 90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


## Performance analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Confusion Matrix =")
print(confusion_matrix(y_test, y_pred))

print("Classification Report =")
print(classification_report(y_test, y_pred))

print("Accuracy =", accuracy_score(y_test, y_pred))
# More Accuracy from Decision Tree (here Random Forest is used)




# Using PCA
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(data[numerical_features])

pca_data = pca.transform(data[numerical_features])

pca_df = pd.DataFrame(data = pca_data, columns = [f'PC{i+1}' for i in range(pca_data.shape[1])])

explained_varaince = pca.explained_variance_ratio_
print(explained_varaince)

### Visualisation
plt.figure(figsize = (10, 6))
plt.plot(range(1, len(explained_varaince) + 1), explained_varaince, marker = 'o', linestyle = '--')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.show()

