import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('Machine Learning/Datasets/Titanic-Dataset.csv')

data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

data = pd.get_dummies(data,columns=['Sex','Embarked'],drop_first=True)

x = data.drop('Survived',axis=1)
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter = 200)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy = {accuracy}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print(class_report)
