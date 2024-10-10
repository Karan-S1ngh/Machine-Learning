# Decision Tree Simple Example


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {'A':[1,1,2,2], 'B':[2,1,2,3], 'C':[1,2,2,1], 'Y':['No','No','Yes','Yes']}
df = pd.DataFrame(data)

# Feature and Target
x = df[['A','B','C']]
y = df['Y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy = {accuracy}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print(class_report) 
