# Sample Code for Evaluating and Comparing Ensemble Methods


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load Dataset
iris = load_iris()
x, y = iris.data, iris.target

# Split Dataset into Training and Testing Sets
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Initialize Matrix
models = {
    'Bagging': BaggingClassifier(DecisionTreeClassifier(), n_estimators = 50, random_state = 42),
    'Random Forest': RandomForestClassifier(n_estimators = 50, random_state = 42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42),
    'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 50, learning_rate = 1.0, random_state = 42)
}

# Train, Predict and Evaluate Each Model
results = {}
for model_name, model in models.items():
    # Train the Model
    model.fit(x_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(x_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred, average = 'macro')
    recall = recall_score(y_test, y_pred, average = 'macro') 
    f1 = f1_score(y_test, y_pred, average = 'macro')
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class = 'ovo', average = 'macro')
    conf_matrix = confusion_matrix(y_test, y_pred) 
    
    # Store the results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Confusion Matrix': conf_matrix
    } 
    
# Print the Results
for model_name, metrics in results.items():
    print(f'Model: {model_name}')
    for metric_name, metric_value in metrics.items():
        if metric_name == 'Confusion Matrix':
            print(f'{metric_name}:\n {metric_value}')
        else:
            print(f'{metric_name}: {metric_value:.2f}')
    print('\n')
