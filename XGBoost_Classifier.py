import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

dt_to_train = pd.read_csv("cleaned_adult_data.csv")
data_eval = pd.read_csv("cleaned_adult_test.csv")
dt_to_train = dt_to_train.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data_eval = data_eval.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
dt_to_train.dropna(inplace=True)
data_eval.dropna(inplace=True)

# Let's try encoding variables
enc = OneHotEncoder(handle_unknown='ignore')
X_train = enc.fit_transform(dt_to_train.select_dtypes(include=['object']))
y_train = dt_to_train['income'].map({'<=50K': 0, '>50K': 1})
# Confirming that data is encoded with the same encoder
X_eval = enc.transform(data_eval.select_dtypes(include=['object']))
y_eval = data_eval['income'].map({'<=50K.': 0, '>50K.': 1})

# Define parameter grid for grid search
gridd = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
}

# Initialize XGBoost classifier
xgb = XGBClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb, param_grid=gridd, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the evaluation dataset
predictions = best_model.predict(X_eval)
accuracy = accuracy_score(y_eval, predictions)
classification_error_rate = (1 - accuracy) * 100

#print("Best Parameters:", best_params)
print("Classification Error Rate:", classification_error_rate,"%")

