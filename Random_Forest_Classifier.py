import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

# Loading the dataset
dt_to_train = pd.read_csv("cleaned_adult_data.csv")
data_eval = pd.read_csv("cleaned_adult_test.csv")

#Random Forest Classifier
def rnd_class(x__train, y__train):
    mdl_m = RandomForestClassifier(n_estimators=150, random_state=10)
    mdl_m.fit(x__train, y__train)
    return mdl_m


# Performing data preprocessing
dt_to_train = dt_to_train.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data_eval = data_eval.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Now we'll let go of rows with missing values
dt_to_train.dropna(inplace=True)
data_eval.dropna(inplace=True)

# Let's encode variables
enc = OneHotEncoder(handle_unknown='ignore')
X_train = enc.fit_transform(dt_to_train.select_dtypes(include=['object']))
y_train = dt_to_train['income'].map({'<=50K': 0, '>50K': 1})

# Confirming that data is encoded with the same encoder
X_eval = enc.transform(data_eval.select_dtypes(include=['object']))
y_eval = data_eval['income'].map({'<=50K.': 0, '>50K.': 1})

model_classfr = rnd_class(X_train, y_train)
# Evaluate the model
predictions = model_classfr.predict(X_eval)
accuracy = accuracy_score(y_eval, predictions)
cl_err_rate = (1 - accuracy) * 100
print("This is the classification error rate: {:.2f}%".format(cl_err_rate))



