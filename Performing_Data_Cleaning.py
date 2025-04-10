import pandas as pd

# Here we will read the data from files
dt_to_train = pd.read_csv(r'adult.data', header=None)
dt_to_test = pd.read_csv(r'adult.test', header=None, skiprows=1)

# Let's define column names
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Here we will assign column names
dt_to_train.columns = columns
dt_to_test.columns = columns


dt_to_train.replace(' ?', pd.NA, inplace=True)
dt_to_test.replace(' ?', pd.NA, inplace=True)

# Here is a function to check missing values in adult.data
def values_miss_train(dt):
    cnt = dt.isnull().sum()
    tot = cnt.sum()
    print("Here are the total values which were found to be missing for adult.data :", tot)


# Here is a function to check missing values in adult.test
def values_miss_test(dt):
    cnt = dt.isnull().sum()
    tot = cnt.sum()
    print("Here are the total values which were found to be missing for adult.test :", tot)


# Let's look at missing values in training data
values_miss_train(dt_to_train)


# Let's look at missing values in testing data
values_miss_test(dt_to_test)

# All rows with missing values are removed
cleaned_adult_data = dt_to_train.dropna()
cleaned_adult_test = dt_to_test.dropna()

# Cleaned data to be stored into new files
cleaned_adult_data.to_csv("cleaned_adult_data.csv", index=False)
cleaned_adult_test.to_csv("cleaned_adult_test.csv", index=False)

# Displaying upon successful cleaning
print("\nWe have successfully cleaned the adult.data. \nCleaned data is saved as cleaned_adult_data.csv file.")
print("\nWe have successfully cleaned the adult.test. \nCleaned data is saved as cleaned_adult_test.csv file.")
