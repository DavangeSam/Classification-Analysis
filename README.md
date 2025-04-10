# Predicting Income Levels

This project explores supervised learning techniques using the Adult Income dataset. The dataset includes demographic and employment-related attributes used to predict whether an individual's income exceeds $50K annually.

📂 Dataset Files:

    adult.data: Training dataset

    adult.test: Test dataset

    adult.names: Metadata including attribute descriptions and benchmark results



🧹Performing Data Cleaning:

Both training and test datasets may contain missing values. 

A data preprocessing pipeline was designed and implemented to handle these cases, based on best practices for handling missing data (e.g., imputation or removal).

The cleaning ensures the datasets are ready for reliable model training and evaluation.

Cleaned Adult Data:

![Cleaned Adult Data](https://github.com/DavangeSam/Predicting-Income-Levels/blob/main/Cleaned%20Adult%20Data.png?raw=true)

Cleaned Adult Test:

![Cleaned Adult Test](https://github.com/DavangeSam/Predicting-Income-Levels/blob/main/Cleaned%20Adult%20Test.png?raw=true)

🤖 Classification Modeling:

Two classification algorithms were implemented and evaluated:

    Random Forest Classifier

    XGBoost Classifier


Models were trained on the cleaned adult.data and evaluated on adult.test. 

Performance was measured using classification error rate, defined as:

Error Rate = (# of incorrect predictions) / (total number of test instances)

Random Forest Classifier Result:

![random forest](https://github.com/DavangeSam/Predicting-Income-Levels/blob/main/Random%20Forest%20Classifier%20Result.png?raw=true)

XGBoost Classifier Result:

![xgboost](https://github.com/DavangeSam/Predicting-Income-Levels/blob/main/XGBoost%20Result.png?raw=true)

