# Train and Save the model

from utils import label_encode, change_feature, drop_corr_features

import numpy as np
import pandas as pd


from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

import joblib


# get our datasets
df_train = pd.read_csv(
    "./data/bank_marketing_train.csv")
df_test = pd.read_csv(
    "./data/bank_marketing_test.csv")

# check our dataset
# df_train.shape, df_test.shape
# df_train.columns, df_test.columns


# remove duplicates values
# df_train.drop_duplicates(inplace=True)
# df_test.drop_duplicates(inplace=True)

# change the columns names
df_train.rename(columns={'emp.var.rate': 'emp_var_rate',
                         'cons.price.idx': 'cons_price_idx', 'cons.conf.idx': 'cons_conf_idx', 'nr.employed': 'nr_employed'}, inplace=True)
df_test.rename(columns={'emp.var.rate': 'emp_var_rate', 'cons.price.idx': 'cons_price_idx',
               'cons.conf.idx': 'cons_conf_idx', 'nr.employed': 'nr_employed'}, inplace=True)


# drop corr columns
df_train = drop_corr_features(df_train)
df_test = drop_corr_features(df_test)

# change the categorical_value to integer
df_train = change_feature(df_train)
df_test = change_feature(df_test)


# check if we did mistake when changing categorical value to integer
# df_train.isna().sum(), df_test.isna().sum()


# lets init randomundersampler
rus = RandomUnderSampler(random_state=2)

# sepate target variable from our dataset
X_train, y_train = df_train.drop("y", axis=1), df_train['y']
# check shape of our dataset
# X_train.shape, y_train.value_counts()


# change categorical target variable to binary
y_train = y_train.map(label_encode)
# check value of our newly formed target variable
# y_train.value_counts()


# First vectorize our data
dv = DictVectorizer(sparse=False)
# change train data to dict
X_train_dict = X_train.to_dict("records")
# vectorize
X_train = dv.fit_transform(X_train_dict)


# now undersample our data
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
# print(X_train_rus.shape)
# check value and shape
# y_train_rus.value_counts(), X_train_rus.shape

# check our test dataset
# df_test.shape

# split our test datset
X_test, y_test = df_test.drop("y", axis=1), df_test['y']

# check value of our test target variable
# y_test.value_counts()

# now we change our test dataset to vectorizer
X_test_dict = X_test.to_dict("records")
X_test = dv.transform(X_test_dict)

# y_test.value_counts()

# change categorical test target variable to binary
y_test = y_test.map(label_encode)
y_test.value_counts(), y_train_rus.value_counts()

# X_test.shape


# This is our model parameters : {'catboost__learning_rate': 0.1, 'catboost__max_depth': 3, 'catboost__n_estimators': 100}

params = {'eta': 0.1,
          'max_depth': 3,
          'n_estimators': 100, }

pipe_train = Pipeline([('scaler', StandardScaler()), ('catboost', CatBoostClassifier(
    silent=True, random_state=10, task_type="GPU", devices='0:1', **params))])
pipe_train.fit(X_train_rus, y_train_rus)


y_pred = pipe_train.predict(X_test)
y_pred_proba = pipe_train.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))

print(roc_auc_score(y_test, y_pred_proba))

# we got recall of 0.93 and precision of 0.42, which is quite good.
print(classification_report(y_test, y_pred, digits=5))

# save vectorizer and model
joblib.dump(dv, "./dictvectorizer.joblib")
joblib.dump(pipe_train, "./catboost.model")
