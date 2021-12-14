import streamlit as st

import pandas as pd
import joblib


from catboost import CatBoostClassifier
from sklearn.feature_extraction import DictVectorizer


from utils import drop_corr_features, change_feature

st.title("Portugese Bank Telemarketing Prediction")


st.caption(
    "Checking if a client will subscribe to term deposit or not based on data provided by the user.")
url = "https://archive.ics.uci.edu/ml/datasets/bank+marketing"
st.caption(f"Data link: {url}")

job_check = ["admin.", "blue-collar",
             "entrepreneur", "housemaid", "management", "retired", "student", "self-employed", "technician", "unemployed", "unknown"]

marital_check = ["divorced", "married", "single", "unknown"]
education_check = ["basic.4y", "basic.6y", "basic.9y", "high.school",
                   "illiterate", "professional.course", "university.degree", "unknown"]
month_check = ["mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
day_of_week_check = ["mon", "tue", "wed", "thu", "fri"]
poutcome_check = ["failure", "nonexistent", "success"]
contact_check = ["cellular", "telephone"]
default_check = ["yes", "no", "unknown"]
housing_check = ["no", "yes", "unknown"]
loan_check = ["no", "yes", "unknown"]


st.sidebar.title("User input parameters")


def create_data():
    age = st.sidebar.slider("Age",  17, 98)
    job = st.sidebar.selectbox("Jobs", job_check)
    marital = st.sidebar.selectbox("Marital", marital_check)
    education = st.sidebar.selectbox("Education", education_check)
    default = st.sidebar.selectbox("Default", default_check)
    housing = st.sidebar.selectbox("Housing", housing_check)
    loan = st.sidebar.selectbox("Loan", loan_check)
    contact = st.sidebar.selectbox("Contact", contact_check)
    month = st.sidebar.selectbox("Month", month_check)
    day_of_week = st.sidebar.selectbox("Day of Week", day_of_week_check)
    poutcome = st.sidebar.selectbox("Poutcome", poutcome_check)
    duration = st.sidebar.slider("Duration", 0, 4199)
    campaign = st.sidebar.slider("Campaign", 1, 56)
    pdays = st.sidebar.slider("Pdays", -1, 27)
    previous = st.sidebar.slider("Previous", 0, 7)
    emp_var_rate = st.sidebar.slider("Emp Var Rate", -3.4, 1.4)
    cons_price_idx = st.sidebar.slider("Cons Price Index", 92.2, 94.7)
    cons_conf_idx = st.sidebar.slider("Cons Conf Index", -50.8, -27.9)
    euribor3m = st.sidebar.slider("Euribor3m", 0.6, 5.0)
    nr_employed = st.sidebar.slider("NR Employed", 4963, 5228)

    data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx,
        "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed
    }

    df_predict = pd.DataFrame([data])

    return data, df_predict


data, df = create_data()

st.subheader("User Input Values")
st.write(df)

st.subheader("Class Labels and Their Corresponding index number")

st.write(pd.DataFrame(["No", "Yes"],  columns={"label"}))

# get model and vectorizer
dv = joblib.load("./dictvectorizer.joblib")
model = joblib.load("./catboost.model")

# change to vectors

df_predict = change_feature(df)
df_predict = drop_corr_features(df_predict)
df_predict = df_predict.to_dict("records")

X_test = dv.transform(df_predict)
y_pred = model.predict_proba(X_test)

print(y_pred)
st.subheader("Prediction:")


def decision(x):
    if x[0][0] > x[0][1]:
        return "Customer won't subscribe"
    else:
        return "Customer will subscribe"


st.write(y_pred)
st.write(decision(y_pred))


st.caption(
    "Note: The model takes time to predict and is very much skewed to class 0.")
