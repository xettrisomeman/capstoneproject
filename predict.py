import joblib
import pandas as pd
import uvicorn


from catboost import CatBoostClassifier
from sklearn.feature_extraction import DictVectorizer


from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


from utils import drop_corr_features, change_feature


class Campaigns(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    duration: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: int
    cons_price_idx: int
    cons_conf_idx: int
    euribor3m: int
    nr_employed: int


app = FastAPI()

# get model and vectorizer
dv = joblib.load("./dictvectorizer.joblib")
model = joblib.load("./catboost.model")


@app.post("/predict/")
def predict(user_input: Campaigns):
    # change the base model to dataframe
    df_test = pd.DataFrame([jsonable_encoder(user_input)])
    df_test = change_feature(df_test)
    df_test = drop_corr_features(df_test)

    df_test = df_test.to_dict("records")
    X_test = dv.transform(df_test)
    y_pred = model.predict(X_test)[0]

    if y_pred == 0:
        return {
            "Prediction": "Customer won't subscribe"
        }
    return {
        "Prediction": "Customer will subscribe"
    }


if __name__ == "__main__":
    uvicorn.run("predict:app", debug=True,
                host='localhost', port=5000, reload=True)
