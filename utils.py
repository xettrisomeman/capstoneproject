def label_encode(x): return 0 if x == "no" else 1


def drop_corr_features(df):
    df = df.drop(['euribor3m', 'nr_employed'], axis=1)
    return df


poutcome_change = {
    "nonexistent": -1,
    "failure": 0,
    "success": 1
}

loan_change = {
    "no": 0,
    "yes": 1,
    "unknown": -1
}

housing_change = {
    "no": 0,
    "yes": 1,
    "unknown": -1
}

default_change = {
    "yes": 1,
    "unknown": -1,
    "no": 0
}


def change_feature(df):
    df.default = df.default.map(default_change)
    df.housing = df.housing.map(housing_change)
    df.loan = df.loan.map(loan_change)
    df.poutcome = df.poutcome.map(poutcome_change)
    df.pdays = df.pdays.replace({999: -1})
    return df
