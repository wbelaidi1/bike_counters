## import data
#!pip install jours_feries_france

#from jours_feries_france import JoursFeries

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")

_target_column_name = "log_bike_count"

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    
    X["morning"] = X["hour"].isin([7, 8, 9])
    X["afternoon"] = X["hour"].isin([16, 17, 18])
    X["spring"] = X["month"].isin([3,4,5])
    X["summer"] = X["month"].isin([6,7,8])
    X["automn"] = X["month"].isin([9,10,11])
    X["winter"] = X["month"].isin([12,1,2])   

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

def _merge_external_data(X):
    df_ext = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv", parse_dates=["date"])
    df_ext = df_ext.fillna(0)
    df_ext["date"] = df_ext["date"].astype('datetime64[us]')
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", "rr3", "etat_sol", "ff"]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def get_train_data(path="/kaggle/input/msdb-2024/train.parquet"):
    data = pd.read_parquet(path)
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

params = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 5,
    "learning_rate": 0.02,
    "loss": "squared_error",
}

X, y = get_train_data()
X = _merge_external_data(X)

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)
date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]
numerical_cols = ["t", "rr3", "ff"]

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

#regressor = ensemble.GradientBoostingRegressor(**params)
regressor = LinearRegression()
#regressor = linear_model.Lasso(alpha=0.05)

pipe = make_pipeline(date_encoder, preprocessor, regressor)
pipe.fit(X, y)

from sklearn.metrics import mean_squared_error

X_test = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")

X_test = _merge_external_data(X_test)

y_pred = pipe.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("/kaggle/working/submission.csv", index=False)

print(results.head())