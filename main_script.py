!pip install holidays
!pip install xgboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import holidays

import utils
_target_column_name = "log_bike_count"

def get_data():
    # Define the starting directory (e.g., current directory)
    file_name = "train.parquet"
    start_dir = Path(".")  # Current directory

    # Search for the specific file
    parquet_file = next(start_dir.rglob(file_name), None)

    if parquet_file:
        print(f"Loading file: {parquet_file}")
        
        # Load the Parquet file into a pandas DataFrame
        df = pd.read_parquet(parquet_file)  # Use the found file path

        # y_array = df["log_bike_count"].values
        # X_df = df[["date", "counter_name"]]
        df = df[["date", "counter_name", "bike_count", "log_bike_count"]]
        return df
    else:
        print(f"File '{file_name}' not found in the directory.")


def merge_ext_data(X):
    # Define the starting directory (e.g., current directory)
    file_name = "external_data.csv"
    start_dir = Path(".")  # Current directory

    # Search for the specific file
    csv_file = next(start_dir.rglob(file_name), None)

    if csv_file:
        print(f"Loading file: {csv_file}")
        
        # Load the Parquet file into a pandas DataFrame
        df_ext = pd.read_csv(csv_file)  # Use the found file path
        df_ext = df_ext.fillna(0)
        df_ext["date"] = df_ext["date"].astype('datetime64[us]')
        X = X.copy()
        X["orig_index"] = np.arange(X.shape[0])
        X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", "etat_sol", 'rr1', 'rr12', "ff", "ht_neige" ]].sort_values("date"), on="date")
        X = X.sort_values("orig_index")
        X["rr1"] = abs(X["rr1"])
        X["rr3"] = abs(X["rr12"])
        del X["orig_index"]

        df_expanded = X.loc[X.index.repeat(3)].reset_index(drop=True)
        # Add hourly intervals to the timestamp
        df_expanded["date"] += pd.to_timedelta(df_expanded.groupby(df_expanded.index // 3).cumcount(), unit="h")
        # Sort by timestamp
        df_expanded = df_expanded.sort_values("date").reset_index(drop=True)
        return X
    else:
        print(f"File '{file_name}' not found in the directory.")


# Fonction pour définir les périodes de confinement
def set_periods(df, column_name, periods):
    df[column_name] = 0
    for start_date, end_date in periods:
        df.loc[(df['date'] >= start_date) & (df['date'] < end_date), column_name] = 1

def encode_dates(X):
    X = X.copy()  # modify a copy of X
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["weekend_x_hour"] = np.where(X["weekday"].isin([5, 6]), 'weekend_', 'weekday_') + X["hour"].astype(str)
    fr_holidays = holidays.France(years=[(2015 + i) for i in range(10)])
    X["is_red_day"] = X["date"].dt.date.isin(fr_holidays.keys()).astype(int)
    # Définir les périodes de confinement
    lockdown_periods = [
        ('2020-10-30', '2020-12-15'),
        ('2021-04-03', '2021-05-04')
    ]
    set_periods(X, 'Lockdown', lockdown_periods)

    # Définir les périodes de couvre-feu souple
    soft_curfew_periods = [
        ('2020-10-17', '2020-10-30'),
        ('2020-12-15', '2021-01-16'),
        ('2021-05-19', '2021-06-21')
        ]
    set_periods(X, 'soft-curfew', soft_curfew_periods)

    # Définir les périodes de couvre-feu strict
    hard_curfew_periods = [
        ('2021-01-16', '2021-04-03'),
        ('2021-05-04', '2021-05-19')]

    X = X.drop(columns=["date", "hour"])

    return X

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]
    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

data = get_data()
y = data["log_bike_count"].values
X = data[["date", "counter_name"]]
X = merge_ext_data(X)

date_encoder = FunctionTransformer(encode_dates)
date_cols = encode_dates(X[["date"]]).columns.tolist()

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "etat_sol"]
numerical_cols = ['ff', 'rr1', 'rr12', 't', 'ht_neige']

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn import ensemble
from sklearn.linear_model import Ridge
import xgboost as xgb


param = {
    'n_estimators': 500,  # 3 values
    'max_depth': 9,  # 3 values
    'learning_rate': 0.15,  # 2 values
    'subsample': 1.0,  # 1 values
    'colsample_bytree': 1.0,  # 1 values
    'reg_alpha': 0.1,  # 1 values
    'reg_lambda': 5  # 1 values
}

regressor = xgb.XGBRegressor(**param)

pipe = make_pipeline(date_encoder, preprocessor, regressor)

pipe.fit(X_train, y_train)

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error


X_test = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")

y_pred = pipe.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("/kaggle/working/submission.csv", index=False)