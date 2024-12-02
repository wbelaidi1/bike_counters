# %% [code]
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error



# Function to get train data (as in original script)

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    JF = pd.to_datetime(['2021-01-01', '2021-04-05', '2021-05-01', '2021-05-08',
               '2021-05-13', '2021-05-24', '2021-07-14', '2021-08-15',
               '2021-11-01', '2021-11-11', '2021-12-25', '2020-01-01',
               '2020-04-13', '2020-05-01', '2020-05-08', '2020-05-21',
               '2020-06-01', '2020-07-14', '2020-08-15', '2020-11-01',
               '2020-11-11', '2020-12-25']).astype('datetime64[us]')
    X["ferie"] = X["date"].isin(JF)
    
    X["morning"] = X["hour"].isin([7, 8, 9])
    X["afternoon"] = X["hour"].isin([16, 17, 18])
    X["spring"] = X["month"].isin([3,4,5])
    X["summer"] = X["month"].isin([6,7,8])
    X["automn"] = X["month"].isin([9,10,11])
    X["winter"] = X["month"].isin([12,1,2])   

    # Finally we can drop the original columns from the dataframe
    return X

def get_train_data(path="/kaggle/input/msdb-2024/train.parquet"):
    data = pd.read_parquet(path)
    data = data.sort_values(["date", "counter_name"])
    #y_array = data["log_bike_count"].values
    #X_df = data.drop(["log_bike_count", "bike_count"], axis=1)
    #return X_df, y_array
    return data

def _merge_external_data(X):
    df_ext = pd.read_csv("/kaggle/input/external-data/external_data.csv", parse_dates=["date"])
    df_ext = df_ext.fillna(0)
    df_ext["date"] = df_ext["date"].astype('datetime64[us]')
    X = X.copy()
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t", "rr3", "etat_sol", 'rr1', "rr6", "ff"]].sort_values("date"), on="date"
    )
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]
    return X_train, y_train, X_valid, y_valid

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = get_train_data()
mask = (data["date"] > pd.to_datetime("2021/08/01"))
data = data[mask]
y = data["log_bike_count"].values
X = data.drop(["log_bike_count", "bike_count"], axis=1)
X_train = _merge_external_data(X)

# Temporal train-test split
#X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

# Prepare preprocessing steps
date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]
numerical_cols = ["t", "rr3", "ff", "rr1", "rr6"]

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)


params = {
    'max_depth': 6,
    'learning_rate': 0.2,
    'min_samples_split': 5,
    'n_estimators': 500
}

from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# ensemble.GradientBoostingRegressor(**params)
regressor = ensemble.GradientBoostingRegressor()

# Create pipeline
pipe = make_pipeline(date_encoder, preprocessor, regressor)

pipe.fit(X_train, y)


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

print(y_pred[0])

# print(results.head())

# Best parameters and score
# print("Best Parameters:", random_search.best_params_)
# print("Best Negative MSE Score:", random_search.best_score_)

# Optional: Evaluate on validation set
#from sklearn.metrics import mean_squared_error
#best_model = random_search.best_estimator_
#y_pred = best_model.predict(X_valid)
#rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
#print(f"Validation RMSE: {rmse}")