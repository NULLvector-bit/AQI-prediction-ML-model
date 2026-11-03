import pandas as pd
from xgboost import XGBRegressor

# 1. Load datasets
data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
pollutants = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
]


def make_features(df):
    features = df[pollutants].copy()

    features["hour"] = pd.to_datetime(
        df["Time"], format="%I:%M %p", errors="coerce"
    ).dt.hour

    features["month"] = pd.to_datetime(
        df["Date"], format="%m/%d/%Y", errors="coerce"
    ).dt.month

    regions = pd.get_dummies(df["location"], prefix="region", dtype=int)

    return pd.concat([features, regions], axis=1)


X = make_features(data)
y = data["AQI"]
X_submission = make_features(test_data)
X_submission = X_submission.reindex(columns=X.columns, fill_value=0)
final_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    objective="reg:squarederror",
    random_state=67,
    n_jobs=-1,
)
final_model.fit(X, y)
predicted_aqi = final_model.predict(X_submission)
submission = pd.read_csv("sample_submission.csv")
assert len(submission) == len(predicted_aqi), "Row count mismatch!"
submission["AQI"] = predicted_aqi
submission.to_csv("submission.csv", index=False)
print("Submission created successfully!")
