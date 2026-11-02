import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("train.csv")
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
X = data[pollutants].copy()
y = data["AQI"].copy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=67)

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
test_data = pd.read_csv("test.csv")
X_submission = test_data[pollutants].copy()
predicted_aqi = final_model.predict(X_submission)
submission = pd.read_csv("sample_submission.csv")
submission["AQI"] = predicted_aqi
submission.to_csv("submission.csv", index=False)
