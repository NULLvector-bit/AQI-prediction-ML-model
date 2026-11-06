import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

VALIDATION_MODE = False

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
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=6,
    subsample=0.8,
    objective="reg:squarederror",
    random_state=67,
    n_jobs=-1,
)

if VALIDATION_MODE:
    from sklearn.model_selection import GroupKFold

    complete_mask = data[pollutants].notna().all(axis=1).to_numpy()
    splitter = GroupKFold(n_splits=5)

    single_scores = []
    ensemble_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups=data["City"]), start=1
    ):
        val_idx = val_idx[complete_mask[val_idx]]

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        predictions = []

        for seed in [21, 42, 67]:
            final_model.set_params(random_state=seed)
            final_model.fit(X_train, y_train)
            predictions.append(final_model.predict(X_val))

        single_rmse = np.sqrt(mean_squared_error(y_val, predictions[2]))

        predicted_aqi = np.mean(predictions, axis=0)

        ensemble_rmse = np.sqrt(mean_squared_error(y_val, predicted_aqi))

        single_scores.append(single_rmse)
        ensemble_scores.append(ensemble_rmse)

        print(f"\nFold {fold}")
        print(f"Seed 67 RMSE: {single_rmse:.5f}")
        print(f"Ensemble RMSE: {ensemble_rmse:.5f}")

    print(f"\nSeed 67 average: {np.mean(single_scores):.5f}")
    print(f"Ensemble average: {np.mean(ensemble_scores):.5f}")


else:
    seeds = [21, 42, 67]
    predictions = []

    for seed in seeds:
        final_model.set_params(random_state=seed)
        final_model.fit(X, y)

        predicted_aqi = final_model.predict(X_submission)
        predictions.append(predicted_aqi)

        print(f"Finished training seed {seed}")

    predicted_aqi = np.mean(predictions, axis=0)

    submission = pd.read_csv("sample_submission.csv")
    submission["AQI"] = predicted_aqi
    submission.to_csv("submission.csv", index=False)

    print("Submission created")
