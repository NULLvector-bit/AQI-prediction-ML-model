import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

VALIDATION_MODE = True

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


def make_features(df, extended=True):
    features = df[pollutants].copy()
    features["hour"] = pd.to_datetime(
        df["Time"], format="%I:%M %p", errors="coerce"
    ).dt.hour
    features["month"] = pd.to_datetime(
        df["Date"], format="%m/%d/%Y", errors="coerce"
    ).dt.month
    regions = pd.get_dummies(df["location"], prefix="region", dtype=int)
    features = pd.concat([features, regions], axis=1).astype(np.float32)

    timestamps = pd.to_datetime(
        df["Date"], format="%m/%d/%Y", errors="raise"
    ) + pd.to_timedelta(features["hour"], unit="h")
    order = np.lexsort(
        (timestamps.to_numpy(dtype="datetime64[ns]"), df["City"].to_numpy())
    )
    chronological = df.iloc[order]
    grouped = chronological.groupby("City", sort=False)

    windows = [3, 8, 24, 48] if extended else [8, 24]
    for pollutant in ["PM2.5", "PM10", "CO", "NO2", "O3", "SO2"]:
        for window in windows:
            features[f"{pollutant}_mean{window}"] = (
                grouped[pollutant]
                .transform(lambda series: series.rolling(window, min_periods=1).mean())
                .astype(np.float32)
            )

    if extended:
        for pollutant in ["PM2.5", "PM10", "CO", "NO2"]:
            for window in [8, 24]:
                features[f"{pollutant}_std{window}"] = (
                    grouped[pollutant]
                    .transform(
                        lambda series: series.rolling(window, min_periods=2).std()
                    )
                    .astype(np.float32)
                )
            for lag in [1, 3, 8, 24]:
                features[f"{pollutant}_lag{lag}"] = (
                    grouped[pollutant].shift(lag).astype(np.float32)
                )

        features["weekday"] = timestamps.dt.dayofweek.to_numpy(dtype="float32")
        features["dayofyear"] = timestamps.dt.dayofyear.to_numpy(dtype="float32")
        features["year"] = timestamps.dt.year.to_numpy(dtype="float32")
        features["dayofmonth"] = timestamps.dt.day.to_numpy(dtype="float32")
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    return features


X = make_features(data, extended=True)
X_small = make_features(data, extended=False)
y = data["AQI"]
X_submission = make_features(test_data, extended=True)
X_small_submission = make_features(test_data, extended=False)
X_submission = X_submission.reindex(columns=X.columns, fill_value=0)
X_small_submission = X_small_submission.reindex(columns=X_small.columns, fill_value=0)

assert X.shape[1] == 78 and X_small.shape[1] == 34
assert X_submission.columns.equals(X.columns)
assert X_small_submission.columns.equals(X_small.columns)
assert len(X_submission) == len(test_data)

final_model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=6,
    subsample=0.8,
    objective="reg:squarederror",
    random_state=67,
    n_jobs=4,
    tree_method="hist",
)

second_model = LGBMRegressor(
    n_estimators=500,
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=2,
    random_state=67,
    n_jobs=4,
    verbosity=-1,
)

if VALIDATION_MODE:
    complete_mask = data[pollutants].notna().all(axis=1).to_numpy()
    splitter = GroupKFold(n_splits=5)
    rmse_scores = []
    out_of_fold = np.full(len(data), np.nan)

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups=data["City"]), start=1
    ):
        val_idx = val_idx[complete_mask[val_idx]]
        final_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        second_model.fit(X_small.iloc[train_idx], y.iloc[train_idx])

        predicted_aqi = 0.6 * final_model.predict(
            X.iloc[val_idx]
        ) + 0.4 * second_model.predict(X_small.iloc[val_idx])
        out_of_fold[val_idx] = predicted_aqi
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], predicted_aqi))
        rmse_scores.append(rmse)
        print(f"Fold {fold} RMSE: {rmse:.5f}")

    evaluated = np.isfinite(out_of_fold)
    pooled_rmse = np.sqrt(mean_squared_error(y.iloc[evaluated], out_of_fold[evaluated]))
    print(f"Average RMSE: {np.mean(rmse_scores):.5f}")
    print(f"Pooled RMSE: {pooled_rmse:.5f}")

else:
    final_model.fit(X, y)
    second_model.fit(X_small, y)

    predicted_aqi = 0.6 * final_model.predict(
        X_submission
    ) + 0.4 * second_model.predict(X_small_submission)
    submission = pd.read_csv("sample_submission.csv")
    assert len(submission) == len(predicted_aqi)
    assert np.isfinite(predicted_aqi).all()
    submission["AQI"] = predicted_aqi
    submission.to_csv("submission_rolling_blend.csv", index=False)
    print("Submission created: submission_rolling_blend.csv")
