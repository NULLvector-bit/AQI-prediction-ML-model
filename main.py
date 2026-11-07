"""AQI predictor - pollutant lag/rolling study winner.

Requires train.csv, test.csv and sample_submission.csv beside this script.
Set VALIDATION_MODE=True to reproduce five city-held-out validation folds.
Set VALIDATION_MODE=False to create the Kaggle CSV.
No target-derived feature is used.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

VALIDATION_MODE = False
XGB_WEIGHT = 0.40
LGB_WEIGHT = 0.60

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


def extras(df):
    tt = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(
        pd.to_datetime(df["Time"], format="%I:%M %p").dt.hour, unit="h"
    )
    order = np.lexsort((tt.to_numpy(dtype="datetime64[ns]"), df.City.to_numpy()))
    s = df.iloc[order]
    g = s.groupby("City", sort=False)
    result = {z: {} for z in ["CO", "PM", "GASES", "DYNAMICS"]}

    def add(group, name, series):
        result[group][name] = series.astype(np.float32)

    def roll_mean(c, w):
        return g[c].transform(lambda t: t.rolling(w, min_periods=1).mean())

    def roll_stat(c, w, kind):
        return g[c].transform(lambda t: getattr(t.rolling(w, min_periods=2), kind)())

    def lag(c, w):
        return g[c].shift(w)

    # Additional CO history beyond the successful baseline, with finer time resolution.
    c = "CO"
    for w in [4, 6, 12, 16, 36, 72]:
        add("CO", f"{c}_mean{w}", roll_mean(c, w))
    for w in [2, 4, 6, 12, 16, 36, 48, 72]:
        add("CO", f"{c}_lag{w}", lag(c, w))
    for w in [3, 6, 12, 48]:
        add("CO", f"{c}_std{w}", roll_stat(c, w, "std"))
    for w in [8, 24, 48]:
        for stat in ["max", "min"]:
            add("CO", f"{c}_{stat}{w}", roll_stat(c, w, stat))
    # PM daily averaging, robust stats, fine timings; 24-hour means already baseline
    for c in ["PM2.5", "PM10"]:
        for w in [6, 12, 16, 20, 22, 26, 28, 32, 36, 72]:
            add("PM", f"{c}_mean{w}", roll_mean(c, w))
        for w in [2, 4, 6, 12, 16, 36, 48]:
            add("PM", f"{c}_lag{w}", lag(c, w))
        for w in [3, 6, 12, 48]:
            add("PM", f"{c}_std{w}", roll_stat(c, w, "std"))
        for w in [8, 24, 48]:
            for stat in ["max", "min"]:
                add("PM", f"{c}_{stat}{w}", roll_stat(c, w, stat))
    # Temporal evidence from the other pollutants missing from the original baseline.
    for c in ["NO", "NO2", "NOx", "NH3", "SO2", "O3", "Benzene", "Toluene"]:
        # NO2 SO2 O3 already have 8/24 means, NO2 already has lag
        for w in [8, 24, 48]:
            if c in ["NO2", "SO2", "O3"] and w in [8, 24]:
                continue
            if c == "NO2" and w == 48:
                pass
            add("GASES", f"{c}_mean{w}", roll_mean(c, w))
        for w in [1, 3, 8, 24]:
            if c == "NO2":
                continue
            add("GASES", f"{c}_lag{w}", lag(c, w))
    # Differences and trends are derived only from measurements, never AQI labels.
    for c in ["PM2.5", "PM10", "CO", "O3"]:
        a = roll_mean(c, 3)
        b = roll_mean(c, 24)
        add("DYNAMICS", f"{c}_short_long_diff", a - b)
        add("DYNAMICS", f"{c}_short_long_ratio", a / (b.abs() + 1))
        for w in [8, 24]:
            add(
                "DYNAMICS",
                f"{c}_range{w}",
                roll_stat(c, w, "max") - roll_stat(c, w, "min"),
            )
    for group in result:
        result[group] = pd.DataFrame(result[group], index=df.index)
        assert not result[group].columns.duplicated().any()
    return result


# XGBoost retains the 78 successful features unchanged.
# LightGBM receives the 34 simple features plus 140 additional history features.
def prepare_features(df):
    X_xgb = make_features(df, extended=True)
    X_lgb = make_features(df, extended=False)
    additional = extras(df)
    X_lgb = pd.concat(
        [
            X_lgb,
            additional["CO"],
            additional["PM"],
            additional["GASES"],
            additional["DYNAMICS"],
        ],
        axis=1,
    )
    assert X_xgb.columns.is_unique and X_lgb.columns.is_unique
    return X_xgb, X_lgb


ROOT = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(ROOT, "train.csv"))
test_data = pd.read_csv(os.path.join(ROOT, "test.csv"))
print("Loaded CSVs", len(data), len(test_data), flush=True)
X, X_lgb = prepare_features(data)
print("Training features", X.shape, X_lgb.shape, flush=True)
X_submission, X_lgb_submission = prepare_features(test_data)
print("Test features", X_submission.shape, X_lgb_submission.shape, flush=True)

# Align training and test columns exactly, especially location one-hot columns.
assert X.shape[1] == 78 and X_lgb.shape[1] == 174
X_submission = X_submission.reindex(columns=X.columns, fill_value=0)
X_lgb_submission = X_lgb_submission.reindex(columns=X_lgb.columns, fill_value=0)
assert X_submission.columns.equals(X.columns)
assert X_lgb_submission.columns.equals(X_lgb.columns)
assert len(X_submission) == len(test_data)
assert len(X_lgb_submission) == len(test_data)

y = data["AQI"]
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
    fold_scores = []
    sum_sq_error = 0.0
    count = 0
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups=data["City"]), start=1
    ):
        val_idx = val_idx[complete_mask[val_idx]]
        final_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        second_model.fit(X_lgb.iloc[train_idx], y.iloc[train_idx])
        predicted_aqi = XGB_WEIGHT * final_model.predict(
            X.iloc[val_idx]
        ) + LGB_WEIGHT * second_model.predict(X_lgb.iloc[val_idx])
        squared_error = (predicted_aqi - y.iloc[val_idx].to_numpy()) ** 2
        fold_rmse = np.sqrt(squared_error.mean())
        fold_scores.append(fold_rmse)
        sum_sq_error += squared_error.sum()
        count += len(val_idx)
        print(f"Fold {fold} RMSE: {fold_rmse:.5f}")
    print(f"Average fold RMSE: {np.mean(fold_scores):.5f}")
    print(f"Pooled RMSE: {np.sqrt(sum_sq_error / count):.5f}")
else:
    print("Fitting XGBoost", flush=True)
    final_model.fit(X, y)
    print("Fitting LightGBM", flush=True)
    second_model.fit(X_lgb, y)
    print("Making test predictions", flush=True)
    predicted_aqi = XGB_WEIGHT * final_model.predict(
        X_submission
    ) + LGB_WEIGHT * second_model.predict(X_lgb_submission)
    submission = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    assert len(submission) == len(predicted_aqi) == len(test_data)
    assert list(submission.columns) == ["Index", "AQI"]
    assert np.isfinite(predicted_aqi).all()
    submission["AQI"] = predicted_aqi
    path = os.path.join(ROOT, "submission_lag_optimized.csv")
    submission.to_csv(path, index=False, float_format="%.10f")
    print(f"Submission created: {path}, {len(submission)} rows")
