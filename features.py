import numpy as np
import pandas as pd

from config import pollutants


def make_features(df, extended):
    """Core pollutant and observation-based history features."""
    features = df[pollutants].copy()
    features["hour"] = pd.to_datetime(
        df["Time"], format="%I:%M %p", errors="coerce"
    ).dt.hour
    features["month"] = pd.to_datetime(
        df["Date"], format="%m/%d/%Y", errors="coerce"
    ).dt.month
    regions = pd.get_dummies(df["location"], prefix="region", dtype=int)
    features = pd.concat([features, regions], axis=1).astype(np.float32)

    timestamps = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="raise")
    timestamps += pd.to_timedelta(features["hour"], unit="h")
    order = np.lexsort(
        (timestamps.to_numpy(dtype="datetime64[ns]"), df["City"].to_numpy())
    )
    grouped = df.iloc[order].groupby("City", sort=False)

    windows = [3, 8, 24, 48] if extended else [8, 24]
    for pollutant in ["PM2.5", "PM10", "CO", "NO2", "O3", "SO2"]:
        for window in windows:
            features[f"{pollutant}_mean{window}"] = (
                grouped[pollutant]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .astype(np.float32)
            )

    if extended:
        for pollutant in ["PM2.5", "PM10", "CO", "NO2"]:
            for window in [8, 24]:
                features[f"{pollutant}_std{window}"] = (
                    grouped[pollutant]
                    .transform(lambda s: s.rolling(window, min_periods=2).std())
                    .astype(np.float32)
                )
            for lag in [1, 3, 8, 24]:
                features[f"{pollutant}_lag{lag}"] = (
                    grouped[pollutant].shift(lag).astype(np.float32)
                )

        for name, values in {
            "weekday": timestamps.dt.dayofweek,
            "dayofyear": timestamps.dt.dayofyear,
            "year": timestamps.dt.year,
            "dayofmonth": timestamps.dt.day,
        }.items():
            features[name] = values.to_numpy(dtype="float32")
        for name, period in [("hour", 24), ("month", 12)]:
            features[f"{name}_sin"] = np.sin(2 * np.pi * features[name] / period)
            features[f"{name}_cos"] = np.cos(2 * np.pi * features[name] / period)
    return features


def historical_extras(df):
    """The 140 extra CO, PM, gas and trend columns used by LightGBM."""
    timestamp = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(
        pd.to_datetime(df["Time"], format="%I:%M %p").dt.hour, unit="h"
    )
    order = np.lexsort(
        (timestamp.to_numpy(dtype="datetime64[ns]"), df["City"].to_numpy())
    )
    grouped = df.iloc[order].groupby("City", sort=False)
    groups = {name: {} for name in ["CO", "PM", "GASES", "DYNAMICS"]}

    def add(group, column, values):
        groups[group][column] = values.astype(np.float32)

    def rolling_mean(pollutant, window):
        return grouped[pollutant].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    def rolling_stat(pollutant, window, stat):
        return grouped[pollutant].transform(
            lambda s: getattr(s.rolling(window, min_periods=2), stat)()
        )

    def lagged(pollutant, lag):
        return grouped[pollutant].shift(lag)

    # CO history
    for window in [4, 6, 12, 16, 36, 72]:
        add("CO", f"CO_mean{window}", rolling_mean("CO", window))
    for lag in [2, 4, 6, 12, 16, 36, 48, 72]:
        add("CO", f"CO_lag{lag}", lagged("CO", lag))
    for window in [3, 6, 12, 48]:
        add("CO", f"CO_std{window}", rolling_stat("CO", window, "std"))
    for window in [8, 24, 48]:
        for stat in ["max", "min"]:
            add("CO", f"CO_{stat}{window}", rolling_stat("CO", window, stat))

    # Fine-grained particulate matter history
    for pollutant in ["PM2.5", "PM10"]:
        for window in [6, 12, 16, 20, 22, 26, 28, 32, 36, 72]:
            add("PM", f"{pollutant}_mean{window}", rolling_mean(pollutant, window))
        for lag in [2, 4, 6, 12, 16, 36, 48]:
            add("PM", f"{pollutant}_lag{lag}", lagged(pollutant, lag))
        for window in [3, 6, 12, 48]:
            add(
                "PM", f"{pollutant}_std{window}", rolling_stat(pollutant, window, "std")
            )
        for window in [8, 24, 48]:
            for stat in ["max", "min"]:
                add(
                    "PM",
                    f"{pollutant}_{stat}{window}",
                    rolling_stat(pollutant, window, stat),
                )

    for pollutant in ["NO", "NO2", "NOx", "NH3", "SO2", "O3", "Benzene", "Toluene"]:
        for window in [8, 24, 48]:
            if pollutant in ["NO2", "SO2", "O3"] and window in [8, 24]:
                continue
            add("GASES", f"{pollutant}_mean{window}", rolling_mean(pollutant, window))
        if pollutant != "NO2":
            for lag in [1, 3, 8, 24]:
                add("GASES", f"{pollutant}_lag{lag}", lagged(pollutant, lag))

    for pollutant in ["PM2.5", "PM10", "CO", "O3"]:
        short = rolling_mean(pollutant, 3)
        long = rolling_mean(pollutant, 24)
        add("DYNAMICS", f"{pollutant}_short_long_diff", short - long)
        add("DYNAMICS", f"{pollutant}_short_long_ratio", short / (long.abs() + 1))
        for window in [8, 24]:
            spread = rolling_stat(pollutant, window, "max") - rolling_stat(
                pollutant, window, "min"
            )
            add("DYNAMICS", f"{pollutant}_range{window}", spread)

    return pd.concat(
        [pd.DataFrame(group, index=df.index) for group in groups.values()], axis=1
    )
