import numpy as np
import pandas as pd

from config import AQI_BREAKPOINTS


def clock_and_physics(df):
    timestamp = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="raise")
    timestamp += pd.to_timedelta(
        pd.to_datetime(df["Time"], format="%I:%M %p", errors="raise").dt.hour, unit="h"
    )
    sorted_df = df.assign(_t=timestamp).sort_values(["City", "_t"])
    city_groups = sorted_df.groupby("City", sort=False)

    previous = city_groups["_t"].shift(1)
    gap = ((sorted_df["_t"] - previous).dt.total_seconds() / 3600).fillna(9999.0)
    clock = {
        "gap_since_prev_hours": gap.clip(0, 9999).astype("float32"),
        "has_gap_over_1h": (gap > 1).astype("float32"),
    }
    previous_24 = city_groups["_t"].shift(23)
    clock["elapsed_24_obs_hours"] = (
        ((sorted_df["_t"] - previous_24).dt.total_seconds() / 3600)
        .fillna(9999.0)
        .clip(0, 9999)
        .astype("float32")
    )

    rolling_values = {}
    for pollutant in ["PM2.5", "PM10", "CO", "O3", "SO2", "NO2", "NH3"]:
        windows = [8, 24] if pollutant in ["CO", "O3"] else [24]
        for hours in windows:
            means = np.full(len(df), np.nan, dtype="float32")
            maxima = np.full(len(df), np.nan, dtype="float32")
            counts = np.zeros(len(df), dtype="float32")
            for _, part in city_groups:
                series = part.set_index("_t")[pollutant]
                rolling = series.rolling(f"{hours}h", min_periods=1)
                idx = part.index.to_numpy()
                means[idx] = rolling.mean().to_numpy(dtype="float32")
                maxima[idx] = rolling.max().to_numpy(dtype="float32")
                counts[idx] = rolling.count().to_numpy(dtype="float32")
            rolling_values[f"{pollutant}_clockmean{hours}"] = means
            rolling_values[f"{pollutant}_clockmax{hours}"] = maxima
            rolling_values[f"{pollutant}_clockcount{hours}"] = counts

    clock_frame = pd.DataFrame(
        {
            name: values.reindex(df.index).to_numpy(dtype="float32")
            for name, values in clock.items()
        },
        index=df.index,
    )
    for name, values in rolling_values.items():
        clock_frame[name] = values

    physics = {}
    subindices = []
    for pollutant, (concentrations, subindex_breaks) in AQI_BREAKPOINTS.items():
        hours = 8 if pollutant in ["CO", "O3"] else 24
        averages = rolling_values[f"{pollutant}_clockmean{hours}"]
        values = np.interp(
            np.maximum(averages, 0), concentrations, subindex_breaks
        ).astype("float32")
        physics[f"proxy_cpcb_{pollutant}"] = values
        subindices.append(values)

    indices = np.stack(subindices, axis=1)
    physics["proxy_cpcb_max_subindex"] = np.nanmax(indices, axis=1)
    physics["proxy_cpcb_top2_gap"] = (
        np.sort(indices, axis=1)[:, -1] - np.sort(indices, axis=1)[:, -2]
    )
    physics["proxy_PM25_over_PM10"] = rolling_values["PM2.5_clockmean24"] / (
        rolling_values["PM10_clockmean24"] + 5
    )
    physics_frame = pd.DataFrame(physics, index=df.index)

    for frame in [clock_frame, physics_frame]:
        for column in frame:
            values = frame[column].to_numpy(dtype="float32", copy=True)
            values[~np.isfinite(values)] = np.nan
            frame[column] = values
    return clock_frame, physics_frame
