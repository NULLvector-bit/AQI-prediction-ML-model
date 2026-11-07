import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from features import make_features, historical_extras
from physics import clock_and_physics


def model_inputs(df):
    xgb_features = make_features(df, extended=True)
    lgb_features = make_features(df, extended=False)
    clock, physics = clock_and_physics(df)
    lgb_features = pd.concat(
        [lgb_features, historical_extras(df), clock, physics], axis=1
    )
    return xgb_features, lgb_features


xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=6,
    subsample=0.8,
    objective="reg:squarederror",
    random_state=67,
    n_jobs=-1,
    tree_method="hist",
)
lgb = LGBMRegressor(
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
    n_jobs=-1,
    verbosity=-1,
)
