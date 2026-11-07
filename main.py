from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from config import VALIDATION_MODE, XGB_WEIGHT, LGB_WEIGHT, POLLUTANTS

from models import model_inputs, xgb, lgb


def main():
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    y = train["AQI"]

    x_train, l_train = model_inputs(train)
    x_test, l_test = model_inputs(test)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    l_test = l_test.reindex(columns=l_train.columns, fill_value=0)
    assert x_train.shape == (len(train), 78) and l_train.shape == (len(train), 214)
    assert x_test.shape == (len(test), 78) and l_test.shape == (len(test), 214)
    assert x_train.columns.is_unique and l_train.columns.is_unique
    assert x_test.columns.equals(x_train.columns) and l_test.columns.equals(
        l_train.columns
    )
    print("Training features:", x_train.shape, l_train.shape, flush=True)

    if VALIDATION_MODE:
        complete = train[POLLUTANTS].notna().all(axis=1).to_numpy()
        fold_scores, sum_squared, n_rows = [], 0.0, 0
        for fold, (train_idx, val_idx) in enumerate(
            GroupKFold(n_splits=5).split(x_train, y, groups=train["City"]), start=1
        ):
            val_idx = val_idx[complete[val_idx]]
            xgb.fit(x_train.iloc[train_idx], y.iloc[train_idx])
            lgb.fit(l_train.iloc[train_idx], y.iloc[train_idx])
            pred = XGB_WEIGHT * xgb.predict(
                x_train.iloc[val_idx]
            ) + LGB_WEIGHT * lgb.predict(l_train.iloc[val_idx])
            squared = (pred - y.iloc[val_idx].to_numpy()) ** 2
            score = np.sqrt(squared.mean())
            fold_scores.append(score)
            sum_squared += squared.sum()
            n_rows += len(val_idx)
            print(f"Fold {fold} RMSE: {score:.5f}")
        print(f"Mean RMSE: {np.mean(fold_scores):.5f}")
        print(f"Pooled RMSE: {np.sqrt(sum_squared / n_rows):.5f}")
    else:
        xgb.fit(x_train, y)
        lgb.fit(l_train, y)
        predictions = XGB_WEIGHT * xgb.predict(x_test) + LGB_WEIGHT * lgb.predict(
            l_test
        )
        submission = pd.read_csv(root / "sample_submission.csv")
        assert submission.columns.tolist() == ["Index", "AQI"]
        assert len(submission) == len(test) == len(predictions)
        assert submission["Index"].is_unique and np.isfinite(predictions).all()
        submission["AQI"] = predictions
        path = root / "submission.csv"
        submission.to_csv(path, index=False, float_format="%.10f")
        print("Saved:", path, flush=True)


if __name__ == "__main__":
    main()
