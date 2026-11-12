# Air Quality Index (AQI) Prediction

A Python machine-learning project that **estimates India's Air Quality Index (AQI)** from pollutant concentrations and historical observations. It combines **XGBoost**, **LightGBM**, time-series feature engineering, and pollutant-based AQI features to learn relationships between measured air pollution and AQI.

The current pipeline works with structured CSV files. It estimates AQI for rows with pollutant readings; **it does not forecast future AQI without future pollutant measurements**.

## Model overview

1. **Read observations:** Load pollution measurements, city and region identifiers, dates, and times.
2. **Build historical features:** Calculate pollutant lags, rolling averages, standard deviations, extrema, and trends separately for each city.
3. **Add time and AQI-inspired features:** Include calendar information, elapsed observation times, and approximate pollutant subindices derived from concentration breakpoints.
4. **Train two regressors:** XGBoost uses **78 features**; LightGBM uses **214 features**.
5. **Combine their predictions:**

   ```text
   Predicted AQI = (0.10 × XGBoost) + (0.90 × LightGBM)
   ```

LightGBM assigns training rows with measurements for all 11 pollutants a sample weight of **4.0**; other rows receive **1.0**. XGBoost does not use these sample weights.

### Evaluation

Validation uses **5-fold GroupKFold by city**, keeping each validation city's observations out of that fold's training set. RMSE is calculated on validation rows containing all 11 pollutant measurements.

| Recorded local validation metric | RMSE (AQI points) |
| --- | ---: |
| Mean across five folds | 19.23312 |
| Pooled across evaluated rows | 21.87182 |

These results were obtained using the GPU model configuration. Results may vary with data, dependencies, and hardware. RMSE penalizes large errors; it does **not** mean every individual prediction is within the reported number of AQI points.

## Input data requirements

The project keeps example input files in a **`sample_data/`** subfolder: **`train.csv`**, **`test.csv`**, and **`sample_submission.csv`**. The last file is simply an **output template** containing row identifiers; the program copies it and replaces its AQI values with predictions. The current `main.py` reads the CSV files from the project root, so copy these three files there before running (see **Run the project** below).

### Required columns

| Column | Expected data | Notes |
| --- | --- | --- |
| `Date` | Date string, e.g. `01/29/2020` | **Month/day/year** (`MM/DD/YYYY`); one date per observation |
| `Time` | Time string, e.g. `09:00 AM` | **12-hour AM/PM** format |
| `City` | Text, e.g. `Delhi` | Must be present and consistently spelled; time-series features are grouped by this field |
| `location` | Region label, e.g. `North` | Case-sensitive categorical value; see accepted labels below |
| `PM2.5` | Numeric pollutant concentration | Particulate matter ≤ 2.5 µm |
| `PM10` | Numeric pollutant concentration | Particulate matter ≤ 10 µm |
| `NO` | Numeric pollutant concentration | Nitric oxide |
| `NO2` | Numeric pollutant concentration | Nitrogen dioxide |
| `NOx` | Numeric pollutant concentration | Nitrogen oxides |
| `NH3` | Numeric pollutant concentration | Ammonia |
| `CO` | Numeric pollutant concentration | Carbon monoxide |
| `SO2` | Numeric pollutant concentration | Sulfur dioxide |
| `O3` | Numeric pollutant concentration | Ozone |
| `Benzene` | Numeric pollutant concentration | Benzene |
| `Toluene` | Numeric pollutant concentration | Toluene |
| `AQI` | Numeric target | **Required in `train.csv` only**; omit from `test.csv` |

The **11 pollutant column names must match exactly**, including capitalization and punctuation.

**Measurement units:** The model's AQI-inspired calculations assume the conventional Indian AQI concentration units: **CO in mg/m³** and **PM2.5, PM10, NO2, SO2, NH3, O3 in µg/m³**. The other pollutant inputs must use units consistent with the training data, typically µg/m³. Check units against your data provider and convert them **before** running the model. Mixing units can seriously distort predictions. Do not use precomputed AQI subindices in place of measured pollutant concentrations.

### Accepted `location` values

The current feature-count checks assume the **training dataset contains all nine region labels** shown below:

```text
Central, East, North, Northeast, Northwest,
South, Southeast, Southwest, West
```

Use these spellings exactly. This field represents a broad geographic **region**, not a street address or monitoring-station name. `test.csv` may contain a subset of these known labels. A completely different set of region categories will require adapting the feature engineering and feature-count checks.

### CSV file formats

**`train.csv`** — observations with a known AQI target:

```csv
Date,Time,City,location,PM2.5,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,AQI
01/29/2020,09:00 AM,Delhi,North,75.2,142.0,12.4,34.5,46.9,26.0,1.2,8.6,42.0,1.5,3.2,165
```

**`test.csv`** — observations for which AQI should be estimated:

```csv
Date,Time,City,location,PM2.5,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene
01/29/2020,10:00 AM,Delhi,North,77.1,146.3,13.0,35.2,48.2,25.5,1.3,8.1,40.2,1.6,3.0
```

**`sample_submission.csv`** — one distinct `Index` per test row, in the same row order:

```csv
Index,AQI
1,0
```

These rows are **illustrative**, not a complete runnable dataset. The actual input must have enough observations and category coverage for the code's feature checks, and the output template must have exactly as many rows as `test.csv`.

### Data quality guidelines

- **Use observed pollutant concentrations** from a consistent monitoring/data source. Values should be numeric, physically plausible, and expressed in consistent units.
- **Keep timestamps unambiguous.** Use the specified date/time formats and a consistent local time convention across the dataset. Avoid duplicate `City` + `Date` + `Time` observations.
- **Preserve historical sequences.** The model calculates features from earlier observations *within the same city*. It uses up to **72 previous observation positions** and time-based windows of up to **24 hours**. A single isolated row may run but cannot provide representative historical features; include sufficient recent history whenever possible.
- **Record missing measurements as empty cells or `NaN`, not zero.** Some missing pollutant readings are supported by the models, but incomplete history reduces the usefulness of rolling features. `AQI` targets in `train.csv` must be present and numeric.
- **Keep the schema consistent.** The current script expects **78 XGBoost features** and **214 LightGBM features** from the training data. Arbitrary new categorical values or removed columns can trigger assertions or change model behavior.
- **Use enough cities for validation.** `VALIDATION_MODE=True` requires at least **five distinct training cities** for 5-fold city-grouped validation. Prediction mode does not have that specific requirement.
- **Do not supply `AQI_Bucket` as a substitute for `AQI`.** It is not used as a feature or target in this pipeline. Additional descriptive columns may be present but are ignored by the feature builders.

## Repository structure

```text
AQI-prediction-ML-model/
├── main.py             # Data loading, validation, training, predictions
├── models.py           # Regressor settings and model feature assembly
├── features.py         # Pollutant history and calendar feature engineering
├── physics.py          # Time-based features and AQI-inspired subindices
├── config.py           # Model weights, validation mode and breakpoints
├── sample_data/        # Example input datasets
│   ├── train.csv       # Labeled pollutant observations and AQI targets
│   ├── test.csv        # Pollutant observations to predict
│   └── sample_submission.csv  # Output format / row identifiers
├── pyrightconfig.json  # Python type-checker configuration
├── .gitignore          # Local file exclusions
└── README.md
```

The `sample_data/` folder holds the input files separately from the Python source. Check dataset redistribution permissions before publishing data files in a public repository. If `.gitignore` excludes CSV files, Git will not add them automatically.

## Installation

**Requirements:** Python 3, NumPy, pandas, scikit-learn, XGBoost, and LightGBM.

```bash
git clone https://github.com/NULLvector-bit/AQI-prediction-ML-model.git
cd AQI-prediction-ML-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn xgboost lightgbm
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

### GPU configuration

The model settings in `models.py` use:

- XGBoost: `tree_method="hist"`, `device="cuda"`
- LightGBM: `device_type="gpu"` (OpenCL GPU trainer)

These settings require compatible GPU hardware, drivers, and a LightGBM build with GPU support. To train on a CPU instead, change them to `device="cpu"` and `device_type="cpu"`, respectively.

## Run the project

Keep the three CSV files in `sample_data/`. Because `main.py` currently expects them next to itself, copy them into the project root before running:

```bash
cp sample_data/train.csv sample_data/test.csv sample_data/sample_submission.csv .
```

On Windows PowerShell, use `Copy-Item sample_data\*.csv .` instead. This copies the files without changing the originals in `sample_data/`.

**Evaluate using cross-validation:**

1. Set `VALIDATION_MODE = True` in `config.py`.
2. Run `python main.py`.
3. The program prints each fold's RMSE, mean RMSE, and pooled RMSE.

**Train on all labeled data and generate predictions:**

1. Set `VALIDATION_MODE = False` in `config.py`.
2. Run `python main.py`.
3. The program writes **`submission.csv`**, containing the `Index` values from `sample_submission.csv` and predicted `AQI` values corresponding to the rows of `test.csv`.

> **Note:** The output filename `submission.csv` is set in the existing code. Running prediction mode again overwrites this file. The script trains models each time; it does not load saved models or accept an individual observation interactively.

## Interpretation and limitations

- This is an **AQI estimator based on pollutant inputs**, not a weather or emissions forecast.
- Historical lag and rolling features depend on the available observations for each city; sparse data can reduce accuracy.
- The AQI subindices in `physics.py` are **approximate model features** and should not be presented as an official AQI calculation or an official monitoring-station reading.
- Validation performance describes the evaluated historical dataset. It does not guarantee the same error for a different location, time period, monitoring instrument, or pollutant distribution.
- Predictions are not a substitute for official air-quality bulletins or public-health guidance.
