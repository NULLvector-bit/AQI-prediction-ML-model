import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

imputer = SimpleImputer(strategy="median")
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=67)
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
model = LinearRegression()
model.fit(X_train_imputed, y_train)
y_pred = model.predict(X_val_imputed)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE:", rmse)
