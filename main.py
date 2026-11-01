import numpy as np
import pandas as pd

data = pd.read_csv("train.csv")
print(data.head)
print(data.shape)
print(data.columns)
print(data.info())
print(data.isnull().sum())
