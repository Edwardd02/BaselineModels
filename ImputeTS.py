from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import torch
import pandas as pd
from pathlib import Path
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Set the correct local path
PREFIX = Path(r"data_imputation")

# Activate pandas to R conversion
pandas2ri.activate()

# Import the imputeTS R package
imputeTS = importr('imputeTS')

# Load data from pickle files
gaps = pd.read_pickle(PREFIX / r"5GapsHourly.pkl")
actual = pd.read_pickle(PREFIX / r"NoGapsHourly.pkl")

# Function definitions remain the same
def RMSE(predicted, actual):
    return math.sqrt(((predicted - actual) ** 2).mean())

def MAE(predicted, actual):
    return abs(predicted - actual).mean()

def MAPE(predicted, actual):
    return (abs((actual - predicted) / actual)).mean()

def plot(pred, actual, start, end, title="Prediction vs Actual", y="VWC (mm)", x="Time", extras=[]):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.plot(pred[start:end], label="Predicted")
    ax.plot(actual[start:end], label="Actual")
    for extra in extras:
        ax.plot(extra[start:end], label=extra.name)
    ax.legend()

def statistics(pred, actual):
    p = torch.tensor(pred.values) if not isinstance(pred, torch.Tensor) else pred
    a = torch.tensor(actual.values) if not isinstance(actual, torch.Tensor) else actual
    rmse = RMSE(p, a)
    mae = MAE(p, a)
    mape = MAPE(p, a)
    print(f"RMSE: {rmse.item() if isinstance(rmse, torch.Tensor) else rmse}")
    print(f"MAE: {mae.item() if isinstance(mae, torch.Tensor) else mae}")
    print(f"MAPE: {mape.item() if isinstance(mape, torch.Tensor) else mape}")

def addStats(methodName, pred, actual, intervals, path="stats/stats.csv", desc="NA", dataset="New Mexico"):
    stats = pd.read_csv(path, index_col=False)
    for key in intervals:
        gap = key
        interval = intervals[key]
        p = pred[key][interval[0]:interval[1]]
        a = actual[key][interval[0]:interval[1]]
        rmse = RMSE(p, a)
        mae = MAE(p, a)
        mape = MAPE(p, a)
        intervalLen = interval[1] - interval[0]
        timeLen = actual.index[interval[1]] - actual.index[interval[0]]
        stats.loc[len(stats)] = [methodName, desc, dataset, gap, intervalLen, timeLen, rmse, mae, mape]
    stats.to_csv(path, index=False)

# Define intervals
intervals = {
    "1VWCGrass_Avg": [457, 1176],
    "1VWCShrub_Avg": [3603, 5763],
    "1VWCBare_Avg": [8644, 12243],
    "2VWCGrass_Avg": [12244, 18003],
    "2VWCShrub_Avg": [32404, 41043]
}

pred_r = imputeTS.na_seadec(pandas2ri.py2rpy(gaps))
pred = pandas2ri.rpy2py(pred_r)
pred.index = actual.index

for key in intervals:
    print(key)
    statistics(pred[key][intervals[key][0]:intervals[key][1]], actual[key][intervals[key][0]:intervals[key][1]])
    print("\n")

for key in intervals:
    plot(pred[key], actual[key], intervals[key][0]-400, intervals[key][1]+400, title=key)
plt.show()

# Repeat similar steps for other imputation methods (na_interpolation, na_kalman, na_locf, na_mean)
naInterpolation_r = imputeTS.na_interpolation(pandas2ri.py2rpy(gaps), option="spline")
naInterpolation = pandas2ri.rpy2py(naInterpolation_r)
naInterpolation.index = actual.index

for key in intervals:
    print(key)
    statistics(naInterpolation[key][intervals[key][0]:intervals[key][1]], actual[key][intervals[key][0]:intervals[key][1]])
    print("\n")

for key in intervals:
    plot(naInterpolation[key], actual[key], intervals[key][0]-400, intervals[key][1]+400, title=key)
plt.show()

naLocf_r = imputeTS.na_locf(pandas2ri.py2rpy(gaps), option="spline")
naLocf = pandas2ri.rpy2py(naLocf_r)
naLocf.index = actual.index

for key in intervals:
    print(key)
    statistics(naLocf[key][intervals[key][0]:intervals[key][1]], actual[key][intervals[key][0]:intervals[key][1]])
    print("\n")

for key in intervals:
    plot(naLocf[key], actual[key], intervals[key][0]-400, intervals[key][1]+400, title=key)
plt.show()