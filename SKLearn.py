from pathlib import Path
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

# Set PREFIX to your local data directory
PREFIX = Path(r"data_imputation")

# Statistics functions
def RMSE(predicted, actual):
    return math.sqrt(sum((predicted - actual) ** 2) / len(predicted))

def MAE(predicted, actual):
    return sum(abs(predicted - actual)) / len(predicted)

def MAPE(predicted, actual):
    return sum(abs((actual - predicted) / actual)) / len(actual)

# Creates a plot
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
    plt.show()

# Prints all the statistics
def statistics(pred, actual):
    if type(pred) != torch.Tensor:
        p = torch.tensor(pred.values)
    else:
        p = pred
    if type(actual) != torch.Tensor:
        a = torch.tensor(actual.values)
    else:
        a = actual

    rmse = RMSE(p, a)
    mae = MAE(p, a)
    mape = MAPE(p, a)
    if type(rmse) == torch.Tensor:
        rmse = rmse.item()
    if type(mae) == torch.Tensor:
        mae = mae.item()
    if type(mape) == torch.Tensor:
        mape = mape.item()
    print("RMSE: %s" % (rmse))
    print("MAE: %s" % (mae))
    print("MAPE: %s" % (mape))


# Load data
gaps = pd.read_pickle(PREFIX / "SMPSingleGap.pkl")
actual = pd.read_pickle(PREFIX / "OriginalSMPData.pkl")

# Create Imputer
imp = IterativeImputer()

# Train Imputer
imp.fit(gaps)

# Make Prediction
pred = imp.transform(gaps)

# Formatting the data to a usable DataFrame
pred = pd.DataFrame(pred)
pred = pred.rename(columns=dict(zip(pred.columns, actual.columns)))
pred.index = actual.index

# Define intervals
intervals = {"P2_VWC": [1440, 2160]}

# Run statistics and plotting
for col in intervals:
    statistics(pred[col][intervals[col][0]:intervals[col][1]], actual[col][intervals[col][0]:intervals[col][1]])

for col in intervals:
    plot(pred[col], actual[col], intervals[col][0], intervals[col][1], title="Not Filled P2_VWC")
