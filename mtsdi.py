import subprocess
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
from pathlib import Path

# Set PREFIX to your local data directory
PREFIX = Path(r"data_imputation")


# Execute R script and pass the directory path
proc = subprocess.run(['Rscript', 'mtsdi_generate_pred.R', os.path.abspath(str(PREFIX))], capture_output=True, text=True)
print("R script output:")
print(proc.stdout)
print("R script errors:")
print(proc.stderr)

# Read prediction data
pred = pd.read_csv(PREFIX / "pred.csv", index_col=0)
actual = pd.read_pickle(PREFIX / "OriginalSMPData.pkl")

pred.index = actual.index  # Ensure indices match actual data

# Define intervals for analysis
intervals = {"P2_VWC": [1440, 2160]}

# Statistics functions
def RMSE(predicted, actual):
    return math.sqrt(sum((predicted - actual) ** 2) / len(predicted))

def MAE(predicted, actual):
    return sum(abs(predicted - actual)) / len(predicted)

def MAPE(predicted, actual):
    return sum(abs((actual - predicted) / actual)) / len(actual)

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

# Adds statistics to the stats file
def addStats(methodName, pred, actual, start, end, path=PREFIX / "clean/stats.csv", desc="Default", dataset="SMPNotFilled"):
    if os.path.exists(path):
        stats = pd.read_csv(path, index_col=False)
    else:
        stats = pd.DataFrame(columns=["Method", "Description", "Dataset", "Gap Length (rows)", "Gap Length (time)", "RMSE", "MAE", "MAPE"])

    rmse = RMSE(pred[start:end], actual[start:end])
    mae = MAE(pred[start:end], actual[start:end])
    mape = MAPE(pred[start:end], actual[start:end])
    intervalLen = end - start
    timeLen = actual.index[end] - actual.index[start]
    stats.loc[len(stats)] = [methodName, desc, dataset, intervalLen, timeLen, rmse, mae, mape]

    stats.to_csv(path, index=False)

# Run statistics and plotting
for col in intervals:
    statistics(pred[col][intervals[col][0]:intervals[col][1]], actual[col][intervals[col][0]:intervals[col][1]])

for col in intervals:
    plot(pred[col], actual[col], intervals[col][0], intervals[col][1], title="Not Filled P2_VWC")
