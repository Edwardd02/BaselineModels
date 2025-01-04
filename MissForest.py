from pathlib import Path
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from missforest.missforest import MissForest
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

PREFIX = Path(r"data_imputation")


def RMSE(predicted, actual):
    return math.sqrt(sum((predicted - actual) ** 2) / len(predicted))


def MAE(predicted, actual):
    return sum(abs(predicted - actual)) / len(predicted)


def MAPE(predicted, actual):
    return sum(abs((actual - predicted) / actual)) / len(predicted)


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


def statisics(pred, actual):
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


def addStats(methodName, pred, actual, start, end, path=PREFIX / r"stats/stats.csv", desc="Default",
             dataset="SMPNotFilled"):
    if os.path.exists(path):
        stats = pd.read_csv(path, index_col=False)
    else:
        # Method is the name of the method. For example, SKLearn, MLP, etc
        # Description is for any misc info
        # Dataset is used if there are multiple datasets being worked on simultaneously.
        # There are 2 Gap Lengths: one for the number of rows, one for the time duration.
        # The rest are the statistics
        stats = pd.DataFrame(
            columns=["Method", "Description", "Dataset", "Gap Length (rows)", "Gap Length (time)", "RMSE", "MAE",
                     "MAPE"])

    method = methodName
    rmse = RMSE(pred[start:end], actual[start:end])
    mae = MAE(pred[start:end], actual[start:end])
    mape = MAPE(pred[start:end], actual[start:end])
    intervalLen = end - start
    timeLen = actual.index[end] - actual.index[start]
    stats.loc[len(stats)] = [method, desc, dataset, intervalLen, timeLen, rmse, mae, mape]

    stats.to_csv(path, index=False)


gaps = pd.read_pickle(PREFIX / r"SMPSingleGap.pkl")
actual = pd.read_pickle(PREFIX / r"OriginalSMPData.pkl")

clf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=2,
                             max_features='log2', n_jobs=-1, random_state=32
                             )
rgr = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=2,
                             max_features='log2', n_jobs=-1, random_state=32
                            )
model = MissForest(clf, rgr, max_iter=15, initial_guess="mean", verbose=True)
pred = model.fit_transform(gaps)

intervals = {"P2_VWC": [1440, 2160]}
for col in intervals:
    statisics(pred[col][intervals[col][0]:intervals[col][1]], actual[col][intervals[col][0]:intervals[col][1]])

for col in intervals:
    plot(pred[col], actual[col], intervals[col][0], intervals[col][1], title="MissForest prediction on SMP data")
# for col in intervals:
#   addStats("MissForest", pred[col], actual[col], intervals[col][0], intervals[col][1], desc = "Using SKLearn Regressors")
plt.show()
