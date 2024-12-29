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
from itertools import product

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

# param_grid = {
#     'n_estimators': [100, 200],  # Number of trees in Random Forest
#     'max_depth': [5, 10],        # Maximum depth of trees
#     'min_samples_split': [2, 5], # Minimum samples to split a node
#     'min_samples_leaf': [1, 2],  # Minimum samples at a leaf node
#     'max_features': ['sqrt', 'log2'],  # Number of features to consider for splits
# }
param_grid = {
    'n_estimators': [100],  # Number of trees in Random Forest
    'max_depth': [5],        # Maximum depth of trees
    'min_samples_split': [2], # Minimum samples to split a node
    'min_samples_leaf': [2],  # Minimum samples at a leaf node
    'max_features': ['log2'],  # Number of features to consider for splits
    'random_state': range(1, 50)
}
results = []

gaps = pd.read_pickle(PREFIX / r"SMPSingleGap.pkl")
actual = pd.read_pickle(PREFIX / r"OriginalSMPData.pkl")

for params in product(*param_grid.values()):
    # Unpack parameters
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, random_state = params

    # Define Random Forest models
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state
    )
    rgr = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state
    )
    print("Fitting with parameters:", params)

    # Define MissForest
    model = MissForest(
        clf,
        rgr,
        max_iter=15,
        initial_guess="mean",
        verbose=False
    )

    # Fit and transform the data
    pred = model.fit_transform(gaps)

    # Evaluate performance (e.g., RMSE, MSE, MAPE for a specific column)
    col = "P2_VWC"
    start, end = 1440, 2160
    rmse = RMSE(pred[col][start:end], actual[col][start:end])
    mae = MAE(pred[col][start:end], actual[col][start:end])
    mape = MAPE(pred[col][start:end], actual[col][start:end])

    # Store results
    results.append({
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'random_state': random_state
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define weights for the composite score
weight_rmse = 1 / 3
weight_mse = 1 / 3
weight_mape = 1 / 3

# Calculate a composite score
results_df['Composite_Score'] = (
        weight_rmse * results_df['RMSE'] +
        weight_mse * results_df['MAE'] +
        weight_mape * results_df['MAPE']
)

# Sort by the composite score (lower is better)
results_df = results_df.sort_values(by='Composite_Score')

# Save the sorted results to a file (e.g., CSV)
results_df.to_csv('param_results_missforest.csv', index=False)

# Print the best parameters
print("Best Parameters:")
print(results_df.iloc[0])
