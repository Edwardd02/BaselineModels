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
import time

# Set PREFIX to your local data directory
PREFIX = Path(r"data_imputation")

hyperparam_combinations = [
    {'method': 'spline', 'm': 5},
    {'method': 'spline', 'm': 10},
    {'method': 'norm', 'm': 5},
    {'method': 'norm', 'm': 10},
    {'method': 'spline', 'm': 5},
]

# Initialize list to store results
results = []

# Read actual data once
actual = pd.read_pickle(PREFIX / "OriginalSMPData.pkl")

for hyperparams in hyperparam_combinations:
    # Create a list of arguments for the R script
    args = ['Rscript', 'mtsdi_generate_pred.R', os.path.abspath(str(PREFIX))]
    # Add hyperparameters to the arguments
    for param, value in hyperparams.items():
        args.append(f"--{param}")
        args.append(str(value))

    # Execute R script and capture output
    proc = subprocess.run(args, capture_output=True, text=True)
    print("R script output:")
    print(proc.stdout)
    print("R script errors:")
    print(proc.stderr)

    # Extract training time from stdout
    stdout = proc.stdout
    training_time_line = [line for line in stdout.split('\n') if line.startswith("Training time:")]
    if training_time_line:
        training_time = float(training_time_line[0].split(':')[1].strip())
    else:
        training_time = None  # Handle error

    # Check if "pred.csv" exists before reading
    pred_path = Path("C:/Users/78405/PycharmProjects/BaselineModels/pred.csv")
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, index_col=0)
        pred.index = actual.index  # Ensure indices match actual data
    else:
        print("pred.csv not found. Skipping this combination.")
        continue

    # Define intervals for analysis
    intervals = {"P2_VWC": [1440, 2160]}

    # Calculate performance metrics
    def RMSE(predicted, actual):
        return math.sqrt(sum((predicted - actual) ** 2) / len(predicted))

    def MAE(predicted, actual):
        return sum(abs(predicted - actual)) / len(predicted)

    def MAPE(predicted, actual):
        return sum(abs((actual - predicted) / actual)) / len(actual)

    # Collect metrics for each interval
    metrics = {}
    for col in intervals:
        start, end = intervals[col]
        predicted = pred[col][start:end]
        actual_col = actual[col][start:end]
        if len(predicted) == 0 or len(actual_col) == 0:
            print("Empty data for metrics. Skipping this combination.")
            continue
        metrics[f'RMSE_{col}'] = RMSE(predicted, actual_col)
        metrics[f'MAE_{col}'] = MAE(predicted, actual_col)
        metrics[f'MAPE_{col}'] = MAPE(predicted, actual_col)

    # Record results
    result = hyperparams.copy()
    result['training_time'] = training_time
    result.update(metrics)
    results.append(result)

    # Optionally, remove or move "pred.csv" to avoid interference
    os.remove(pred_path)

# Create DataFrame from results and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(PREFIX / "hyperparam_tuning_results.csv", index=False)

# Analyze results to find the best hyperparameters
# For example, find the combination with the lowest RMSE
best_rmse = results_df['RMSE_P2_VWC'].min()
best_params = results_df.loc[results_df['RMSE_P2_VWC'] == best_rmse]
print("Best hyperparameters based on RMSE:")
print(best_params)