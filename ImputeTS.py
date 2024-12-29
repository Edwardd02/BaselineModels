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
def perform_imputation_and_analysis(imputation_function, gaps, actual, intervals, title_prefix="Imputation", **kwargs):
    """
    Perform imputation, calculate statistics, and generate plots.

    Parameters:
    - imputation_function: R imputation function from the imputeTS package.
    - gaps: Pandas DataFrame with missing values to impute.
    - actual: Pandas DataFrame with actual (ground truth) values.
    - intervals: Dictionary defining the intervals for evaluation.
    - title_prefix: Prefix for plot titles.
    - kwargs: Additional arguments for the R imputation function.

    Returns:
    - imputed_df: Pandas DataFrame containing the imputed values.
    """
    from rpy2.robjects import pandas2ri
    import matplotlib.pyplot as plt

    # Convert gaps DataFrame to R format and perform imputation
    imputed_r = imputation_function(pandas2ri.py2rpy(gaps), **kwargs)
    imputed_df = pandas2ri.rpy2py(imputed_r)
    imputed_df.index = actual.index

    # Calculate and display statistics for each interval
    for key in intervals:
        print(key)
        start, end = intervals[key]
        statistics(imputed_df[key][start:end], actual[key][start:end])
        print("\n")

    # Plot the results
    for key in intervals:
        start, end = intervals[key]
        plot(imputed_df[key], actual[key], start - 400, end + 400, title=f"{title_prefix} - {key}")
    plt.show()

    return imputed_df

# na_seadec = perform_imputation_and_analysis(
#     imputeTS.na_seadec,
#     gaps,
#     actual,
#     intervals,
#     title_prefix="na_seadec"
# )
#
# na_interpolation = perform_imputation_and_analysis(
#     imputeTS.na_interpolation,
#     gaps,
#     actual,
#     intervals,
#     title_prefix="na_interpolation",
# )
#
# na_Kalman = perform_imputation_and_analysis(
#     imputeTS.na_Kalman,
#     gaps,
#     actual,
#     intervals,
#     title_prefix="na_Kalman"
# )

na_locf = perform_imputation_and_analysis(
    imputeTS.na_locf,
    gaps,
    actual,
    intervals,
    title_prefix="na_locf",
)

# na_mean = perform_imputation_and_analysis(
#     imputeTS.na_mean,
#     gaps,
#     actual,
#     intervals,
#     title_prefix="na_mean"
# )

# na_random = perform_imputation_and_analysis(
#     imputeTS.na_random,
#     gaps,
#     actual,
#     intervals,
#     title_prefix="na_random"
# )
