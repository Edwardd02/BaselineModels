import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
# Load data from pickle files
PREFIX = Path(r"data_imputation")
def getDevice():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = getDevice()

def RMSE(predicted, actual):
  return math.sqrt(sum((predicted - actual)**2)/len(predicted))

def MAE(predicted, actual):
  return sum(abs(predicted - actual))/len(predicted)

def MAPE( predicted,  actual):
  return sum(abs((actual - predicted)/ actual))/len(predicted)

def plot(pred, actual, start, end, title = "Prediction vs Actual", y = "VWC (mm)", x = "Time", extras=[]):
  fig, ax = plt.subplots(figsize=(20, 5))
  ax.set_title(title)
  ax.set_ylabel(y)
  ax.set_xlabel(x)
  ax.plot( pred[start:end], label="Predicted")
  ax.plot( actual[start:end], label="Actual")
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

def save(model, path):
  torch.save(model.state_dict(), path)

def load(model, path):
  model.load_state_dict(torch.load(path))

def get_indices(table, lag, output_col):
  numRows = len(table)
  # Identify rows with NaN values
  nanRows = table.isna().any(axis=1).to_numpy()
  # Initialize the set of valid indices
  indices = set(range(numRows))
  # Create a mask to identify invalid indices due to NaNs in the lag period
  invalid_indices = set()
  # Remove indices with NaN values in the lag period
  for row in range(len(nanRows)):
    for l in range(lag+1):
      if row + lag+1 >= numRows:
        invalid_indices.add(row)
      else:
        if nanRows[row+l]:
          invalid_indices.add(row)
          break
    if row + lag + 1 < numRows and math.isnan(table.iloc[row + lag + 1][output_col]):
      invalid_indices.add(row)

  # Remove invalid indices from the indices set
  indices.difference_update(invalid_indices)

  return sorted(indices)

def locate_gap(table, col, start):
  gapSize = 0
  startIndex = start
  #loops through column and updates start and end when it sees a gap (a series of nan values)
  for i in range(start, len(table[col])):
    if math.isnan(table.iloc[i][col]):
      if startIndex == start:
        startIndex = i
      gapSize +=1
    else:
      if gapSize > 25:
        return startIndex, i
      else:
        startIndex = 0
        gapSize = 0
  return -1, -1

def remove_columns_with_gap(table, start, end, exclude_col):
  newTable = table
  if start > end:
    return -1
  #loops throgh the columns
  #If there is an nan value between start and end (of gap)
  #remove column
  for col in newTable.columns:
    if col != exclude_col:
      for i in range (start, end):
        if math.isnan(newTable.iloc[i][col]):
          newTable = newTable.drop(columns=[col])
          break
  return newTable

def get_input(index, data):
  #creates an empty 1d array
  x = []
  for l in range(lags+1):
    #Adds to the 1d array (entire row)
    x.extend(data.iloc[index+l])
  #adds the last row to the 1d array (add all values except the last - this is the current timestamp, so you want all values except for the nan value)
  x.extend(data.iloc[index+lags+1, :len(data.iloc[0])-1])
  return x

#normalize between 0,1
def normalize(data, min, max):
  return (data - min) / (max - min)

#denormalize
def denormalize(data, min, max):
  return data * (max - min) + min

class ArticleModel(nn.Module):
  #Creates the model
  def __init__(self, hidden_layers, input_size, hidden_size, dropout_rate = .5, output_size = 1):
    super().__init__()
    self.input = nn.Linear(input_size, hidden_size, dtype=torch.double)
    self.layers = nn.ModuleList()
    self.dropout = nn.Dropout(dropout_rate)
    for i in range(hidden_layers):
      self.layers.append(nn.Linear(hidden_size, hidden_size, dtype=torch.double))
      self.layers.append(self.dropout)
    self.output = nn.Linear(hidden_size, 1, dtype=torch.double)

  #This function gets called when model(val) is used in the code
  #It takes an input of input size and outputs a value of output size
  def forward(self, x):
    x = F.relu(self.input(x))
    for layer in self.layers:
      x = F.relu(layer(x))
    x = self.output(x)
    return x

def extract_batch(batch, data, num_columns):
  # Create a range tensor on the same device as batch
  offsets = torch.arange(((lags + 1) * len(gaps.iloc[0])) + len(gaps.iloc[0]) - 1).to(batch.device)

  # Expand batch indices to create a 2D index matrix
  indices = (batch[:, None]*num_columns) + offsets[None, :]
  indices = indices.cpu()  # Add this line

  # Extract data using advanced indexing
  x = data[indices]

  # Extract corresponding y values (assuming next element after sequence)
  y_indices = ((batch[:, None] + lags+1)*num_columns + len(gaps.iloc[0]) - 1)
  y_indices = y_indices.cpu() # Move y_indices to CPU
  y = data[y_indices]

  return x, y

def get_loss(model, batch, data, num_columns):
  #Get the input and output data using the batch (list of indices)
  x,y = extract_batch(batch, data, num_columns)

  #Convert x,y to tensors for faster processing / gpu sability
  #Do not know if this is actually useful here
  if type(x) != torch.Tensor:
    x = torch.tensor(x).double().to(device)
  if type(y) != torch.Tensor:
    y = torch.tensor(y).double().to(device)
    y = y.reshape(-1, 1)

  out = model(x)

  #mse loss function
  loss = F.mse_loss(out, y)

  return loss

def fit(model, epochs, train_dl, validation_dl, lr, data, num_columns):
  history = []
  data = data.values.flatten()
  #adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  #loop through epochs
  for epoch in range(epochs):
    #loop through batches
    for batch in train_dl:

      #train model
      loss = get_loss(model, batch, data, num_columns)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    #validate model
    currentHistory = []
    for batch in validation_dl:
      loss = get_loss(model, batch, data, num_columns)
      currentHistory.append(loss.item())

    history.append(sum(currentHistory)/len(currentHistory))
    print("Epoch: %s Loss: %s" % (epoch, sum(currentHistory)/len(currentHistory)))
  return history

batch_size = 128
epochs = 5
hidden_size = 256
layers = 3
dropout = 0
lags = 14

gaps = pd.read_pickle(PREFIX / r"SMPSKLearnGap.pkl")
actual = pd.read_pickle(PREFIX / r"OriginalSMPData.pkl")

def prep_data(original_data, col, lags, batch_size):
  #Find the gap
  start, end = locate_gap(original_data, col, 0)
  if start == -1:
    return -1, -1, -1, -1

  #If other columns have gaps in the same interval, remove them
  data = remove_columns_with_gap(original_data, start, end, col)

  #Rearrange columns so that the columns with the gap is the last column
  names = []
  for k in data.columns:
    if k != col:
      names.append(k)
  names.append(col)
  data = data.reindex(columns=names)

  #get usable indices (as per indices algoritms)
  indices = get_indices(data, lags, col)
  indices = torch.tensor(indices).to(device)

  #85/15 split for train / val sets
  xSplit = int((.85*len(indices)))
  trainInd = indices[0:xSplit]
  valInd = indices[xSplit:]
  train_dl = DataLoader(trainInd, batch_size)
  val_dl = DataLoader(valInd, batch_size)

  #normalize data
  #norm params saves the min and max values for later denormalization
  #normalizes the training and validation sets separately
  norm_params = {}
  for c in gaps.columns:
    trainNorm = pd.DataFrame()
    valNorm = pd.DataFrame()
    norm_params[c] = [data[c][0:xSplit].min(), data[c][0:xSplit].max()]
    trainNorm[c] = normalize(data[c][0:xSplit], norm_params[c][0], norm_params[c][1])
    valNorm[c] = normalize(data[c][xSplit:], norm_params[c][0], norm_params[c][1])
    trainNorm.index = data.index[0:xSplit]
    valNorm.index = data.index[xSplit:]
    data[c] = pd.concat([trainNorm[c], valNorm[c]])

  return train_dl, val_dl, data, norm_params

def impute(model, data, col, lags, norm_params, gaps, pred):
  #Impute missing Values
  #output_column is the column with the gap (this is necessary because the loop will fill in this column as it goes)
  output_column = gaps[col]

  output_column = torch.tensor(output_column.values).double().to(device)

  #llop through output column
  for i in range(len(output_column)):
    #if it encounters a nan
    if math.isnan(output_column[i]):
    #compute the missing value and update the lagged table with the missing value
      if i - lags - 1 >= 0:
        inp = torch.tensor(get_input(i-lags-1, data)).double().to(device)
        out = model(inp)
        out = out.item()
        output_column[i] = out
      #update the table with the missing value
      data.iloc[i, data.columns.get_loc(col)] = out

      #this is important only if multiple columns have gaps
      gaps.iloc[i, gaps.columns.get_loc(col)] = denormalize(out, norm_params[col][0], norm_params[col][1])


  #Add the output column to prediction dataframe
  pred[col] = output_column.cpu().numpy()
  pred[col] = denormalize(pred[col], norm_params[col][0], norm_params[col][1])
  pred.index = gaps.index
  return pred, gaps
start_time = time.time()

#Pred is the data frame containing all predictions
#Note it will only contain columns with gaps in them
pred = pd.DataFrame()

#loop through all columns in the data frame
for col in gaps.columns:

  #This is here for testing purposes
  if col != "P2_VWC":
    continue

  #Prepare the data
  train_dl, val_dl, data, norm_params = prep_data(gaps, col, lags, batch_size)

  #If there is no gap then all previoud variables will be -1
  if train_dl == -1:
    continue

  #Create Model
  model = ArticleModel(hidden_layers = layers, input_size = ((len(gaps.columns)*(lags+1))+len(gaps.columns)-1), hidden_size = hidden_size, dropout_rate = dropout)

  model.to(device)

  #Train Model
  history = fit(model, epochs, train_dl, val_dl, 0.001, data, len(gaps.columns))
  print(history)

  #Impute missing Values
  #output_column is the column with the gap (this is necessary because the loop will fill in this column as it goes)
  pred, gaps = impute(model, data, col, lags, norm_params, gaps, pred)
end_time = time.time()
print(f"Training time: {end_time - start_time}")
intervals = {"P2_VWC": [1440, 2160]}
for col in intervals:
  plot(pred[col], actual[col], intervals[col][0], intervals[col][1], title="MLP prediction on SKLearn filled SMP data")
plt.show()