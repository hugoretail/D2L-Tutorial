import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

import pandas as pd
data = pd.read_csv(data_file)
# print(data)

# This line of code is splitting the data into two parts: `inputs` and `targets`.
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

inputs = inputs.fillna(inputs.mean())
# print(inputs)

import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
# print(X, "\n", y)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
cols = [
    "Sex", "Length", "Diameter", "Height", "Whole weight",
    "Shucked weight", "Viscera weight", "Shell weight", "Rings"
]
abalone = pd.read_csv(url, names=cols)
# print(abalone.head())

# NOT WORKING
# def count_missing_values(data, cols):
#     cols_dict = {col: 0 for col in cols}
#     counter = 0
#     for col in range(len(data)):
#         for y in range(len(cols)):
#             if data[col][y] == None: counter+=1
#         cols_dict[col] = counter
#         counter = 0
#     return cols_dict
def count_missing_values(df,cols):
    return df[cols].isna().sum().to_dict()

missing_values = count_missing_values(abalone, cols)
# print(missing_values)