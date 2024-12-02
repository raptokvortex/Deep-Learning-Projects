import numpy as np
import matplotlib.pyplot as plt
import Data_Cleaning2
import pandas as pd

# Set up some sensible column names for our columns
column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

# Import the data
raw_data = pd.read_csv("Data_A.csv", names = column_names)

# Split the data into training and test sets
split_ratio = 0.9
upper = int(raw_data.shape[0]*split_ratio)
training_data = raw_data.iloc[0:upper]
test_data = raw_data.iloc[upper:]


# Split off the objectives
training_objective = training_data['midprice change direction']
test_objective = test_data['midprice change direction']
training_data = training_data.drop(labels = ['midprice change direction'], axis = 1)
test_data = test_data.drop(labels = ['midprice change direction'], axis = 1)

scale = False

if not scale:
    training_data = Data_Cleaning2.transform_data(training_data, drop = True)
    test_data = Data_Cleaning2.transform_data(test_data, drop = True)




if scale:
    training_data = Data_Cleaning2.transform_data(training_data, drop = True)
    test_data = Data_Cleaning2.transform_data(test_data, drop = True)
    column_names = training_data.columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    training_data = scaler.fit_transform(training_data)
    test_data = scaler.transform(test_data) 

    training_data = pd.DataFrame(training_data, columns = column_names)
    test_data = pd.DataFrame(test_data, columns = column_names)

print(training_data.head())
print(test_data.head())

training_data.to_csv('training.csv', index = False)
test_data.to_csv('test.csv', index = False)
training_objective.to_csv('trainingobjective.csv', index = False)
test_objective.to_csv('testobjective.csv', index = False)

print(training_data)
print(test_data)
print(training_objective)
print(test_objective)