import numpy as np
import matplotlib.pyplot as plt
import Data_Cleaning2
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up some sensible column names for our columns
column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

# Import the data
raw_data = pd.read_csv("Data_A.csv", names = column_names)
discriminator = Data_Cleaning2.add_distribution_statistics(raw_data)

# Import the data again to make sure it's not been altered by the above
raw_data = pd.read_csv("Data_A.csv", names = column_names)

stock_1 = raw_data[discriminator['overall mean'] > 550000]
stock_2 = raw_data[discriminator['overall mean'] <= 550000]

stock_1_labels = stock_1['midprice change direction']
stock_2_labels = stock_2['midprice change direction']

stock_1.drop(labels = ['midprice change direction'], axis = 1)
stock_1.drop(labels = ['midprice change direction'], axis = 1)

seed = 42
# Split the data into training and validation sets
stock_1_training_data, stock_1_test_data, stock_1_training_objective, stock_1_test_objective = train_test_split(stock_1
    , stock_1_labels, test_size=0.2, random_state=seed)

stock_2_training_data, stock_2_test_data, stock_2_training_objective, stock_2_test_objective = train_test_split(stock_2
    , stock_2_labels, test_size=0.2, random_state=seed)


# # Split off the objectives
# stock_1_training_objective = stock_1_training_data['midprice change direction']
# stock_1_test_objective = stock_1_test_data['midprice change direction']
stock_1_training_data = stock_1_training_data.drop(labels = ['midprice change direction'], axis = 1)
stock_1_test_data = stock_1_test_data.drop(labels = ['midprice change direction'], axis = 1)

# stock_2_training_objective = stock_2_training_data['midprice change direction']
# stock_2_test_objective = stock_2_test_data['midprice change direction']
stock_2_training_data = stock_2_training_data.drop(labels = ['midprice change direction'], axis = 1)
stock_2_test_data = stock_2_test_data.drop(labels = ['midprice change direction'], axis = 1)


scale = False

if not scale:
    stock_1_training_data = Data_Cleaning2.transform_data(stock_1_training_data, drop = True)
    stock_1_test_data = Data_Cleaning2.transform_data(stock_1_test_data, drop = True)

    stock_2_training_data = Data_Cleaning2.transform_data(stock_2_training_data, drop = True)
    stock_2_test_data = Data_Cleaning2.transform_data(stock_2_test_data, drop = True)



# Will have to fix this if I want to use it
if scale:
    column_names = stock_1_training_data.columns

    from sklearn.preprocessing import StandardScaler
    stock_1_scaler = StandardScaler()
    stock_1_training_data = stock_1_scaler.fit_transform(stock_1_training_data)
    stock_1_test_data = stock_1_scaler.transform(stock_1_test_data) 

    stock_1_training_data = pd.DataFrame(stock_1_training_data, columns = column_names)
    stock_1_test_data = pd.DataFrame(stock_1_test_data, columns = column_names)

    stock_2_scaler = StandardScaler()
    stock_2_training_data = stock_2_scaler.fit_transform(stock_2_training_data)
    stock_2_test_data = stock_2_scaler.transform(stock_2_test_data) 

    stock_2_training_data = pd.DataFrame(stock_2_training_data, columns = column_names)
    stock_2_test_data = pd.DataFrame(stock_2_test_data, columns = column_names)

print(stock_1_training_data.head())
print(stock_1_test_data.head())

print(stock_2_training_data.head())
print(stock_2_test_data.head())

stock_1_training_data.to_csv('stock_1_training.csv', index = False)
stock_1_test_data.to_csv('stock_1_test.csv', index = False)
stock_1_training_objective.to_csv('stock_1_trainingobjective.csv', index = False)
stock_1_test_objective.to_csv('stock_1_testobjective.csv', index = False)

print(stock_1_training_data)
print(stock_1_test_data)
print(stock_1_training_objective)
print(stock_1_test_objective)

stock_2_training_data.to_csv('stock_2_training.csv', index = False)
stock_2_test_data.to_csv('stock_2_test.csv', index = False)
stock_2_training_objective.to_csv('stock_2_trainingobjective.csv', index = False)
stock_2_test_objective.to_csv('stock_2_testobjective.csv', index = False)

print(stock_2_training_data)
print(stock_2_test_data)
print(stock_2_training_objective)
print(stock_2_test_objective)