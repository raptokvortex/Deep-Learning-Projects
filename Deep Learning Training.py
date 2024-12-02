import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import Data_Cleaning
plt.style.use('ggplot')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up some sensible column names for our columns
column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

# Import the data
raw_data = pd.read_csv("Data_A.csv", names = column_names)

# Split the data into training and test sets
training_data = raw_data.iloc[0:180000]
test_data = raw_data.iloc[180000:]

# Split off the objectives
training_objective = training_data['midprice change direction']
test_objective = test_data['midprice change direction']
training_data = training_data.drop(labels = ['midprice change direction'], axis = 1)
test_data = test_data.drop(labels = ['midprice change direction'], axis = 1)

training_data = Data_Cleaning.transform_data(training_data, drop = True)

print(training_data.head())

X = training_data.values

Y = training_objective.values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = scaler.transform(X) 



# We'll try some hyper parameter optimisation
max_layers = 10
min_layers = 3
widths = 10
max_drop_layer_chance = 0.2
layers = [i for i in range(min_layers, max_layers + 1)]
learning_rates = [0.1**i for i in range(2,4)]

rng = np.random.default_rng(seed=42)

widths = rng.integers(low = 5, high = 400, size = (widths, max_layers))


for layer in layers:
    for eta in learning_rates:

        
        for width in widths:
            model = keras.Sequential()
            model.add(keras.layers.InputLayer(input_shape=(29,)))
            for i in range(1, layer + 1):
                model.add(keras.layers.Dense(width[i], activation = 'elu'))
                model.add(keras.layers.Dropout(max_drop_layer_chance))


            model.add(keras.layers.Dense(1, activation = 'sigmoid'))

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=eta), loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X, Y, batch_size= 300, epochs=5)
            print(layer, width, eta)
