import tensorflow.keras as keras
import numpy as np
import pandas as pd

import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


# Import the preprocessed data

X_train = pd.read_csv('stock_2_training.csv')

Y_train = pd.read_csv('stock_2_trainingobjective.csv')

X_val = pd.read_csv('stock_2_test.csv')

Y_val = pd.read_csv('stock_2_testobjective.csv')

print(X_train.head())
print(Y_train.head())
print(X_val.head())
print(Y_val.head())

print(X_train.shape[1])
input("tap enter to continue")
# Define the model with hyperparameters
def build_model(hp):

    # Choose the number of layers (this is the hyperparameter)
    num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)  # Tune between 1 and 10 layers

    model = Sequential()
    model.add(InputLayer(input_shape = (X_train.shape[1],)))
    for i in range(num_layers):
        # Choose the number of neurons in each layer
        units = hp.Int('units_' + str(i), min_value=32, max_value=512, step=32)
        model.add(Dense(units=units, kernel_regularizer=l2(0.01), activation=hp.Choice('activation', ['relu', 'tanh', 'elu', 'swish'])))
        model.add(Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Define the tuner
tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, hyperband_iterations=2, directory='stock2', project_name='hyperparameter_tuning')



early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Search for the best hyperparameters
tuner.search(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]