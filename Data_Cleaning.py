#Cleaning the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

raw_data = pd.read_csv("Data_A.csv", names = column_names)

# print(raw_data.head())

training_data = raw_data.iloc[0:180000]
test_data = raw_data.iloc[180000:]


training_objective = training_data['midprice change direction']
test_objective = test_data['midprice change direction']

training_data = training_data.drop(labels = ['midprice change direction'], axis = 1)
test_data = test_data.drop(labels = ['midprice change direction'], axis = 1)

# print(training_data)
# print(test_data)

# print(training_objective)
# print(test_objective)

# Are there any NaN values in the data:
# Otherwise let's try to normalise the data
#Then let's see if we can do some PCA to remove noise in the data, which should make our training more robust

# How can we normalise the prices. Probably want to look at the total demand for each stock or sth like that.

# Let's look at a single row for now:

#print(raw_data.iloc[0]) # Mid point price goes, down, up, down up, down and then the value we would want to predict is up
#print(raw_data.iloc[1]) # Mid point price goes, up, up, up, down, down, and then predicted value is down
#print(raw_data.iloc[2]) # Mid point price goes down, up, down, up, down and then predicted value is up

# A starting guess is that just mean reversion (based on stylistic facts) might occur. Therefore we could try this as a method of prediction.
# If the sum is 5, we would guess the next would be 0. 4, 0 but a bit less, 0 and we would guess 1 pretty strongly. So we'll use a formula of (5 - sum)/5. If the sum is 5
# We are saying very non rigorously, that there is a chance of 1 of the asset going up
def mean_reversion(training_data):
    training_data['mean reversion'] = (5 - training_data[['previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']].sum(axis = 1))/5
    return training_data

training_data = mean_reversion(training_data)
# print(training_data.head())
# print(training_objective.head())
# print((((training_data['mean reversion'] > 0.5) - training_objective) == 0).sum()/180000) # Slightly better than chance? So we can probably leave it in

# What about pattern reversion
# We will try reversing the last value seen. Which is probably midpoint price 1, but this isn't clear so we'll test with each.
# print(((1 - training_data['previous midprice change 1'] - training_objective == 0)).sum()/180000)
# print(((1 - training_data['previous midprice change 2'] - training_objective == 0)).sum()/180000)
# print(((1 - training_data['previous midprice change 3'] - training_objective == 0)).sum()/180000)
# print(((1 - training_data['previous midprice change 4'] - training_objective == 0)).sum()/180000)
# print(((1 - training_data['previous midprice change 5'] - training_objective == 0)).sum()/180000)

# Sum seems to alternate... which is a bit weird

# Let's try doing an alternating sum and see if that gives some predictive power

def alternating_sum(training_data):
    training_data['alternating sum weight'] = (((1 - training_data['previous midprice change 1']) + training_data['previous midprice change 2'] + 1 - training_data['previous midprice change 3'] + training_data['previous midprice change 4'] + 1 - training_data['previous midprice change 5']))/5
    return training_data
# print((((training_data['alternating sum weight'] > 0.5) - training_objective) == 0).sum()/(training_data['alternating sum weight']).count())

# Not really better beyond what we saw already

# Let's try all of the sequences and see what we get
# training_data['price change string'] = training_data['previous midprice change 1'].astype(str) + training_data['previous midprice change 2'].astype(str) + training_data['previous midprice change 3'].astype(str) + training_data['previous midprice change 4'].astype(str) + training_data['previous midprice change 5'].astype(str)

# print(training_data.head())
def midprice_change_direction(training_data):
    column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

    raw_data = pd.read_csv("Data_A.csv", names = column_names)


    raw_data['price change string'] = raw_data['previous midprice change 1'].astype(str) + raw_data['previous midprice change 2'].astype(str) + raw_data['previous midprice change 3'].astype(str) + raw_data['previous midprice change 4'].astype(str) + raw_data['previous midprice change 5'].astype(str)

    # print(raw_data.head())

    pattern_chance = raw_data.groupby('price change string').mean()['midprice change direction']

    # print(pattern_chance)

    training_data['midprice change string'] = training_data['previous midprice change 1'].astype(str) + training_data['previous midprice change 2'].astype(str) + training_data['previous midprice change 3'].astype(str) + training_data['previous midprice change 4'].astype(str) + training_data['previous midprice change 5'].astype(str)

    training_data['midprice change prediction'] = 0.0001

    # for i in range(0,180000):
    #     if pattern_chance[training_data['price change string'][i]] > 0.5:
    #         training_data.at[i, 'price change prediction'] = 1

    # print((((training_data['price change prediction'] - training_objective) == 0).sum()/(training_data['price change prediction']).count()))

    # That's actually not bad. We have got 58% predictive accuracy on the training set from nothing but the midpoint price movements.


    # So we add to our training data our useful predicted chance based on this, and drop the strings, which are non-numeric (since the chances for each of the strings is unique, this also serves as an encoding of the pattern)
    for i in range(0,180000):
        training_data.at[i, 'midprice change prediction'] = pattern_chance[training_data['midprice change string'][i]]

    # print(training_data.head())

    training_data = training_data.drop('midprice change string', axis = 1)

    #print(training_data.head())

    return training_data

training_data = midprice_change_direction(training_data)


# Probably want to add in columns for relative prices, relative volumes, and relative demands

# Then maybe for overall volumes, and overall prices
# def add_net_demand(training_data):
#     training_data['level 1 net demand'] = training_data['limit order level 1 bid price']*training_data['limit order level 1 bid volume'] - training_data['limit order level 1 ask price']*training_data['limit order level 1 ask volume']
#     training_data['level 2 net demand'] = training_data['limit order level 2 bid price']*training_data['limit order level 2 bid volume'] - training_data['limit order level 2 ask price']*training_data['limit order level 2 ask volume']
#     training_data['level 3 net demand'] = training_data['limit order level 3 bid price']*training_data['limit order level 3 bid volume'] - training_data['limit order level 3 ask price']*training_data['limit order level 3 ask volume']
#     training_data['level 4 net demand'] = training_data['limit order level 4 bid price']*training_data['limit order level 4 bid volume'] - training_data['limit order level 4 ask price']*training_data['limit order level 4 ask volume']
#     training_data['overall net demand'] = training_data['level 1 net demand'] + training_data['level 2 net demand'] + training_data['level 3 net demand'] + training_data['level 4 net demand']
#     return training_data

# The features we made for this don't seem that useful, so let's just remove them
# add_net_demand(training_data)

# print(training_data.head())

# Not much really going on below... maybe a little bit
# print(((training_data['level 1 net demand'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['level 2 net demand'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['level 3 net demand'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['level 4 net demand'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['overall net demand'] > 0 - training_objective) == 0).sum()/training_objective.count())

# # Let's try and look at net volumes

# def add_net_volumes(training_data):
#     training_data['level 1 net volume'] = training_data['limit order level 1 bid volume'] - training_data['limit order level 1 ask volume']

#     training_data['level 2 net volume'] = training_data['limit order level 2 bid volume'] - training_data['limit order level 2 ask volume']

#     training_data['level 3 net volume'] = training_data['limit order level 3 bid volume'] - training_data['limit order level 3 ask volume']

#     training_data['level 4 net volume'] = training_data['limit order level 4 bid volume'] - training_data['limit order level 4 ask volume']

#     training_data['overall net volume'] = training_data['level 1 net volume'] + training_data['level 2 net volume'] + training_data['level 3 net volume'] + training_data['level 4 net volume']
#     return training_data

# add_net_volumes(training_data)

# print(training_data.head())

# # Not much really going on below at all
# print(((training_data['level 1 net volume'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['level 2 net volume'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['level 3 net volume'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['level 4 net volume'] > 0 - training_objective) == 0).sum()/training_objective.count())
# print(((training_data['overall net volume'] > 0 - training_objective) == 0).sum()/training_objective.count())


# # Finally let's do with net prices
# def add_net_prices(training_data):
#     training_data['level 1 net price'] = training_data['limit order level 1 bid price'] - training_data['limit order level 1 ask price']

#     training_data['level 2 net price'] = training_data['limit order level 2 bid price'] - training_data['limit order level 2 ask price']

#     training_data['level 3 net price'] = training_data['limit order level 3 bid price'] - training_data['limit order level 3 ask price']

#     training_data['level 4 net price'] = training_data['limit order level 4 bid price'] - training_data['limit order level 4 ask price']

#     training_data['overall net price'] = training_data['level 1 net price'] + training_data['level 2 net price'] + training_data['level 3 net price'] + training_data['level 4 net price']
#     return training_data

# add_net_prices(training_data)

# print(training_data.head())

# No predictive power at all... obviously the bid ask spread exists so the bid is always going to be lower than the ask. # If we buy and then sell of course we should lose money
# print((((training_data['level 1 net price'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['level 2 net price'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['level 3 net price'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['level 4 net price'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['overall net price'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())

# # Maybe there is an order level at which the bid exceeds ask...
# print(training_data.iloc[0]['limit order level 1 ask price'])
# print(training_data.iloc[0]['limit order level 2 ask price'])
# print(training_data.iloc[0]['limit order level 3 ask price'])
# print(training_data.iloc[0]['limit order level 4 ask price'])
# print(training_data.iloc[0]['limit order level 1 bid price'])
# print(training_data.iloc[0]['limit order level 2 bid price'])
# print(training_data.iloc[0]['limit order level 3 bid price'])
# print(training_data.iloc[0]['limit order level 4 bid price'])

# Nope they get further away from each other... obviously... but mmaybe we could compute the skew and variance?
def add_distribution_statistics(training_data):
    
    # Computing totals
    training_data['level 1 total'] = training_data['limit order level 1 bid price']*training_data['limit order level 1 bid volume'] + training_data['limit order level 1 ask price']*training_data['limit order level 1 ask volume']
    training_data['level 2 total'] = training_data['limit order level 2 bid price']*training_data['limit order level 2 bid volume'] + training_data['limit order level 2 ask price']*training_data['limit order level 2 ask volume']
    training_data['level 3 total'] = training_data['limit order level 3 bid price']*training_data['limit order level 3 bid volume'] + training_data['limit order level 3 ask price']*training_data['limit order level 3 ask volume']
    training_data['level 4 total'] = training_data['limit order level 4 bid price']*training_data['limit order level 4 bid volume'] + training_data['limit order level 4 ask price']*training_data['limit order level 4 ask volume']
    training_data['overall total'] = training_data['level 1 total'] + training_data['level 2 total'] + training_data['level 3 total'] + training_data['level 4 total']
    
    # Computing volumes
    training_data['level 1 volume'] = (training_data['limit order level 1 bid volume'] + training_data['limit order level 1 ask volume'])
    training_data['level 2 volume'] = (training_data['limit order level 2 bid volume'] + training_data['limit order level 2 ask volume'])
    training_data['level 3 volume'] = (training_data['limit order level 3 bid volume'] + training_data['limit order level 3 ask volume'])
    training_data['level 4 volume'] = (training_data['limit order level 4 bid volume'] + training_data['limit order level 4 ask volume'])
    training_data['overall volume'] = training_data['level 1 volume'] + training_data['level 2 volume'] + training_data['level 3 volume'] + training_data['level 4 volume']

    # Computing means
    training_data['level 1 mean'] = training_data['level 1 total'] / training_data['level 1 volume']
    training_data['level 2 mean'] = training_data['level 2 total'] / training_data['level 2 volume']
    training_data['level 3 mean'] = training_data['level 3 total'] / training_data['level 3 volume']
    training_data['level 4 mean'] = training_data['level 4 total'] / training_data['level 4 volume']
    training_data['overall mean'] = training_data['overall total'] / training_data['overall volume']

    # Computing Variances
    training_data['level 1 variance'] = ((training_data['limit order level 1 bid price'] - training_data['level 1 mean'])**2 * training_data['limit order level 1 bid volume'] + (training_data['limit order level 1 ask price'] - training_data['level 1 mean'])**2 * training_data['limit order level 1 ask volume'] )/training_data['level 1 volume']
    training_data['level 2 variance'] = ((training_data['limit order level 2 bid price'] - training_data['level 2 mean'])**2 * training_data['limit order level 2 bid volume'] + (training_data['limit order level 2 ask price'] - training_data['level 2 mean'])**2 * training_data['limit order level 2 ask volume'] )/training_data['level 2 volume']
    training_data['level 3 variance'] = ((training_data['limit order level 3 bid price'] - training_data['level 3 mean'])**2 * training_data['limit order level 3 bid volume'] + (training_data['limit order level 3 ask price'] - training_data['level 3 mean'])**2 * training_data['limit order level 3 ask volume'] )/training_data['level 3 volume']
    training_data['level 4 variance'] = ((training_data['limit order level 4 bid price'] - training_data['level 4 mean'])**2 * training_data['limit order level 4 bid volume'] + (training_data['limit order level 4 ask price'] - training_data['level 4 mean'])**2 * training_data['limit order level 4 ask volume'] )/training_data['level 4 volume']
    training_data['overall variance'] = (training_data['level 1 variance']*training_data['level 1 volume'] + training_data['level 2 variance']*training_data['level 2 volume'] + training_data['level 3 variance']*training_data['level 3 volume'] + training_data['level 4 variance']*training_data['level 4 volume'])/training_data['overall volume']

    # Computing Skews
    training_data['level 1 skew'] = ((training_data['limit order level 1 bid price'] - training_data['level 1 mean'])**3 * training_data['limit order level 1 bid volume'] + (training_data['limit order level 1 ask price'] - training_data['level 1 mean'])**3 * training_data['limit order level 1 ask volume'] )/training_data['level 1 volume']
    training_data['level 2 skew'] = ((training_data['limit order level 2 bid price'] - training_data['level 2 mean'])**3 * training_data['limit order level 2 bid volume'] + (training_data['limit order level 2 ask price'] - training_data['level 2 mean'])**3 * training_data['limit order level 2 ask volume'] )/training_data['level 2 volume']
    training_data['level 3 skew'] = ((training_data['limit order level 3 bid price'] - training_data['level 3 mean'])**3 * training_data['limit order level 3 bid volume'] + (training_data['limit order level 3 ask price'] - training_data['level 3 mean'])**3 * training_data['limit order level 3 ask volume'] )/training_data['level 3 volume']
    training_data['level 4 skew'] = ((training_data['limit order level 4 bid price'] - training_data['level 4 mean'])**3 * training_data['limit order level 4 bid volume'] + (training_data['limit order level 4 ask price'] - training_data['level 4 mean'])**3 * training_data['limit order level 4 ask volume'] )/training_data['level 4 volume']
    training_data['overall skew'] = (training_data['level 1 skew']*training_data['level 1 volume'] + training_data['level 2 skew']*training_data['level 2 volume'] + training_data['level 3 skew']*training_data['level 3 volume'] + training_data['level 4 skew']*training_data['level 4 volume'])/training_data['overall volume']

    training_data['level 1 skew'] = training_data['level 1 skew'] / (np.sqrt(training_data['level 1 variance'])**3)
    training_data['level 2 skew'] = training_data['level 2 skew'] / (np.sqrt(training_data['level 2 variance'])**3)
    training_data['level 3 skew'] = training_data['level 3 skew'] / (np.sqrt(training_data['level 3 variance'])**3)
    training_data['level 4 skew'] = training_data['level 4 skew'] / (np.sqrt(training_data['level 4 variance'])**3)
    training_data['overall skew'] = training_data['overall skew'] / (np.sqrt(training_data['overall variance'])**3)

    # Computing Kurtosis - because why not
    training_data['level 1 kurtosis'] = ((training_data['limit order level 1 bid price'] - training_data['level 1 mean'])**4 * training_data['limit order level 1 bid volume'] + (training_data['limit order level 1 ask price'] - training_data['level 1 mean'])**4 * training_data['limit order level 1 ask volume'] )/training_data['level 1 volume']
    training_data['level 2 kurtosis'] = ((training_data['limit order level 2 bid price'] - training_data['level 2 mean'])**4 * training_data['limit order level 2 bid volume'] + (training_data['limit order level 2 ask price'] - training_data['level 2 mean'])**4 * training_data['limit order level 2 ask volume'] )/training_data['level 2 volume']
    training_data['level 3 kurtosis'] = ((training_data['limit order level 3 bid price'] - training_data['level 3 mean'])**4 * training_data['limit order level 3 bid volume'] + (training_data['limit order level 3 ask price'] - training_data['level 3 mean'])**4 * training_data['limit order level 3 ask volume'] )/training_data['level 3 volume']
    training_data['level 4 kurtosis'] = ((training_data['limit order level 4 bid price'] - training_data['level 4 mean'])**4 * training_data['limit order level 4 bid volume'] + (training_data['limit order level 4 ask price'] - training_data['level 4 mean'])**4 * training_data['limit order level 4 ask volume'] )/training_data['level 4 volume']
    training_data['overall kurtosis'] = (training_data['level 1 kurtosis']*training_data['level 1 volume'] + training_data['level 2 kurtosis']*training_data['level 2 volume'] + training_data['level 3 kurtosis']*training_data['level 3 volume'] + training_data['level 4 kurtosis']*training_data['level 4 volume'])/training_data['overall volume']

    training_data['level 1 kurtosis'] = training_data['level 1 kurtosis'] / (np.sqrt(training_data['level 1 variance'])**4)
    training_data['level 2 kurtosis'] = training_data['level 2 kurtosis'] / (np.sqrt(training_data['level 2 variance'])**4)
    training_data['level 3 kurtosis'] = training_data['level 3 kurtosis'] / (np.sqrt(training_data['level 3 variance'])**4)
    training_data['level 4 kurtosis'] = training_data['level 4 kurtosis'] / (np.sqrt(training_data['level 4 variance'])**4)
    training_data['overall kurtosis'] = training_data['overall kurtosis'] / (np.sqrt(training_data['overall variance'])**4)

    return training_data



raw_data = add_distribution_statistics(raw_data)

# print(raw_data['level 2 skew'])

# # Look at my beautiful skew, seems like this is a pretty good predictor!

# plt.scatter(raw_data['midprice change direction'], raw_data['overall mean'], alpha = 0.01)
# plt.grid()
# plt.show()


training_data = add_distribution_statistics(training_data)

# The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
# print((((training_data['level 1 skew'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['level 2 skew'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['level 3 skew'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['level 4 skew'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
# print((((training_data['overall skew'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())

# Kurtosis didn't really have any noticeable differences, probably can leave it out


# We can also use the statistics we computed to normalise the data
def normalise_data(training_data):
    # Normalise prices, on some distribution
    training_data['limit order level 1 ask price'] = (training_data['limit order level 1 ask price'] - training_data['overall mean'])/training_data['overall variance']
    training_data['limit order level 2 ask price'] = (training_data['limit order level 2 ask price'] - training_data['overall mean'])/training_data['overall variance']
    training_data['limit order level 3 ask price'] = (training_data['limit order level 3 ask price'] - training_data['overall mean'])/training_data['overall variance']
    training_data['limit order level 4 ask price'] = (training_data['limit order level 4 ask price'] - training_data['overall mean'])/training_data['overall variance']
    
    training_data['limit order level 1 bid price'] = (training_data['limit order level 1 bid price'] - training_data['overall mean'])/training_data['overall variance']
    training_data['limit order level 2 bid price'] = (training_data['limit order level 2 bid price'] - training_data['overall mean'])/training_data['overall variance']
    training_data['limit order level 3 bid price'] = (training_data['limit order level 3 bid price'] - training_data['overall mean'])/training_data['overall variance']
    training_data['limit order level 4 bid price'] = (training_data['limit order level 4 bid price'] - training_data['overall mean'])/training_data['overall variance']



    # Also need to normalise volumes, will just do over the total volume, so they represent proportions in some sense

    training_data['limit order level 1 ask volume'] = training_data['limit order level 1 ask volume'] / training_data['overall volume']
    training_data['limit order level 2 ask volume'] = training_data['limit order level 2 ask volume'] / training_data['overall volume']
    training_data['limit order level 3 ask volume'] = training_data['limit order level 3 ask volume'] / training_data['overall volume']
    training_data['limit order level 4 ask volume'] = training_data['limit order level 4 ask volume'] / training_data['overall volume']

    training_data['limit order level 1 bid volume'] = training_data['limit order level 1 bid volume'] / training_data['overall volume']
    training_data['limit order level 2 bid volume'] = training_data['limit order level 2 bid volume'] / training_data['overall volume']
    training_data['limit order level 3 bid volume'] = training_data['limit order level 3 bid volume'] / training_data['overall volume']
    training_data['limit order level 4 bid volume'] = training_data['limit order level 4 bid volume'] / training_data['overall volume']

    # Now we'll just normalise everything that is pretty big (or drop it depending on how we feel)
    if True:
        # Totals
        training_data['level 1 total'] = training_data['level 1 total']/training_data['level 1 total'].abs().max()
        training_data['level 2 total'] = training_data['level 2 total']/training_data['level 2 total'].abs().max()
        training_data['level 3 total'] = training_data['level 3 total']/training_data['level 3 total'].abs().max()
        training_data['level 4 total'] = training_data['level 4 total']/training_data['level 4 total'].abs().max()
        training_data['overall total'] = training_data['overall total']/training_data['overall total'].abs().max()

        # Volumes
        training_data['level 1 volume'] = training_data['level 1 volume']/training_data['level 1 volume'].abs().max()
        training_data['level 2 volume'] = training_data['level 2 volume']/training_data['level 2 volume'].abs().max()
        training_data['level 3 volume'] = training_data['level 3 volume']/training_data['level 3 volume'].abs().max()
        training_data['level 4 volume'] = training_data['level 4 volume']/training_data['level 4 volume'].abs().max()
        training_data['overall volume'] = training_data['overall volume']/training_data['overall volume'].abs().max()

        # Means
        training_data['level 1 mean'] = training_data['level 1 mean']/training_data['level 1 mean'].abs().max()
        training_data['level 2 mean'] = training_data['level 2 mean']/training_data['level 2 mean'].abs().max()
        training_data['level 3 mean'] = training_data['level 3 mean']/training_data['level 3 mean'].abs().max()
        training_data['level 4 mean'] = training_data['level 4 mean']/training_data['level 4 mean'].abs().max()
        training_data['overall mean'] = training_data['overall mean']/training_data['overall mean'].abs().max()

        # Variance
        training_data['level 1 variance'] = training_data['level 1 variance']/training_data['level 1 variance'].abs().max()
        training_data['level 2 variance'] = training_data['level 2 variance']/training_data['level 2 variance'].abs().max()
        training_data['level 3 variance'] = training_data['level 3 variance']/training_data['level 3 variance'].abs().max()
        training_data['level 4 variance'] = training_data['level 4 variance']/training_data['level 4 variance'].abs().max()
        training_data['overall variance'] = training_data['overall variance']/training_data['overall variance'].abs().max()

        # I feel like Kurtosis could also be pretty large...
        training_data['level 1 kurtosis'] = training_data['level 1 kurtosis']/training_data['level 1 kurtosis'].abs().max()
        training_data['level 2 kurtosis'] = training_data['level 2 kurtosis']/training_data['level 2 kurtosis'].abs().max()
        training_data['level 3 kurtosis'] = training_data['level 3 kurtosis']/training_data['level 3 kurtosis'].abs().max()
        training_data['level 4 kurtosis'] = training_data['level 4 kurtosis']/training_data['level 4 kurtosis'].abs().max()
        training_data['overall kurtosis'] = training_data['overall kurtosis']/training_data['overall kurtosis'].abs().max()

    return training_data

def drop_stuff(training_data):
    if True:
        # Totals
        training_data = training_data.drop('level 1 total', axis = 1)
        training_data = training_data.drop('level 2 total', axis = 1)
        training_data = training_data.drop('level 3 total', axis = 1)
        training_data = training_data.drop('level 4 total', axis = 1)
        training_data = training_data.drop('overall total', axis = 1)

        # Volumes
        training_data = training_data.drop('level 1 volume', axis = 1)
        training_data = training_data.drop('level 2 volume', axis = 1)
        training_data = training_data.drop('level 3 volume', axis = 1)
        training_data = training_data.drop('level 4 volume', axis = 1)
        training_data = training_data.drop('overall volume', axis = 1)
 

        # Means
        training_data = training_data.drop('level 1 mean', axis = 1)
        training_data = training_data.drop('level 2 mean', axis = 1)
        training_data = training_data.drop('level 3 mean', axis = 1)
        training_data = training_data.drop('level 4 mean', axis = 1)
        training_data = training_data.drop('overall mean', axis = 1)

        # Variance
        training_data = training_data.drop('level 1 variance', axis = 1)
        training_data = training_data.drop('level 2 variance', axis = 1)
        training_data = training_data.drop('level 3 variance', axis = 1)
        training_data = training_data.drop('level 4 variance', axis = 1)
        training_data = training_data.drop('overall variance', axis = 1)

        # I feel like Kurtosis could also be pretty large...
        training_data = training_data.drop('level 1 kurtosis', axis = 1)
        training_data = training_data.drop('level 2 kurtosis', axis = 1)
        training_data = training_data.drop('level 3 kurtosis', axis = 1)
        training_data = training_data.drop('level 4 kurtosis', axis = 1)
        training_data = training_data.drop('overall kurtosis', axis = 1)



    return training_data

training_data = normalise_data(training_data)

def transform_data(training_data, drop = False):
    training_data = mean_reversion(training_data)
    training_data = alternating_sum(training_data)
    training_data = midprice_change_direction(training_data)
    training_data = add_distribution_statistics(training_data)
    training_data = normalise_data(training_data)
    if drop:
        training_data = drop_stuff(training_data)

    return training_data

print(training_data.head())
print(training_data.iloc[0])