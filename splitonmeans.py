import pandas as pd
import matplotlib.pyplot as plt
from Data_Cleaning2 import add_distribution_statistics, book_volume_statistics

# Set up some sensible column names for our columns
column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

# Import the data
raw_data = pd.read_csv("Data_A.csv", names = column_names)

training_data = raw_data

training_data = add_distribution_statistics(training_data)
training_data = book_volume_statistics(training_data)
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())



high_data = raw_data[raw_data['overall mean'] > 900000] # Use to predict on
middle_data = raw_data[raw_data['overall mean'] <= 900000]
upper_middle_data = middle_data[middle_data['overall mean'] > 795000] # Use to predict on
middle_middle_data = raw_data[raw_data['overall mean'] <= 795000] 
middle_middle_data = middle_middle_data[middle_middle_data['overall mean'] > 707500] # Use to predict on
middle_data = raw_data[raw_data['overall mean'] > 550000]
lower_middle_data = middle_data[middle_data['overall mean'] <= 707500] # Use to predict on
lower_data = raw_data[raw_data['overall mean'] <= 550000]
upper_lower_data = lower_data[lower_data['overall mean'] > 465000] # Use to predict on
lower_lower_data = lower_data[lower_data['overall mean'] <= 465000] # Use to predict on

print("High Data")
training_data = high_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())

print("Middle Data")
training_data = middle_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())


print("Upper Middle Data")
training_data = upper_middle_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())

print("Middle Middle Data")
training_data = middle_middle_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())


print("Lower Middle Data")
training_data = lower_middle_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())


print("Lower Data")
training_data = lower_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())


print("Upper Lower Data")
training_data = upper_lower_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())


print("Lower Lower Data")
training_data = lower_lower_data
training_objective = training_data['midprice change direction']



# Look at my beautiful skew, seems like this is a pretty good predictor!

plt.scatter(training_data['midprice change direction'], training_data['overall mean'], alpha = 0.01)
plt.grid()
plt.show()

#The Level 1 Skew statistic gives us 68% prediction accuracy! That's pretty good before any machine learning.
print((((training_data['level 1 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 2 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 3 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['level 4 volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())
print((((training_data['overall volume split'] > 0).astype(int) - training_objective) == 0).sum()/training_objective.count())


# This tells us pretty clearly that they operate in two modes... probably the first stock has a mean and prices above 550k, and the second below 550k
# It might be worth training the algorithm separately on each.