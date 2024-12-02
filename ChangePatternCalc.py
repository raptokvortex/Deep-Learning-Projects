import pandas as pd

column_names = ['midprice change direction', 'limit order level 1 ask price', 'limit order level 1 ask volume', 'limit order level 1 bid price', 'limit order level 1 bid volume', 'limit order level 2 ask price', 'limit order level 2 ask volume', 'limit order level 2 bid price', 'limit order level 2 bid volume', 'limit order level 3 ask price', 'limit order level 3 ask volume', 'limit order level 3 bid price', 'limit order level 3 bid volume', 'limit order level 4 ask price', 'limit order level 4 ask volume', 'limit order level 4 bid price', 'limit order level 4 bid volume', 'previous midprice change 1', 'previous midprice change 2', 'previous midprice change 3', 'previous midprice change 4', 'previous midprice change 5']

raw_data = pd.read_csv("Data_A.csv", names = column_names)


raw_data['price change string'] = raw_data['previous midprice change 1'].astype(str) + raw_data['previous midprice change 2'].astype(str) + raw_data['previous midprice change 3'].astype(str) + raw_data['previous midprice change 4'].astype(str) + raw_data['previous midprice change 5'].astype(str)


pattern_chance = raw_data.groupby('price change string').mean()['midprice change direction']

pattern_chance.to_csv('changepatternchance.csv', index=True)


print(pattern_chance)