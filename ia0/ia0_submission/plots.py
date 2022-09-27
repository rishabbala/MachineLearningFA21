import pandas as pd
from pandas.core.algorithms import unique
import numpy as np
import matplotlib.pyplot as plt

## To plot the effect of encoded_date in place of date

FILE_PATH = 'pa0(train-only).csv'
data = pd.read_csv(FILE_PATH)
data.drop('id', inplace=True, axis=1)
data[['month','day', 'year']] = data.date.str.split("/",expand=True).astype('int32')
# data.drop('date', inplace=True, axis=1)
data['encoded_date'] = data['year'] + data['month']*100 + data['day']
# data.drop(['day', 'month', 'year'], inplace=True, axis=1)
print(data['encoded_date'])
data.plot.scatter(x='encoded_date', y='price')
# plt.gca().axes.set_xticks([])
plt.show()
