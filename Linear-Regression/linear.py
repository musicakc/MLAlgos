import pandas as pd
import matplotlib.pyplot as plt

#reading data
data = pd.read_csv('Advertising.csv', index_col=0)
data.head()

data.shape
