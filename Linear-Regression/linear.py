'''
Based on tutorial given here ['http://www.dataschool.io/linear-regression-in-python/']
'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#reading data
data = pd.read_csv('Advertising.csv', index_col=0)
data.head()
data.shape

#Visualise by making scatterplot
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16,8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

#fitted model created
lm = smf.ols(formula='Sales~TV', data=data).fit()

#for coefficients
print lm.params

#new Dataframe created for statsmodel formula and plot least squares line
X_new = pd.DataFrame({'TV':[data.TV.min(), data.TV.max()]})
X_new.head()

#make predictions on a new value
preds = lm.predict(X_new)
print preds

#plot data observed
data.plot(kind='scatter', x='TV', y='Sales')

#plot least squares line
plt.plot(X_new, preds, c='red', linewidth=2)

#print confidence intervals for model coefficients
print lm.conf_int()

#print p values for model coefficients
print lm.pvalues

#print R squared value for the model
print lm.rsquared
