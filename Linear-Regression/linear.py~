import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#reading data
data = pd.read_csv('Advertising.csv', index_col=0)
data.head()

data.shape

fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16,8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
#plt.show

lm = smf.ols(formula='Sales~TV', data=data).fit()
lm.params

X_new = pd.DataFrame({'TV':[data.TV.min(), data.TV.max()]})
X_new.head()

preds = lm.predict(X_new)
#print preds

data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, preds, c='red', linewidth=2)

print lm.conf_int()
print lm.pvalues
print lm.rsquared
