import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


csv_path = 'C:\\Users\\Naveed\\Desktop\\iris.csv'
names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
data = pd.read_csv(csv_path)
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

prstd, iv_l, iv_u = wls_prediction_std(results)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(X, y, 'o', label="data")
ax.plot(X, y, 'b-', label="True")
ax.plot(X, results.fittedvalues, 'r--.', label="OLS")
ax.plot(X, iv_u, 'r--')
ax.plot(X, iv_l, 'r--')
ax.legend(loc='best');

plt.show()