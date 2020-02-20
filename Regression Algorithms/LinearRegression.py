import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

csv_path = 'C:\\Users\\Naveed\\Desktop\\iris.csv'
names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
data = pd.read_csv(csv_path)
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values


print("Simple linear regression")
plt.figure(figsize=(16, 8))
plt.scatter(
    data['sepallength'],
    data['class'],
    c='black'
)
plt.xlabel("Parts of iris")
plt.ylabel("Class")
plt.show()

X = data['sepallength'].values.reshape(-1,1)
y = data['class'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(
    data['sepallength'],
    data['class'],
    c='black'
)
plt.plot(
    data['sepallength'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Parts of Iris")
plt.ylabel("class")
plt.show()

X = data['sepallength']
y = data['class']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

print("Multiple Linear Regression")
Xs = data.drop(['class'], axis=1)
y = data['class']
reg = LinearRegression()
reg.fit(Xs, y)
#print("The linear model is: Y = {:.5} + {:.5}*sepallength + {:.5}*sepalwidth + {:.5}*petallength".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))
X = np.column_stack((data['sepallength'], data['sepalwidth'], data['petallength']))
y = data['class']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())