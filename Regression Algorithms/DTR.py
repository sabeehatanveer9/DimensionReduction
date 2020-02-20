import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

csv_path = 'C:\\Users\\Naveed\\Desktop\\position_salaries.csv'
names = ['Position', 'Level', 'Salary']
data = pd.read_csv(csv_path)
X = data.iloc[:,1:2].values
y = data.iloc[:,2:3].values

# It doesn't need to split the dataset because we have a small dataset
# Fitting the Decision Tree Regression Model to the dataset
#DecisionTreeRegressor class has many parameters. Input only #random_state=0 or 42.
regressor = DecisionTreeRegressor(random_state=0)
#Fit the regressor object to the dataset.
regressor.fit(X,y)
#Visualising the Decision Tree Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Check It (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()