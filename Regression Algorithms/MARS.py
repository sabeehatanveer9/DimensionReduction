import numpy
import pandas as pd
from pyearth import Earth
from matplotlib import pyplot


csv_path = 'C:\\Users\\Naveed\\Desktop\\iris.csv'
names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
data = pd.read_csv(csv_path)
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

#Fit an Earth model
model = Earth()
model.fit(X,y)

#Print the model
print(model.trace())
print(model.summary())

#Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X[:,6],y,'r.')
pyplot.plot(X[:,6],y_hat,'b.')
pyplot.xlabel('x_6')
pyplot.ylabel('y')
pyplot.title('Simple Earth Example')
pyplot.show()