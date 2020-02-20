import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


csv_path = 'C:\\Users\\Naveed\\Desktop\\iris.csv'
names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
data = pd.read_csv(csv_path)
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

#Create a model and train it
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(X, y)

#Evaluate the model
p_pred = model.predict_proba(X)
y_pred = model.predict(X)
score_ = model.score(X, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
print(report)