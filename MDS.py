import pandas as pd
from sklearn.datasets import load_digits
from sklearn.manifold import MDS


csv_path = 'C:\\Users\\Naveed\\Desktop\\iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv( csv_path)
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#X, _ = load_digits(return_X_y=True)
print(X.shape)
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(X[:100])
print(X_transformed.shape)
