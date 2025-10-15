from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as np
import numpy as np

iris = load_iris()

X, y = iris.data, iris.target

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

mlp_class = MLPClassifier(hidden_layer_sizes=[10, 10, 10, 10], max_iter=1000, random_state=42)

pipe = make_pipeline(StandardScaler(), mlp_class)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print(mse) #0.036

#with 4 hidden layers of 10 neurons each, the MLP captures almost perfectly the relationship
#testing the test set to see if it's similar

y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse) #0.053
