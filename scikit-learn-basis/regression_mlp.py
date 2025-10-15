from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#fetching dataset
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50,50,50], max_iter=1000)
pipe = make_pipeline(StandardScaler(), mlp_reg)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_valid)

MSE = mean_squared_error(y_valid, y_pred)
print(MSE)