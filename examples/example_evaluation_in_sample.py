"""In-sample evaluation.

Trains and evaluates a linear regression model using the same data for both
training and evaluation. This overestimates performance, since the model has
already seen every point it is being scored on -- it is not a valid estimate
of how well the model generalizes to new data.
"""
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

print(f"# features: {n_features}")

regr = LinearRegression()
regr.fit(x, y)

print("Coeficientes del modelo: \n", regr.coef_)

y_pred = regr.predict(x)
print('MSE: \n', mean_squared_error(y, y_pred))
print('MAE: \n', mean_absolute_error(y, y_pred))
print("R^2: \n", r2_score(y, y_pred))
