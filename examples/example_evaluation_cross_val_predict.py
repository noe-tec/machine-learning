"""K-fold cross-validation using scikit-learn's cross_val_predict.

Instead of returning per-fold scores like cross_validate, cross_val_predict
returns one out-of-fold prediction per sample, which can then be scored
directly against the true target values.
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)
print(f"# features: {n_features}")

regr = LinearRegression()
y_pred = cross_val_predict(regr, x, y, cv=5)

print("y_pred shape: ", y_pred.shape)

print('mse = ', mean_squared_error(y, y_pred))
print('mae = ', mean_absolute_error(y, y_pred))
print('r^2= ', r2_score(y, y_pred))
