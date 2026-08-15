"""K-fold cross-validation using scikit-learn's cross_validate.

Same idea as the manual K-fold loop, but automated with scikit-learn's
cross_validate function, which handles the fold splitting, training and
scoring internally.
"""
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)
print(f"# features: {n_features}")

regr = LinearRegression()
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
cv_results = cross_validate(regr, x, y, cv=5, scoring=scoring)

# cross_validate returns negated error scores, so we flip the sign back
mse = -cv_results['test_neg_mean_squared_error'].mean()
mae = -cv_results['test_neg_mean_absolute_error'].mean()
r2 = cv_results['test_r2'].mean()

print("\nMetricas promedio: \n")
print('MSE = ', mse)
print('MAE = ', mae)
print('R^2 = ', r2)
