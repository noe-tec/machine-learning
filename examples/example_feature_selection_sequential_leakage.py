"""Sequential feature selection -- incorrect approach (data leakage).

Uses SequentialFeatureSelector to keep half of the features, but fits the
selector on the entire dataset before evaluating on that same data. This
leaks information from the "test" portion into feature selection, producing
overly optimistic metrics. Compare against
example_feature_selection_sequential_cv.py for the correct approach.
"""
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector

diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

print("\n ----- Feature selection using 50% of predictors -----")

regr = linear_model.LinearRegression()
fselection = SequentialFeatureSelector(regr, n_features_to_select=0.5)
fselection.fit(x, y)
print("Selected features: ", fselection.get_feature_names_out())

x_transformed = fselection.transform(x)
regr.fit(x_transformed, y)
print("Model coefficients: ", regr.coef_)
print("Model intercept: ", regr.intercept_)

y_pred = regr.predict(x_transformed)
print("Evaluation using training data (not recommended): ")
print('MSE: ', mean_squared_error(y, y_pred))
print("MAE: ", mean_absolute_error(y, y_pred))
print("R^2: ", r2_score(y, y_pred))
