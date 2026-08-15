"""Sequential feature selection -- correct approach (cross-validation).

Fits SequentialFeatureSelector only on the training fold in each iteration
of K-fold cross-validation, and evaluates on the held-out fold. This avoids
the data leakage present in example_feature_selection_sequential_leakage.py.
"""
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector

diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

mse_cv = []
mae_cv = []
r2_cv = []

kf = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]

    regr_cv = linear_model.LinearRegression()

    fselection_cv = SequentialFeatureSelector(regr_cv, n_features_to_select=0.5)
    fselection_cv.fit(x_train, y_train)
    x_train = fselection_cv.transform(x_train)

    regr_cv.fit(x_train, y_train)

    x_test = fselection_cv.transform(x[test_index, :])
    y_test = y[test_index]
    y_pred = regr_cv.predict(x_test)

    mse_cv.append(mean_squared_error(y_test, y_pred))
    mae_cv.append(mean_absolute_error(y_test, y_pred))
    r2_cv.append(r2_score(y_test, y_pred))

print("Evaluation using cross-validation (recommended): ")
print('MSE:', np.average(mse_cv), '  MAE:', np.average(mae_cv), '  R^2:', np.average(r2_cv))
