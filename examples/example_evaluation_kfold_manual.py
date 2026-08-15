"""K-fold cross-validation, implemented manually.

Splits the data into 5 folds and, in a manual loop, trains on 4 folds and
evaluates on the remaining fold each time. Averages the metrics across all
folds to get a more robust performance estimate than a single hold-out split.
"""
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)
print(f"# features: {n_features}")

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

mse = 0
mae = 0
r2 = 0
for k, (train_index, test_index) in enumerate(kf.split(x)):
    print(f'Iteración de k-fold: {k + 1}')
    x_train = x[train_index, :]
    y_train = y[train_index]

    regr_cv = LinearRegression()
    regr_cv.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = regr_cv.predict(x_test)

    mse_i = mean_squared_error(y_test, y_pred)
    print('\t mse = ', mse_i)

    mae_i = mean_absolute_error(y_test, y_pred)
    print('\t mae = ', mae_i)

    r2_i = r2_score(y_test, y_pred)
    print('\t r^2= ', r2_i)

    mse += mse_i
    mae += mae_i
    r2 += r2_i

print("\nMetricas promedio: \n")

mse = mse / n_folds
print('MSE = ', mse)

mae = mae / n_folds
print('MAE = ', mae)

r2 = r2 / n_folds
print('R^2 = ', r2)
