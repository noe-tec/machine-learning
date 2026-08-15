"""Hold-out validation.

Splits the diabetes dataset into a train set and a test set, trains on the
train set, and evaluates on the held-out test set. Gives a more realistic
(and typically lower) performance estimate than in-sample evaluation.
"""
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)
print(f"# features: {n_features}")

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Intercepto:", model.intercept_)
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f}")
print('-' * 10)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
