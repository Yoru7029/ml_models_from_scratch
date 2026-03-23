from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from linear_regression import model

# Load data
housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Train OLS models
sklearn_model = linear_model.LinearRegression().fit(X_train, y_train)
my_model = model.LinearRegression().fit(X_train, y_train)

# Score
print("Sklearn model:")
print("- R2 score:", sklearn_model.score(X_test, y_test))
print("- MSE:", mean_squared_error(y_test, sklearn_model.predict(X_test)))
print(f"- intercept: {sklearn_model.intercept_}")
print(f"- coefficients: {sklearn_model.coef_}\n")

print("My model (OLS):")
print("- R2 score:", sklearn_model.score(X_test, y_test))
print("- MSE:", mean_squared_error(y_test, my_model.predict(X_test)))
print(f"- intercept: {my_model.intercept_}")
print(f"- coefficients: {my_model.coef_}\n")

# Train SGD model
my_lin_sgd = model.SGDLinearRegression(tol=1e-4).fit(X_train, y_train)
print("My model (SGD):")
print("- R2 score:", my_lin_sgd.score(X_test, y_test))
print("- MSE:", mean_squared_error(y_test, my_lin_sgd.predict(X_test)))
print("- intercept:", my_lin_sgd.intercept_)
print("- coefficent:", my_lin_sgd.coef_)